import numpy as _np
import scipy.linalg as _sal
from ..._base_algorithm import _Algorithm
from ._convert import Raw2Oxy
from ._dl_sqi import SignalQualityDeepLearning
import xarray as _xr

from ... import scheduler

def load_xrnirs(file):
    nirs = _xr.load_dataset(file)
    attrs = nirs.p.main_signal.attrs
    todel=[]
    for k in attrs.keys():
        if k.endswith('_shape'):
            attr_name = k.split('_shape')[0]
            attr_numpy = attrs[attr_name]
            attr_numpy = attr_numpy.reshape(attrs[k])
            attrs[attr_name] = attr_numpy
            todel.append(k)
    for k in todel:
        del attrs[k]
    
    nirs.p.main_signal.attrs = attrs
    nirs.attrs['history'] = [nirs.attrs['history']]
    return(nirs)

def SDto1darray(nirs):
    for k in nirs.keys():
        SD = nirs[k].p.main_signal.attrs
        for attribute in ['SDkey', 'SDmask', 
                          'SrcPos', 'SrcPos2D', 
                          'DetPos', 'DetPos2D', 
                          'ChnPos', 'ChnPos2D']:
            if attribute in SD.keys():
                attr_np = _np.array(SD[attribute])
                attr_shape = attr_np.shape
                attr_np = attr_np.ravel()
                SD[attribute] = attr_np
                SD[f'{attribute}_shape'] = attr_shape
        nirs[k].p.main_signal.attrs = SD
        
    return(nirs)


class PCAFilter(_Algorithm):
    """
    See Molavi 2012

    """    
    def __init__(self, nSV=0.8, **kwargs):
        _Algorithm.__init__(self, nSV=nSV, **kwargs)
        self.dimensions = {'time':0, 
                           'channel': 0, 
                           'component':0}
    
    # def __call__(self, signal, manage_original):
    #     return _Algorithm.__call__(self, signal,
    #                                by='none', 
    #                                manage_original=manage_original)
    
    def algorithm(self, signal): #TODO: correct syntax for **kwargs

        nSV = self._params['nSV']
        n_channels = signal.p.get_nchannels()
        y = signal.p.get_values()
        # idx_good_channels = signal.get_good_channels()
        # y = y_[:, idx_good_channels]
        
        
        y = _np.concatenate([y[:,:,0], y[:,:,1]], axis=1)
        c = _np.dot(y.T, y)
        V, St, _ = _sal.svd(c)
        svs = St / _np.sum(St)
        
        ev = _np.zeros(len(svs))
        if nSV>1:
            ev[:nSV] = 1
        else:
            svsc = svs
            for idx in _np.arange(1, len(svs)):
                svsc[idx] = svsc[idx-1] + svs[idx]
            ev[svsc<=nSV] = 1
        #%
        ev = _np.diag(ev)
        
        y = y - _np.linalg.multi_dot([y, V, ev, V.T])
        
        y = _np.stack([y[:, :n_channels], y[:, n_channels:]], axis=2)
        return(y)
    
class NegativeCorrelationFilter(_Algorithm):
    '''
    Functional near infrared spectroscopy (NIRS) signal improvement based on negative correlation between oxygenated and deoxygenated hemoglobin dynamics
    '''
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time':0, 'component':0}
        
    # def __call__(self, signal, manage_original):
    #     return _Algorithm.__call__(self, signal,
    #                                by='channel', 
    #                                manage_original=manage_original)
    
    def algorithm(self, signal):
        oxy = signal.values[:,0,0]
        oxy_true = _np.zeros_like(oxy)
        
        deoxy = signal.values[:,0,1]
        deoxy_true = _np.zeros_like(oxy)
        
        
        alpha = _np.std(oxy)/_np.std(deoxy)
    
        oxy_true = 0.5 * (oxy - alpha*deoxy)
        deoxy_true = -oxy_true/alpha
        
        signal_out = _np.zeros_like(signal.values)
        signal_out[:,0,0] = oxy_true
        signal_out[:,0,1] = deoxy_true
        
        return(signal_out)


class ComputeClusters(_Algorithm):
    def __init__(self, clusters, n_min_good=3, normalize=True, **kwargs):
        _Algorithm.__init__(self, clusters = clusters, 
                            n_min_good = n_min_good, 
                            normalize=normalize, **kwargs)
        
        self.dimensions = {'time':0, 
                           'channel': len(clusters), 
                           'component':0}
    
    def __call__(self, signal_in, **kwargs):
        if 'good_channels' in signal_in.attrs:
            self.good_channels = signal_in.attrs['good_channels']
        else:
            self.good_channels = _np.arange(signal_in.dims['channel'])
        
        return(_Algorithm.__call__(self, signal_in, **kwargs))
                                            
                                            
    def algorithm(self, signal):
        def normalize_signal(x):
            return( (x-_np.mean(x))/_np.std(x))
        clusters = self._params['clusters']
        n_min_good = self._params['n_min_good']
        normalize = self._params['normalize']
        
        signal_values = signal.p.get_values()
        
        good_channels = self.good_channels
        
        out_signal = _np.nan*_np.zeros((len(signal_values), len(clusters), 2))
        for i_cluster, cluster_channels in enumerate(clusters):
            
            cluster_good_channels = []
            for ch in cluster_channels:
                if ch in good_channels:
                    cluster_good_channels.append(ch)
            
            if len(cluster_good_channels)>= n_min_good:
                signals_cluster = signal_values[:, cluster_good_channels, :]
                    
                if normalize:
                    signals_cluster = _np.apply_along_axis(normalize_signal, 0,  signals_cluster)
                
                cluster_mean = _np.mean(signals_cluster, axis=1, keepdims=True)
                out_signal[:,i_cluster, :] = cluster_mean[:,0,:]
        
        return(out_signal)
        


# def __finalize_special__(res_sig):
#     # print('----->', self.name, 'finalize')
#     original_coords = list(res_sig.coords)
#     res_sig = res_sig.reset_coords()
    
#     dimensions = list(res_sig.dims)
#     for c in original_coords:
#         if c not in dimensions:
#             res_sig = res_sig.drop(c)
#     res_sig = res_sig.to_array()
#     res_sig = res_sig.squeeze(dim='variable').drop('variable')
#     # print('<-----', self.name, 'finalize')
#     return res_sig

        
"""
class FunctionalSeparationFilter(_Algorithm):
    '''
    Yamada, T., Umeyama, S., & Matsuda, K. (2012). 
    Separation of fNIRS signals into functional and systemic components 
    based on differences in hemodynamic modalities. 
    PloS one, 7(11), e50271.
    
    From:
        https://unit.aist.go.jp/hiiri/nrehrg/download/dl002_download.html
    '''
    
    def __init__(self, kf=-0.6, nbins=8, **kwargs):
        _Algorithm.__init__(self, kf=kf, nbins=nbins, **kwargs)
        self.dimensions = 'special'
    
    def __finalize__(self, res_sig, arr_window):
        return __finalize_special__(res_sig)
    
    def __get_template__(self, signal):
        out = _np.zeros(shape=(signal.sizes['time'],
                               signal.sizes['channel'],
                               4))
        
        out = _xr.DataArray(out, dims=('time', 'channel', 'component'),
                            coords = {'time': signal.coords['time'].values,
                                      'channel': signal.coords['channel'],
                                      'component': _np.arange(4)})
        return {'channel': 1}, out
    
    def algorithm(self, signal):
        def _mi(x1,x2, bins=8):
            c_xy = _np.histogram2d(x1, x2, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            return mi
        
        kf = self._params['kf']
        nbins = self._params['nbins']
        
        signal_values = signal.p.main_signal.values
        signal_functional_out = _np.zeros_like(signal_values)
        signal_systemic_out = _np.zeros_like(signal_values)
        
        ks_grid = _np.arange(0,5,0.01)
        ks_ = []
        
        n_channels = signal.sizes['channel']
        for i_ch in range(n_channels):
            cmin = _np.inf
            ks_min = ks_grid[0]
            signal_ch = signal_values[:,i_ch,:]
            
            done=False
            counter_up=0
            i_grid=0
            c_ = []
            while not done:
                ks = ks_grid[i_grid]
                p = _np.dot(signal_ch, _np.linalg.inv(_np.array([[1,ks],[1,kf]])))
                c = _mi(p[:,0],p[:,1], nbins)
                c_.append(c)
                if c < cmin:
                    cmin = c
                    ks_min = ks
                else:
                    counter_up +=1
                i_grid +=1
                
                #I can stop after I found the first minimum
                if counter_up == 10:
                    done=True
            
            p = _np.dot(signal_ch, _np.linalg.inv(_np.array([[1,ks_min],[1,kf]])))
            ks_.append(ks_min)
            
            signal_systemic_out[:,i_ch, 0] = p[:,0]
            signal_systemic_out[:,i_ch, 1] = ks*p[:,0]
            
            signal_functional_out[:, i_ch, 0] = p[:,1]
            signal_functional_out[:, i_ch, 1] = kf*p[:,1]
        
        signal_out = _np.concatenate([signal_functional_out, signal_systemic_out], axis=2)
        # signal_out = signal.clone_properties(signal_out)
        # signal_out.update_info('ks', ks_)
        out = signal.copy(deep=True)
        out = out.pad(component=(1,1), mode='edge')
        out = out.assign_coords(component=_np.arange(4))
        out.values = signal_out
        
        self._params['ks'] = ks_
        
        return out
"""
#TODO: slowly include pynirs