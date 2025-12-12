import numpy as _np
import scipy.linalg as _sal
from ..._base_algorithm import _Algorithm
from sklearn.decomposition import PCA as _PCA, FastICA as _ICA
from sklearn.preprocessing import StandardScaler as _StandardScaler
import statsmodels.api as _sm

# from ._dl_sqi import SignalQualityDeepLearning
from ._convert import Raw2Oxy, Raw2OD, OD2Oxy
import matplotlib.pyplot as _plt
import matplotlib as _mpl
import matplotlib.cm as _cm


#COMPARE 
def compute_betas_barker(nirs_signal, dm, pmax=10, max_iter = 10):
    Y = nirs_signal
    X = dm
    
    Y = Y.ravel()
    assert X.shape[0] == len(Y)
    # 1. Initialize beta via an OLS fit.
    model_initial = _sm.OLS(Y, X)
    results_initial = model_initial.fit()
    beta_outer = results_initial.params
    residuals = results_initial.resid

    iteration_outer = 0
    done_outer = False
    while ((not done_outer) and (iteration_outer<max_iter)):
        #. Fit the residual to an AR(P) model where P minimizes BIC (Eq. (5)).
        bic = []
        for p in range(pmax):
            model = _sm.tsa.ARIMA(residuals, order=(p,0,0))
            results = model.fit()
            bic.append(results.bic)
            
        p_optim = _np.argmin(bic)+1
        model_AR = _sm.tsa.ARIMA(residuals, order=(p_optim,0,0))
        results_AR = model_AR.fit()
        
        # Generate the whitening filter f:
        f = [1]
        for i in range(p_optim):
            f.append(-results_AR.arparams[i])
        f = _np.array(f)
        
        #. Apply the whitening filter to the data y and column-wise to the design matrix X
        Y_w = _np.convolve(Y, f, 'same')
        X_w = _np.apply_along_axis(_np.convolve, 0, X, *[f, 'same'])
        
        #. Perform iteratively reweighted least squares (IRLS)
        done = False
        iterations = 0
        beta_inner = beta_outer
        #initialize weights to ones
        weights = _np.ones(len(Y_w))
        while((not done) and (iterations<max_iter)):
            #a- solve beta by WLS
            #fit weighted LS
            model_WLS = _sm.WLS(Y_w, X_w, weights=weights)
            results_WLS = model_WLS.fit()
            #get new beta
            beta_new = results_WLS.params
            
            #b- recalculate weights
            residuals_WLS = results_WLS.resid
            weights = _sm.robust.norms.TukeyBiweight(c=4.685).weights(residuals_WLS)
            change = _np.min(abs((beta_new - beta_inner)/beta_inner))
            
            #c- repeat steps 5a-b until changes in beta are small (<1%)
            if (change <0.01) or (iterations >= max_iter):
                done = True
            
            beta_inner = beta_new
            
            iterations +=1

        change_outer = _np.min(abs((beta_outer - beta_inner)/beta_outer))
        
        #Repeat steps 2-5 until changes in β are sufficiently small (e.g., < 1% change).  
        if (change_outer <0.01) or (iteration_outer >= max_iter):
            done_outer = True
        
        beta_outer = beta_inner
        residuals = Y - _np.dot(X, beta_outer)
        iteration_outer +=1

    beta = beta_outer
    return(beta)

#%%
def plot_probe(nirs, values=None):
    if values is not None:
        norm = _mpl.colors.Normalize(vmin=_np.min(values), 
                                     vmax=_np.max(values))
        cmap = _plt.get_cmap('bwr')
        m = _cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = _plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('equal')
 
    for idx_ch in nirs.channel.values:
        ch_pos = get_ch_pos(nirs, idx_ch)
        
        ax.text(ch_pos[0], ch_pos[1], ch_pos[2], idx_ch, color='k', fontsize=14)
        if values is not None:
            color = m.to_rgba(values[idx_ch])
        else:
            color = 'y'
        ax.scatter(ch_pos[0], ch_pos[1], ch_pos[2], color=color, marker='o')
   
    _plt.show()

#%%
def get_ss_ls_channels(nirs, max_dist=1.5):
    idx_ss = []
    idx_ls = []
    distances = []
    for idx_ch, ch in enumerate(nirs.attrs['Channels']):
        distances.append(ch[3])
        if ch[3]<=max_dist:
            idx_ss.append(idx_ch)
        else:
            idx_ls.append(idx_ch)
    return(idx_ss, idx_ls)

def get_ch_pos(nirs, ch_target, twoD=False):
    if twoD:
        src_pos = _np.array(nirs.attrs['SrcPos2D'])
        det_pos = _np.array(nirs.attrs['DetPos2D'])
    else:
        src_pos = _np.array(nirs.attrs['SrcPos'])
        det_pos = _np.array(nirs.attrs['DetPos'])

    ch_target_info = nirs.attrs['Channels'][ch_target]
    ch_src = int(ch_target_info[1])
    ch_det = int(ch_target_info[2])

    ch_src_pos = src_pos[ch_src]
    ch_det_pos = det_pos[ch_det]
    
    ch_pos = (ch_src_pos + ch_det_pos)/2
    return(ch_pos)

def get_near_channels(nirs, ch_target, n_near=3):
    ch_target_pos = get_ch_pos(nirs, ch_target)
    
    distances = []
    for idx_ch in nirs.channel.values:
        ch_curr_pos = get_ch_pos(nirs, idx_ch)
        distances.append(_np.linalg.norm(ch_target_pos - ch_curr_pos))

    sorted_channels = _np.argsort(distances)
    near_channels = sorted_channels[:n_near]
    return(near_channels)

#%%
class PCAFilter(_Algorithm):
    """

    """    
    def __init__(self, nSV=0.8, return_systemic=False, **kwargs):
        _Algorithm.__init__(self, nSV=nSV, 
                            return_systemic=return_systemic,
                            **kwargs)
        self.required_dims = ['time', 'channel', 'component']
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal)
        return (chunk_dict, template)
    
    def algorithm(self, signal): #TODO: correct syntax for **kwargs
        nSV = self._params['nSV']
        return_systemic = self._params['return_systemic']
        n_channels = signal.p.get_nchannels()
        y = signal.p.get_values()
        # idx_good_channels = signal.get_good_channels()
        # y = y_[:, idx_good_channels]
        
        y = _np.concatenate([y[:,:,0], y[:,:,1]], axis=1)
        c = _np.dot(y.T, y)
        V, St, _ = _sal.svd(c)
        
        svs = St / _np.sum(St)
        ev = _np.zeros(len(svs))
        if nSV>=1:
            ev[:nSV] = 1
        else:
            svsc = svs
            for idx in _np.arange(1, len(svs)):
                svsc[idx] = svsc[idx-1] + svs[idx]
            ev[svsc<=nSV] = 1
        #%
        ev = _np.diag(ev)
        
        y_systemic = _np.linalg.multi_dot([y, V, ev, V.T])
        if return_systemic:
            y_systemic = _np.stack([y_systemic[:, :n_channels], y_systemic[:, n_channels:]], axis=2)
            return y_systemic
        
        y_filt = y - y_systemic
        y_filt = _np.stack([y_filt[:, :n_channels], y_filt[:, n_channels:]], axis=2)
        return(y_filt)

class RegressShortSeparation(_Algorithm):
    '''
    T. Sato et al., “Reduction of global interference of scalp-hemodynamics 
    in functional near-infrared spectroscopy using short distance probes,” 
    NeuroImage 141, 120–132 (2016).
    '''
    def __init__(self, var_explained=0.9, max_dist=1.5, **kwargs):
        _Algorithm.__init__(self, var_explained=var_explained, 
                            max_dist=max_dist, **kwargs)
        self.required_dims = ['time', 'channel', 'component']
        
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal)
        return (chunk_dict, template)
    
    def algorithm(self, signal):
        params = self._params
        max_dist = params['max_dist']
        var_explained = params['var_explained']
        idx_ss = []
        idx_ls = []
        
        idx_ss, idx_ls = get_ss_ls_channels(signal, max_dist)
        nirs_ss = signal.isel({'channel':idx_ss})
        
        #normalize SS channels
        nirs_ss_ = nirs_ss.p.get_values()
        nirs_ss_[:,:,0] = _StandardScaler().fit_transform(nirs_ss_[:,:,0])
        nirs_ss_[:,:,1] = _StandardScaler().fit_transform(nirs_ss_[:,:,1])
        
        #separately for oxy and deoxy?
        pca = _PCA()
        oxy_components = pca.fit_transform(nirs_ss_[:,:, 0])
        variance = _np.cumsum(pca.explained_variance_ratio_)
        idx_keep = _np.where(variance<=var_explained)[0]
        oxy_components = oxy_components[:,idx_keep]

        deoxy_components = pca.fit_transform(nirs_ss_[:,:, 1])
        variance = _np.cumsum(pca.explained_variance_ratio_)
        idx_keep = _np.where(variance<=var_explained)[0]
        deoxy_components = deoxy_components[:,idx_keep]
        
        nirs_filtered = signal.p.get_values().copy()

        for idx_ch in idx_ls:
            nirs_ch_ = nirs_filtered[:,idx_ch]
            
            model_ar = _sm.GLS(nirs_ch_[:,0], oxy_components)
            results_ar = model_ar.fit()
            nirs_filtered[:, idx_ch, 0] = results_ar.resid
                
            model_ar = _sm.GLS(nirs_ch_[:,1], deoxy_components)
            results_ar = model_ar.fit()
            nirs_filtered[:, idx_ch, 1] = results_ar.resid
                    
        return(nirs_filtered)


class NegativeCorrelationFilter(_Algorithm):
    '''
    Functional near infrared spectroscopy (NIRS) signal improvement based on negative correlation between oxygenated and deoxygenated hemoglobin dynamics
    '''
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = ['time', 'component']
        
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
    def __init__(self, clusters, n_min_good=0, mode='mean', **kwargs):
        _Algorithm.__init__(self, clusters = clusters, 
                            n_min_good = n_min_good,
                            mode=mode,
                            **kwargs)
        
        self.required_dims = ['time', 'channel']
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        clusters = self._params['clusters']
        template = self.__compute_template__(signal, {'channel': len(clusters)})
        return (chunk_dict, template)
    
    def __call__(self, signal_in, **kwargs):
        if 'good_channels' in signal_in.attrs:
            self.good_channels = signal_in.attrs['good_channels']
        else:
            self.good_channels = _np.arange(_np.max(signal_in.channel)+1)
        
        return(_Algorithm.__call__(self, signal_in, **kwargs))
                                            
                                            
    def algorithm(self, signal):

        clusters = self._params['clusters']
        n_min_good = self._params['n_min_good']
        mode = self._params['mode']
        
        good_channels = self.good_channels

        out_signal = _np.nan*_np.zeros((signal.sizes['time'], len(clusters), 1))
        for i_cluster, cluster_channels in enumerate(clusters):
            
            if n_min_good != 0:
                cluster_good_channels = []
                for ch in cluster_channels:
                    if ch in good_channels:
                        cluster_good_channels.append(ch)
            else:
                cluster_good_channels = cluster_channels
            
            if len(cluster_good_channels)>= n_min_good:
                signals_cluster = signal.sel({'channel':cluster_good_channels}).p.get_values()[:,:,0]
                
                if mode == 'pca':
                    cluster_signal = _PCA(1).fit_transform(signals_cluster.copy())
                                       
                    corr=[]
                    for i in range(len(cluster_good_channels)):
                        corr.append(_np.corrcoef(cluster_signal.ravel(), 
                                                 signals_cluster[:,i].ravel())[1,0])
                    if _np.mean(corr)<0:
                        cluster_signal = -cluster_signal
                    
                elif mode == 'ica':
                    cluster_signal = _ICA(1).fit_transform(signals_cluster.copy())
                    corr=[]
                    for i in range(len(cluster_good_channels)):
                        corr.append(_np.corrcoef(cluster_signal.ravel(), 
                                                 signals_cluster[:,i].ravel())[1,0])
                    if _np.mean(corr)<0:
                        cluster_signal = -cluster_signal
                else:
                    cluster_signal = _np.mean(signals_cluster, axis=1, keepdims=True)
                    
                out_signal[:,i_cluster, :] = cluster_signal
        
        return(out_signal)


    
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
