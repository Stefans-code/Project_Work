# coding=utf-8
# from dask.distributed import Client
# client = Clients()

import matplotlib.pyplot as plt
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

from copy import copy
import numpy as _np
import pandas as _pd

import xarray as _xr

_xr.set_options(keep_attrs = True)

#%% SIGNAL
def create_signal(data, times=None, sampling_freq=None,
                  start_time=0, name='signal', info={}):
    
    assert (times is None) ^ (sampling_freq is None), "Either times or sampling freq"
    
    if data.ndim == 1:
        data = _np.expand_dims(data, [1,2])
    elif data.ndim == 2:
        data = _np.expand_dims(data, 2)
    elif data.ndim > 3:
        raise ValueError
    
    
    if sampling_freq is None: #defined by times
        assert len(times) == data.shape[0]
        
        # #check if there is a fsamp, else fsamp is None (unevenly)
        # dt = _np.unique(_np.diff(times))
        
        # if len(dt)==1:
        #     sampling_freq = 1/dt[0]
        
        #create pandas TimedeltaIndex times
        # times = _pd.to_timedelta(times, unit='s') #NOT OK FOR SAVING
        
    else: #defined by sampling freq
        assert sampling_freq > 0
        
        decimals = _np.max([5, int(_np.ceil(_np.log10(sampling_freq)))])
        times = _np.round(_np.arange(0, data.shape[0])/sampling_freq + start_time, 
                          decimals = decimals)
        
    #start_time is times[0]
    start_time = times[0]
        
    #check dims and set coordinates
    dims = ('time', 'channel', 'component')
    
    coords = {'time':times}
    
    for i_dim in _np.arange(1,3): #assign coords to other dimensions
        coords[dims[i_dim]] = _np.arange(data.shape[i_dim])
        
    # info['sampling_freq'] = sampling_freq
    # info['start_time'] = start_time
        
    signal = _xr.DataArray(data, dims = dims,
                           coords = coords, 
                           attrs = info,
                           name = name).to_dataset()
    
    signal.attrs['MAIN'] = name
    signal.attrs['history'] = [name]
    
    return signal

@_xr.register_dataarray_accessor('p')
class PyphysioDataArray(object):
    def __init__(self, xdataarray):
        self.da = xdataarray
    
    @property
    def main_signal(self):
        return self.da
    
    def clone(self, values, name='signal'):
        assert values.shape[0] == self.da.values.shape[0]
        signal_clone = create_signal(values, times = self.da.coords['time'].values,
                                     name = name, info=copy(self.da.attrs))
        return(signal_clone)
    
    def get_values(self):
        return self.da.values

    def get_times(self):
        time = self.da.coords['time'].values
        # time = time/_np.timedelta64(1, 's')
        return time
    
    def segment_time(self, t_start, t_stop=None):
        """
        Segment the signal given a time interval

        Parameters
        ----------
        t_start : float
            The instant of the start of the interval
        t_stop : float 
            The instant of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        # t_start_timedelta = _pd.to_timedelta(t_start, 's')
        # t_stop_timedelta = _pd.to_timedelta(t_stop, 's')
        
        #TODO t_stop - 1/fsamp
        sub_dataset = self.da.sel(time = slice(t_start,
                                               t_stop))
        return sub_dataset
    
    def get_start_time(self):
        times= self.get_times()
        return(times[0])
    
    def get_end_time(self):
        times= self.get_times()
        return(times[-1])

    def get_sampling_freq(self):
        dt = _np.unique(_np.diff(self.get_times()).round(10))
        if len(dt)==1:
            return 1/dt[0]
        
        return None
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()

    def has_multi_channels(self):
        return(len(self.da.dims)>1)
    
    def get_nchannels(self):
        if self.has_multi_channels():
            return(int(_np.max(self.da.coords['channel'])+1))
        else:
            return(1)
    
    def has_multi_components(self):
        return(len(self.da.dims)>2)
    
    def get_ncomponents(self):
        if self.has_multi_components():
            return(int(_np.max(self.da.coords['component'])+1))
        else:
            return(1)
    
    def get_info(self):
        return self.da.attrs

    def plot(self, marker=None, ncols=4, sharey=True):
        fig = _gcf()
        t_ = self.get_times()
        v_ = self.get_values()
        linestyle='solid'
        #if single signal, then plot
        if len(self.da.dims) == 1:
            
            #TODO if existing figure has many axes, 
            #replicate the plot on each axis
            
            #if good then use asolid line
            #else use a dotted line
            # linestyle='solid'
            # if self.has_good():
            #     good = self.get_good()
            #     if len(good)==0:
            #         linestyle = 'dotted'
            
            #plot the signal
            ax = _gca()
            
            if marker is None:                
                ax.plot(t_, _np.squeeze(v_), linestyle = linestyle)
            else:
                ax.plot(t_, _np.squeeze(v_), marker, linestyle = linestyle)
            _grid(True)
        
        else:
            n_ch = self.get_nchannels()
            n_comp = self.get_ncomponents()

            #if existing figure has enough number of axes
            #use the figure
            if len(fig.axes)>= n_ch:
                axes = fig.axes
            
            #else create a new figure 
            else: 
                if n_ch>1:
                    #compute number of cols and rows and create a new figure
                    n_cols = n_ch if n_ch < ncols else ncols
                    n_rows = int(_np.ceil(n_ch/n_cols))
                    
                    fig, axes = _subplots(n_rows, n_cols, num = fig.number, sharex=True, sharey=sharey)
                    axes = axes.ravel()
                else:
                    fig, axes = _subplots(1, 1, num = fig.number, sharex=True, sharey=sharey)
                    axes = [axes]
            
            #recursive calls to signal.plot()
            #for each channel and component
            for i_ch in range(n_ch):
                _sca(axes[i_ch])
                ax = _gca()
                if n_comp>1:
                    for i_comp in range(n_comp):
                        if marker is None:                
                            ax.plot(t_, v_[:,i_ch, i_comp], linestyle = linestyle)
                        else:
                            ax.plot(t_, v_[:,i_ch, i_comp], marker, linestyle = linestyle)
                else:
                    if marker is None:                
                        ax.plot(t_, v_[:,i_ch], linestyle = linestyle)
                    else:
                        ax.plot(t_, v_[:,i_ch], marker, linestyle = linestyle)

                _ylabel(i_ch)
                _grid(True)
                
            _xlim(self.get_start_time(), self.get_end_time())
            _tight_layout()
            _subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.2)

@_xr.register_dataset_accessor('p')
class PyPhysioDataset(object):
    def __init__(self, xdataset):
        self.ds = xdataset
    
    # def clone(self, values, name='signal'):
    #     assert values.shape[0] == self.da.values.shape[0]
    #     signal_clone = create_signal(values, times = self.da.coords['time'].values,
    #                                  name = name, info=self.da.attrs)
    #     return(signal_clone)
    
    @property
    def main_signal(self):
        main_signal = self.ds.attrs['MAIN']
        da = self.ds[main_signal]
        return da
    
    def get_values(self):
        return self.main_signal.p.get_values()
    
    def get_times(self):
        time = self.main_signal.p.get_times()
        return time
    
    def segment_time(self, t_start, t_stop=None):
        """
        Segment the signal given a time interval

        Parameters
        ----------
        t_start : float
            The instant of the start of the interval
        t_stop : float 
            The instant of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        # t_start_timedelta = _pd.to_timedelta(t_start, 's')
        # t_stop_timedelta = _pd.to_timedelta(t_stop, 's')
        
        #TODO t_stop - 1/fsamp
        sub_dataset = self.ds.sel(time = slice(t_start,
                                               t_stop))
        return sub_dataset
    
    def get_start_time(self):
        return self.main_signal.p.get_start_time()
        
    def get_end_time(self):
        return self.main_signal.p.get_end_time()

    def get_sampling_freq(self):
        return self.main_signal.p.get_sampling_freq()
    
    def get_duration(self):
        return self.main_signal.p.get_duration()

    def get_info(self):
        return self.ds.attrs

    def plot(self, marker=None, ncols=4, sharey=True):
        self.main_signal.p.plot(marker=marker,
                                ncols=ncols,
                                sharey=sharey)
    

#%% create signal
data = _np.random.uniform(size = (100, 10, 2))
sampling_freq = 1000
signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')

#%% ALGORITHMS
class Algorithm(object):
    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)
        self.dimensions = {}
    
    def __call__(self, signal_in, add_signal=True, dimensions=None):
        #This function iteratively calls the self.algorithm on each signal
        #(i.e. channel+component)
        # print('-----> Algorithm.__call__')
        
        #The user will mainly call Algorithms on a Dataset
        #but the __call__ rolling mechanism assumes to operate on DataArray
        if isinstance(signal_in, _xr.Dataset):
            signal = signal_in.p.main_signal.copy(deep=True)
        else:
            signal = signal_in.copy(deep=True)
        
        signal_name = signal.name
        #from here, signal is a DataArray
                    
        #typical usage
        #will include all dimensions except for those
        #along which the algorithm is applied
        # collapse_dims = [] 
        
        signal_stacked = signal.stack(new=['channel', 'component']).transpose('new', ...)
        
        #create a roller over new dimension
        #(so to process one component at a time)
        rr = signal_stacked.rolling(new=1)

        results = []
        for label, block in rr:
            # print(block)
            # print(label)
            result = self.algorithm(block)
            
            result = _np.expand_dims(_np.array(result), 1)
            result_xr = _xr.DataArray(result, dims=('time', 'new'))
            result_xr = result_xr.assign_coords({'new':block.coords['new']})
            
            if result_xr.sizes['time'] == block.sizes['time']:
                result_xr = result_xr.assign_coords({'time':block.coords['time']})
            elif result_xr.sizes['time'] == 1:
                
                t_start = block.coords['time'].values[0]
            
                result_xr = result_xr.assign_coords(time=[t_start])
            
            results.append(result_xr)
            
            
            # result_xr = arr_window.copy(deep=True)
            
        # #recompose
        signal_out = _xr.concat(results, 'new')
        # return(signal_out)
        
        signal_out = signal_out.transpose('time', ...)

        # if len(dimensions)>1:
        signal_out = signal_out.unstack()
        return(signal_out)
    
    def __finalize__(self, result, signal_in):
        '''
        General function to obtain a coherent output 
        from the calls to self.algorithm.
        The output should be a dataaarry or dataset
        '''
        
        # print('-----> Algorithm.__finalize__')
        
        # if one dimensional, assume the processed dimension is time
        if result.ndim == 1:
            result = _np.expand_dims(result, [1,2])
        
        signal_out = _xr.DataArray(result,
                                   dims=('time', 'channel', 'component'), 
                                   name=signal_in.name)
        
        for dim in signal_in.dims:
            if signal_in.sizes[dim] == signal_out.sizes[dim]:
                #same size --> same coords
                signal_out = signal_out.assign_coords({dim:signal_in.coords[dim].values})
            
            elif signal_out.sizes[dim] == 1: 
                #indicators or windowed algorithms
                coord_start = signal_in.coords[dim].values[0]
                signal_out = signal_out.assign_coords({dim:[coord_start]})
            else:
                #we do not know how to assign coordinates to this dimension
                pass
        signal_out.attrs = signal_in.attrs.copy()
        # print('<----- Algorithm.__finalize__')

        return(signal_out)
        # 
    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def algorithm(cls, signal):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        pass

class Mean(Algorithm):
    def __init__(self, **kwargs):
        Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time':1}

    def algorithm(self, signal):
        # print('-----> Mean.algorithm')
        # print(type(signal))
        # print(signal.shape)
        result = _np.mean(signal.values.ravel())
        # print(type(result))
        result = _np.array([float(result)])
        # print(result.shape)
        # print('<----- Mean.algorithm')
        return result

class Normalize(Algorithm):
    def __init__(self, **kwargs):
        Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time':0}
        
    def algorithm(self, signal):
        print('-----> Normalize.algorithm')
        # print(signal)
        signal_values = signal.values
        # print(signal_values.shape)
        # print(signal)
        
        mean = Mean(add_signal=True)(signal, dimensions='none')
        # print(type(mean))
        # print(mean.shape)
        
        result = signal_values - mean.values
        # print(type(result))
        # print(result.shape)
        print('<----- Normalize.algorithm')
        return result
            
#%%
results = Mean()(signal, dimensions={'channel':1})

#%%
result = Normalize()(signal)


#%% PSD
from scipy.signal import welch as _welch

class PSD(Algorithm): #xarray done
    def __init__(self, method, nfft=2048, window='hamming', min_order=10, max_order=30, normalize=False,
                 remove_mean=True, **kwargs):
        
        _method_list = ['welch', 'fft', 'ar']
        _window_list = ['hamming', 'blackman', 'hanning', 'bartlett', 'none']

        assert method in _method_list, "Parameter method should be in " + _method_list.__repr__()
        assert nfft > 0, "nfft value should be positive"
        assert window in _window_list, "Parameter window type should be in " + _window_list.__repr__()
        if method == "ar":
            assert min_order > 0, "Minimum order for the AR method should be positive"
            assert max_order > 0, "Maximum order for the AR method should be positive"
        
        Algorithm.__init__(self, method=method, nfft=nfft, window=window, min_order=min_order,
                           max_order=max_order, normalize=normalize, remove_mean=remove_mean, **kwargs)
        
        self.dimensions = 'special'

    def __finalize__(self, res_sig, arr_window):
        return res_sig
    
    def algorithm(self, signal):
        print('-----> PSD.algorithm')
        # print(signal.shape)
        params = self._params
        nfft = params['nfft'] if "nfft" in params else None
        
        fsamp = 1000#signal.p.get_sampling_freq()
        
        signal_values = signal.values.ravel()
        
        freqs, psd = _welch(signal_values, fsamp, window='hamming', return_onesided=True, nfft=nfft)
        print(type(freqs))
        print(freqs.shape)
        
        print(type(psd))
        print(psd.shape)
        
        N = int(nfft/2 + 1)
        freqs = _np.linspace(start=0, stop=fsamp / 2, num=N)
        psd = _np.expand_dims(psd,[1, 2])
        out = signal.copy(deep=True)
        out = out.expand_dims({'freq':freqs}, axis=0)[:,0]
        out = out.drop('time')
        # out.name = signal.name+'_'+self.name
        # print(out)
        
        out.values = psd
        print('<----- PSD.algorithm')
        return out
    
    def __get_template__(self, signal):
        nfft = self._params['nfft']
        N = int(nfft/2 + 1)
        out = _np.zeros(shape=(N, #1,
                               signal.sizes['channel'], 
                               signal.sizes['component']))
        
        fsamp = 1000# signal.p.get_sampling_freq()
        freqs = _np.linspace(start=0, stop=fsamp / 2, num=N)
        
        out = _xr.DataArray(out, dims=('freq', 'channel', 'component'), #'time', 'channel', 'component'),
                            coords = {'freq': freqs,
                                      # 'time': [signal.coords['time'].values[0]],
                                      'channel': signal.coords['channel'],
                                      'component': signal.coords['component']})
        # print(out)
        return {'channel': 1, 'component':1}, out

#%%
result = PSD('welch')(signal)
#%% NIRS
from pynirs.loaders import load_nirx
import pyphysio.sqi.sqi as sqi
import xarray as xr
from pynirs.quality_control.sqi import ScalpCoupling, ScalpCouplingPower, CVWavelengths, \
    CardiacPowerRatio, compute_cardiac_freq

nirs = load_nirx('/home/bizzego/UniTn/data/fnirs_sexism/original/F01_1', False)

#%%
nirs_ = Mean()(nirs)
nirs_ = Normalize()(nirs)

#%%
nirs_ = Mean()(nirs, dimensions = {'time':1, 'component':1})
