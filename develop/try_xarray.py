# coding=utf-8
import numpy as _np
import pandas as _pd
import numpy.ma as _ma

# from scipy import interpolate as _interp
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

import xarray as _xr

def create_signal(data, times=None, sampling_freq=None,
                  start_time=0, info={}):
    
    assert (times is None) ^ (sampling_freq is None), "Either times or sampling freq"
    
    if sampling_freq is None: #defined by times
        assert len(times) == data.shape[0]
        
        #check if there is a fsamp, else fsamp is None (unevenly)
        dt = _np.unique(_np.diff(times))
        if len(dt)==1:
            sampling_freq = 1/dt[0]
        
        #create pandas TimedeltaIndex times
        times = _pd.to_timedelta(times, unit='s')
        
    else: #defined by sampling freq
        assert sampling_freq > 0
        #define times as TimedeltaIndex
        times = _pd.timedelta_range(start=start_time, periods = data.shape[0], freq=f'{10**9/sampling_freq}ns')
        
    #start_time is times[0]
    start_time = times[0]
        
    #check dims
    dims_template = ('time', 'channel', 'component')
    assert data.ndim <= 3
    dims = dims_template[:data.ndim]
    
    info['sampling_freq'] = sampling_freq
    info['start_time'] = start_time
    
    signal = _xr.DataArray(data, dims = dims,
                           coords = {'time': times}, 
                           attrs = info,
                           name = 'signal')
    
    return(signal.to_dataset())

@_xr.register_dataset_accessor('p')
class Signal(object):
    def __init__(self, xarray_obj):
        self.ds = xarray_obj
    
    def __getitem__(self, item):
        print(item)
        
    @property
    def signal(self):
        return self.ds['signal']
    
    def get_values(self):
        return self.signal.values

    def get_times(self):
        time = self.signal.coords['time'].values
        time = time/_np.timedelta64(1, 's')
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
        t_start_timedelta = _pd.to_timedelta(t_start, 's')
        t_stop_timedelta = _pd.to_timedelta(t_stop, 's')
        
        sub_dataset = self.ds.sel(time = slice(t_start_timedelta,
                                               t_stop_timedelta))
        return sub_dataset
    
    def get_start_time(self):
        times= self.get_times()
        return(times[0])
    
    def get_end_time(self):
        times= self.get_times()
        return(times[-1])

    def get_sampling_freq(self):
        return self.signal.attrs['sampling_freq']
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()

    def has_multi_channels(self):
        return(len(self.ds.dims)>1)
    
    def get_nchannels(self):
        if self.has_multi_channels():
            return(self.ds.dims['channel'])
        else:
            return(1)
    
    def has_multi_components(self):
        return(len(self.ds.dims)>2)
    
    def get_ncomponents(self):
        if self.has_multi_components():
            return(self.ds.dims['component'])
        else:
            return(1)
    
    def get_info(self):
        return self.ds.attr

    def plot(self, marker=None, ncols=4, sharey=True):
        fig = _gcf()
        t_ = self.get_times()
        v_ = self.get_values()
        linestyle='solid'
        #if single signal, then plot
        if len(self.ds.dims) == 1:
            
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
        
#%%
class Algorithm(object):

    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)  # already checked by __init__

    def __call__(self, data):
        assert isinstance(data, _xr.Dataset)
        
        #should only be applied on the main signal
        #convert main signal to dataset
        signal = data.signal.to_dataset()        
        
        #stack dimensions other than time
        dimensions = list(signal.dims)
        if len(dimensions)>1:
            signal_stacked = signal.stack(new=dimensions[1:]).transpose('new', ...)
        else:
            signal_stacked = signal.expand_dims('new', axis=0)

        signal_out = signal_stacked.map(self.algorithm)
        signal_out = signal_out.transpose('time', ...)
        
        if len(dimensions)>1:
            signal_out = signal_out.unstack()
        else:
            signal_out = signal_out.squeeze('new')

        dataset_out = data.copy(data={'signal': signal_out.signal})
        
        return dataset_out

    def __repr__(self):
        return self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def set(self, **kwargs):
        kk = self.get()
        kk.update(kwargs)
        self.__init__(**kk)

    def get(self, param=None):
        """
        Placeholder for the subclasses
        @return
        """
        if param is None:
            return self._params
        else:
            return self._params[param]


    def algorithm(cls, signal):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        pass

class Normalize(Algorithm):
    def __init__(self, **kwargs):
        Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        print(type(signal))
        print(signal.attrs)
        print(signal.shape)
        return (signal - _np.mean(signal)) / _np.std(signal)

#%%

#%%    
import matplotlib.pyplot as plt
data = _np.random.uniform(size = 1000)
sampling_freq = 1000
data = create_signal(data, sampling_freq=sampling_freq)
# signal.ph.plot()

result = Normalize()(data)

plt.plot(data.signal.values[:])
plt.plot(result.signal.values[:])

stim = _np.zeros(1000)
stim[250:300] = 1 
stim[500:800] = 2


#%%
data.assign_coords(stim = ('time', stim))

data.rolling(stim)
