# coding=utf-8
import numpy as _np
import pandas as _pd
import numpy.ma as _ma
import xarray as xr

# from scipy import interpolate as _interp
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

# from numbers import Number as _Number
# import copy

# def from_pickleable(pickle):
#     """
#     Builds a Signal using the pickleable tuple version of it.
#     :param pickle: Tuple of the form (Signal, ph dict).
#     :return: Signal
#     """
#     pass


# def from_pickle(path):
#     """
#     Loads a Signal from a pickle file given the path.
#     :param path: File system path to the pickle file.
#     :return: A Signal.
#     """
#     pass

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
    dims_template = ('time', 'channels', 'components', 'dim4', 'dim5')
    assert data.ndim <= 5
    dims = dims_template[:data.ndim]
    
    info['sampling_freq'] = sampling_freq
    info['start_time'] = start_time
    
    signal = xr.DataArray(data, dims = dims,
                          coords = {'time': times}, 
                          attrs = info,
                          name = 'signal')
    
    return(signal.to_dataset())

@xr.register_dataset_accessor('ph')
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
            return(self.ds.dims['channels'])
        else:
            return(1)
    
    def has_multi_components(self):
        return(len(self.ds.dims)>2)
    
    def get_ncomponents(self):
        if self.has_multi_components():
            return(self.ds.dims['components'])
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
        
data = _np.random.uniform(size = (1000, 10,5))
sampling_freq = 1000
signal = create_signal(data, sampling_freq=sampling_freq)
# signal.ph.plot()

#%%

'''        
    def has_good(self, good_global=True):
        if 'good' in self.ph['info'].keys():
            if not good_global:
                return True
            else:
                good = self.ph['info']['good']
                if good.shape[0] == 1:
                    return True
                return False
        return False
    
    def get_good(self):
        assert self.has_good(), "Quality has not been computed yet"
        
        info = self.get_info()
        is_good = info['good']
        # assert is_good.shape[0] == 1, "Quality has not been computed globally. Please compute global quality first"
        
        if is_good.ndim == 1:
            return(_np.array(_np.where(is_good))[0])
        else:
            return(_np.array(_np.where(is_good)[1:]))
    


    @property
    def pickleable(self):
        """
        Returns a pickleable tuple of this Signal.
        :return: Tuple (Signal, ph dict).
        """
        info = self.get_info()
        
        
        info_pickle = {}
        for info_key in info.keys():
            if info_key == 'sqi':
                info_sqi = {}
                for k in info['sqi'].keys():
                    info_sqi[k] = info['sqi'][k].pickleable
                info_pickle['sqi'] = info_sqi
            
            elif info_key == 'good':
                if hasattr(info['good'], 'pickleable'):
                    info_pickle['good'] = info['good'].pickleable
            
            elif info_key == 'stim':
                if hasattr(info['stim'], 'pickleable'):
                    info_pickle['stim'] = info['stim'].pickleable
            else:
                info_pickle[info_key] = info[info_key]
                
        return self, self._optinfo, info_pickle

    def to_pickle(self, path):
        """
        Saves this Signal into a pickle file.
        :param path: File system path to the file to write (create/overwrite).
        """
        from gzip import open
        from pickle import dump
        f = open(path, "wb")
        dump(self.pickleable, f, protocol=2)
        f.close()
       
    def __repr__(self):
        return self.get_values().__repr__() + '\n'+\
            f'{self.get_sampling_freq()} Hz \n'+\
                f'{self.get_start_time()} s \n'
'''