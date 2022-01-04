# coding=utf-8
import numpy as _np
import xarray as _xr
from copy import copy

_xr.set_options(keep_attrs=True)

from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca


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
        
        # decimals = _np.max([5, int(_np.ceil(_np.log10(sampling_freq)))])
        times = _np.arange(0, data.shape[0])/sampling_freq + start_time#, 
                          # decimals = decimals)
        
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


#TODO: add resampling and interpolate

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
        return(self.get_nchannels()>1)
    
    def get_nchannels(self):
        return(len(self.da.coords['channel']))
        
    def has_multi_components(self):
        return(self.get_ncomponents()>1)
    
    def get_ncomponents(self):
        return(len(self.da.coords['component']))
    
    def get_info(self):
        return self.da.attrs

    def plot(self, marker=None, ncols=4, sharey=True):
        fig = _gcf()
        t_ = self.get_times()
        v_ = self.get_values()
        linestyle='solid'
        
        n_ch = self.get_nchannels()
        n_comp = self.get_ncomponents()
        
        #if single signal, then plot
        if n_ch == 1:
            v_ = v_[:,0,:]

            #TODO if existing figure has many axes, 
            #replicate the plot on each axis
            
            #if good then use a solid line
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