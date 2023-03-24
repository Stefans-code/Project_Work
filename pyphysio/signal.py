# coding=utf-8
import numpy as _np
import xarray as _xr
from copy import copy as _copy

_xr.set_options(keep_attrs=True)

from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca


#TODO:
#    tonumpy: return t and values

#TODO: manage special cases for unevenly signals
# > unevenly if nan in the data? what if imputation?
# - set attribute type (or function) to check if they are unevenly
# - get_values should ignore nans if they are the same across channels and components
# - plot using '.'
# resample, first run: signal.dropna('time')

#TODO: add to_hdf method
#See the _to1darray function in _load_nirx
    
def load(file):
    signal = _xr.load_dataset(file)
    
    return(signal)
    
def create_signal(data, times=None, sampling_freq=None,
                  start_time=0, name='signal', info={}):
    '''
    Create an xarray object where the coordinates are (time, channel, component)
    representing a signal.

    Parameters
    ----------
    data : numpy.array (or list)
        Values of the signal
    times : TYPE, optional
        DESCRIPTION. The default is None.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is None.
    start_time : TYPE, optional
        Ignored if times are provided. The default is 0.
    name : 'str', optional
        Name of the signal. The default is 'signal'.
    info : dict, optional
        Dictionary where to store custom information. The default is {}.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.
        
    If 'sampling_freq' is provided, a signal with evenly (temporally) spaced samples is created;
    times are created accordingly (also considering the 'start_time')
    If 'times' is provided (meaning the sampling frequency is unknown and/or not the same across the signal),
    a signal with unevenly (temporally) spaced samples is created, 
    and the 'sampling_freq' is set to 'unevenly'; 'start_time' is ignored.
    '''    
    
    #TODO: names for channels/components?
    assert (times is None) ^ (sampling_freq is None), "Either times or sampling freq"
    assert data.ndim <= 3, "data should have maximum 3 dimensions"
    
    if data.ndim == 1:
        data = _np.expand_dims(data, [1,2])
    elif data.ndim == 2:
        data = _np.expand_dims(data, 2)
    
    #--> check validity of the temporal information
    if sampling_freq is None: #defined by times
        assert len(times) == data.shape[0], "Length of provided times is different from the number of datapoints"
        #we assume that users that do not provide a sampling_freq
        #want an unevenly signal, 
        #so a signal that does not come from a sampling, 
        #i.e. for which there is not a valid sampling frequency
        sampling_freq = 'unevenly'
    else: 
        assert sampling_freq > 0
        sampling_freq = float(sampling_freq)
        if times is None: #create times
            times = _np.arange(0, data.shape[0])/sampling_freq + start_time
        else: #check that provided times are valid, given the sampling freq
            
            #why should a user provide both times and fsamp????
            #I cannot find a meaningful use case...
            # this 'else' is never executed, given the first assert in __init__
            # but I leave it here in case it is useful in the future.
            # Float precision issues...
            decimals = int(_np.ceil(_np.log10(sampling_freq)+2))
            times = times.astype(_np.float128)
            dt = _np.unique(_np.diff(times).round(decimals=decimals))
            assert len(dt)==1, "Provided times have multiple different dts"
            dt = dt[0]
            assert (1/dt - sampling_freq) < 10**(-decimals), "Sampling frequency derived from times is different from the one provided. Check times, or try to only provide sampling_freq"
            
            #times should be correct, but, just in case,
            #lets overwrite times, to be sure that everything works as expected
            #(it never does)
            start_time = times[0]
            times = _np.arange(0, data.shape[0])/sampling_freq + start_time
            
    
    #start_time is times[0]
    start_time = times[0]
        
    #check dims and set coordinates
    dims = ('time', 'channel', 'component')
    coords = {'time':times}
    
    for i_dim in _np.arange(1,3): #assign coords to other dimensions
        coords[dims[i_dim]] = _np.arange(data.shape[i_dim])
        
    info['sampling_freq'] = sampling_freq
    info['start_time'] = start_time
        
    signal = _xr.DataArray(data, dims = dims,
                           coords = coords, 
                           attrs = info,
                           name = name)
    
    signal = signal.to_dataset()
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
    
    #++++++++++++++++++++++++++++++++++++
    #!!! CHECK
    #The methods that modify the signal (self.da) should always return the new
    #signal. In other words: self.da = new_signal  will not be effective!
    #CHECK process_na, which seems to behave differently
    #++++++++++++++++++++++++++++++++++++
    
    def clone(self, values, name='signal'):
        #TODO: this is probably very rough. Do we need something more efficient?
        assert values.shape[0] == self.da.values.shape[0]
        signal_clone = create_signal(values, times = self.da.coords['time'].values,
                                     name = name, info=_copy(self.da.attrs))
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
        
        #TODO? t_stop - 1/fsamp
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
        # dt = _np.unique(_np.diff(self.get_times()).round(10))
        # if len(dt)==1:
        #     return 1/dt[0]
        
        return self.da.attrs['sampling_freq']
    
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

    def resample(self, f_out):
        t_start = self.get_start_time()
        t_end = self.get_end_time()
        
        t_out = _np.arange(t_start, t_end, 1/f_out)
        resampled_dataarray = self.da.interp(time=t_out, method='cubic')
        resampled_dataarray.attrs['sampling_freq'] = f_out
        return(resampled_dataarray)
    
    def process_na(self, na_action = 'keep'):
        '''
        Impute nans in the signal.
        
        Parameters
        ----------
        na_action : str, optional
            DESCRIPTION. The default is 'keep'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        assert na_action in ['impute', 'keep', 'remove']
        data = self.da.values
        
        #TODO: whole signal of nans??
        #replace with user-defined value?
        
        
        #--> check the nans situation
        nans_in_dataset = False
        if _np.sum(_np.isnan(data)) > 0:
            nans_in_dataset = True
            
            #try to understand if nans across channels and components 
            #share the same timepoints
            n_nans_foreach_timepoint = _np.sum(_np.sum(_np.isnan(data), axis = 1), axis=1)
            tp_with_nans = _np.where(n_nans_foreach_timepoint > 0)[0]
            n_ch = data.shape[1]
            n_cp = data.shape[2]
            
            #if they do not:
            if _np.mean(n_nans_foreach_timepoint[tp_with_nans]) != n_ch*n_cp:
                #we cannot remove timepoints with nans, as not all ch / cp have nans
                #at the same timepoints
                if na_action == 'remove':
                    raise ValueError('Nans in the signal, but impossible to remove timepoints as nan values do not share the same timepoints')
            
        #now we can manage the nans 
        #using the xarray.DataArray.interpolate_na or dropna
        if nans_in_dataset:
            if na_action == 'keep':
                print('Nans in the output signal, please check the results')
                return(self.da)
            elif na_action == 'impute':
                signal = self.da.interpolate_na('time', method='cubic')
                #TODO check initial and final nans
                #now we drop them, should we alert the user?
                signal = signal.dropna(dim='time')
                return(signal)
            elif na_action == 'remove':
                #!ATTENTION!
                #if we remove timepoints, then the signal should be considered
                #with an 'unevely' sampling_freq, independently from the fact that 
                #the user provided information about a sampled signal 
                #(e.g. providing a valid sampling_freq value)
                #after all, the default na_action is 'keep' 
                #so we can assume the user knows what is going on here
                signal = self.da.dropna(dim = 'time')
                signal.attrs['sampling_freq'] = 'unevenly'
                return(signal)
        else:
            print('No nans in the signal, no action performed')
            return(self.da)
    
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
            
            if marker is None and self.get_sampling_freq() == 'unevenly':
                marker = '.'

            if marker is None:
                ax.plot(t_, _np.squeeze(v_), linestyle = linestyle)
            elif marker == '|':
                ymin = ax.get_ylim()[0]
                ymax = ax.get_ylim()[1]
                ax.vlines(t_, ymin, ymax, linestyle = linestyle)
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

    #TODO: TEST: HOW THIS SHOULD APPLY TO DATASETS?
    #should we remove all the other signals ('variables')
    #before computing?
    def resample(self, f_out):
        t_start = self.main_signal.p.get_start_time()
        t_end = self.main_signal.p.get_end_time()
        
        t_out = _np.arange(t_start, t_end, 1/f_out)
        resampled_dataset = self.ds.interp(time=t_out, method='cubic')
        resampled_dataset.p.main_signal.attrs['sampling_freq'] = f_out
        return(resampled_dataset)
    
    #TODO: TEST: HOW THIS SHOULD APPLY TO DATASETS?
    #should we remove all the other signals ('variables')
    #before computing?
    def process_na(self, na_action = 'keep'):
        main_signal = self.ds.attrs['MAIN']
        da = self.ds[main_signal]
        processed_da = da.p.process_na(na_action)
        processed_dataset = self.ds
        processed_dataset[main_signal] = processed_da
        if na_action != 'keep':
            processed_dataset = processed_dataset.dropna('time')
        return(processed_dataset)
        
    def plot(self, marker=None, ncols=4, sharey=True):
        self.main_signal.p.plot(marker=marker,
                                ncols=ncols,
                                sharey=sharey)