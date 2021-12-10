# coding=utf-8
import numpy as _np
import pandas as _pd
from copy import copy
import scipy.stats as sps

# from scipy import interpolate as _interp
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

import xarray as _xr

_xr.set_options(keep_attrs = True)

#%% SIGNAL
def create_signal(data, times=None, sampling_freq=None,
                  start_time=0, name='signal', info={}):
    
    assert (times is None) ^ (sampling_freq is None), "Either times or sampling freq"
    
    if sampling_freq is None: #defined by times
        assert len(times) == data.shape[0]
        
        #check if there is a fsamp, else fsamp is None (unevenly)
        dt = _np.unique(_np.diff(times))
        
        if len(dt)==1:
            sampling_freq = 1/(dt[0]/_np.timedelta64(1, 's'))
        
        #create pandas TimedeltaIndex times
        times = _pd.to_timedelta(times, unit='s')
        
    else: #defined by sampling freq
        assert sampling_freq > 0
        #define times as TimedeltaIndex
        times = _pd.timedelta_range(start=start_time, periods = data.shape[0], freq=f'{10**9/sampling_freq}ns')
        
    #start_time is times[0]
    start_time = times[0]
        
    #check dims and set coordinates
    dims_template = ('time', 'channel', 'component')
    assert data.ndim <= 3
    dims = dims_template[:data.ndim]
    
    coords = {'time':times}
    
    for i_dim, name_dim in enumerate(dims[1:]):
        coords[dims_template[i_dim+1]] = _np.arange(data.shape[i_dim+1])
        
    
    info['sampling_freq'] = sampling_freq
    info['start_time'] = start_time
        
    signal_ = _xr.DataArray(data, dims = dims,
                           coords = coords, 
                           attrs = info,
                           name = name).to_dataset()
    
    signal_.attrs['MAIN'] = name
    # signal.attrs['steps'] = ['created']
    
    return signal_

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
        
        sub_dataset = self.da.sel(time = slice(t_start_timedelta,
                                               t_stop_timedelta))
        return sub_dataset
    
    def get_start_time(self):
        times= self.get_times()
        return(times[0])
    
    def get_end_time(self):
        times= self.get_times()
        return(times[-1])

    def get_sampling_freq(self):
        return self.da.attrs['sampling_freq']
    
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
        return self.main_signal.values
    
    def get_times(self):
        time = self.main_signal.coords['time'].values
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
        return self.main_signal.attrs['sampling_freq']
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()

    def get_info(self):
        return self.ds.attrs
    
    def has_multi_channels(self):
        return(len(self.main_signal.dims)>1)
    
    def get_nchannels(self):
        if self.has_multi_channels():
            return(int(_np.max(self.main_signal.coords['channel'])+1))
        else:
            return(1)
    
    def has_multi_components(self):
        return(len(self.main_signal.dims)>2)
    
    def get_ncomponents(self):
        if self.has_multi_components():
            return(int(_np.max(self.main_signal.coords['component'])+1))
        else:
            return(1)

"""
    # def plot(self, marker=None, ncols=4, sharey=True):
    #     fig = _gcf()
    #     t_ = self.get_times()
    #     v_ = self.get_values()
    #     linestyle='solid'
    #     #if single signal, then plot
    #     if len(self.da.dims) == 1:
            
    #         #TODO if existing figure has many axes, 
    #         #replicate the plot on each axis
            
    #         #if good then use asolid line
    #         #else use a dotted line
    #         # linestyle='solid'
    #         # if self.has_good():
    #         #     good = self.get_good()
    #         #     if len(good)==0:
    #         #         linestyle = 'dotted'
            
    #         #plot the signal
    #         ax = _gca()
            
    #         if marker is None:                
    #             ax.plot(t_, _np.squeeze(v_), linestyle = linestyle)
    #         else:
    #             ax.plot(t_, _np.squeeze(v_), marker, linestyle = linestyle)
    #         _grid(True)
        
    #     else:
    #         n_ch = self.get_nchannels()
    #         n_comp = self.get_ncomponents()

    #         #if existing figure has enough number of axes
    #         #use the figure
    #         if len(fig.axes)>= n_ch:
    #             axes = fig.axes
            
    #         #else create a new figure 
    #         else: 
    #             if n_ch>1:
    #                 #compute number of cols and rows and create a new figure
    #                 n_cols = n_ch if n_ch < ncols else ncols
    #                 n_rows = int(_np.ceil(n_ch/n_cols))
                    
    #                 fig, axes = _subplots(n_rows, n_cols, num = fig.number, sharex=True, sharey=sharey)
    #                 axes = axes.ravel()
    #             else:
    #                 fig, axes = _subplots(1, 1, num = fig.number, sharex=True, sharey=sharey)
    #                 axes = [axes]
            
    #         #recursive calls to signal.plot()
    #         #for each channel and component
    #         for i_ch in range(n_ch):
    #             _sca(axes[i_ch])
    #             ax = _gca()
    #             if n_comp>1:
    #                 for i_comp in range(n_comp):
    #                     if marker is None:                
    #                         ax.plot(t_, v_[:,i_ch, i_comp], linestyle = linestyle)
    #                     else:
    #                         ax.plot(t_, v_[:,i_ch, i_comp], marker, linestyle = linestyle)
    #             else:
    #                 if marker is None:                
    #                     ax.plot(t_, v_[:,i_ch], linestyle = linestyle)
    #                 else:
    #                     ax.plot(t_, v_[:,i_ch], marker, linestyle = linestyle)

    #             _ylabel(i_ch)
    #             _grid(True)
                
    #         _xlim(self.get_start_time(), self.get_end_time())
    #         _tight_layout()
    #         _subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
"""

#%%
import matplotlib.pyplot as plt
data = _np.random.uniform(size = (1000, 1, 1))
sampling_freq = 1000
SIGNAL = create_signal(data, sampling_freq=sampling_freq, name = 'random')
   

print(SIGNAL.p.get_sampling_freq())
# data.p.plot()

# def mean(x):
#     return(_np.mean(x, keepdims=True))

# signal_chunk = signal.chunk({'channel':1, 'component':1})
# mapper = _xr.map_blocks(mean, signal_chunk, template = signal_chunk)

#%% ALGORITHMS
class Algorithm(object):
    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)
        self.name = ''
        self.dimensions = {}
        

    def __mapper_func__(self, signal_in):
        print('-----> Algorithm.__mapper_func__')
        dimensions = ('time', 'channel', 'component')
        for dim in dimensions:
            if dim not in signal_in.dims:
                signal_in = signal_in.expand_dims({dim:1})
        signal_in = signal_in.transpose(*dimensions)
        result_numpy = self.algorithm(signal_in)
        result_out = self.__finalize__(result_numpy, signal_in)
        print('<----- Algorithm.__mapper_func__')
        return(result_out)

    def __call__(self, signal_in, by=None, add_result=False, rename_signal=False):
        #This function iteratively calls the self.algorithm on each signal
        #(i.e. channel+component)
        print('-----> Algorithm.__call__')
        
        #The user will mainly call Algorithms on a Dataset
        #but the __call__ rolling mechanism assumes to operate on DataArray
        if isinstance(signal_in, _xr.Dataset):
            signal_ = signal_in.p.main_signal
        else:
            signal_ = signal_in.copy(deep=True)
        
        signal_name = signal_.name
        #from here, signal is a DataArray
       
        chunk_dict = {}
        for dim in ('time', 'channel', 'component'):
            if dim not in self.dimensions.keys():
                chunk_dict[dim] = 1
        
        signal_dask = signal_.chunk(chunk_dict)
        
        print('>>>>>>>>>>>>>>>> signal_dask')
        print(signal_dask)
        
        template_shape = []
        for dim in ('time', 'channel', 'component'):
            out_dim = signal_.sizes[dim]
            
            if dim in self.dimensions.keys() and self.dimensions[dim] != 0:
                out_dim = self.dimensions[dim]
           
            template_shape.append(out_dim)
                
        output = _np.zeros(template_shape)
        template = create_signal(output, 
                                 times=signal_.coords['time'].values[:output.shape[0]])
        
        print('>>>>>>>>>>>>>>>> template')
        print(template)
        
        template_dask = template.chunk(chunk_dict)
        
        print('>>>>>>>>>>>>>>>> template_dask')
        print(template_dask)
        
        mapper =  _xr.map_blocks(self.__mapper_func__, 
                                 signal_dask, 
                                 template = template_dask)
        
        print('>>>>>>>>>>>>>>>> mapper')
        print(type(mapper))
        print(mapper)
        
        signal_out = mapper.load()
        print('>>>>>>>> here')
        # signal_out = self.__fix_dims__(signal_out, signal_in)
    
        #signal_out is a DataArray
        
        #The user will mainly call Algorithms on a Dataset
        #so it will expect a Dataset as result
        #But also consider the add_result option
        if isinstance(signal_in, _xr.Dataset) or add_result:
            #transform to Dataset
            signal_ds_out = signal_in.copy(deep=True)
            
            result_name = signal_name+'_'+self.name
            
            if add_result:
                #this is only used when the Algorithm is called by the user
                #create a new name and add the result
                signal_ds_out = signal_ds_out.assign({result_name:signal_out})
            else:
                #drop the original signal and substitute with the result
                signal_ds_out = signal_ds_out.drop(signal_name)
                if rename_signal:                    
                    signal_ds_out = signal_ds_out.assign({result_name:signal_out})
                else:
                    signal_ds_out = signal_ds_out.assign({signal_name:signal_out})
                
            print('<----- Algorithm.__call__')
            return(signal_ds_out)
        else:
            return(signal_out)
    
    # def __fix_dims__(signal_out, signal_in):
        
    #     return(signal_out)
    
    def __finalize__(self, result, signal_in):
        '''
        General function to obtain a coherent output 
        from the calls to self.algorithm.
        The output should be a dataaarry or dataset
        '''
        
        def _numpy_to_dataarray(numpy_out, signal_in):
            # if one dimensional, assume the processed dimension is time
            if numpy_out.ndim == 1:
                numpy_out = _np.expand_dims(numpy_out, [1,2])
            
            signal_out = _xr.DataArray(numpy_out, dims=('time', 'channel', 'component'))
            
            for i_dim in range(3):
                dim = signal_in.dims[i_dim]
                
                if signal_in.shape[i_dim] == signal_out.shape[i_dim]:
                    signal_out = signal_out.assign_coords({dim:signal_in.coords[dim].values})
                else:
                    coord_start = signal_in.coords[dim].values[0]
                    # coord_stop = signal_in.coords[dim].values[-1]
                    
                    if signal_out.shape[i_dim] == 1: #indicators or windowed algorithms
                        signal_out = signal_out.assign_coords({dim:[coord_start]})
                        
                        # signal_out = signal_out.assign_coords({f'{dim}_start': (dim, [coord_start])})
                        # signal_out = signal_out.assign_coords({f'{dim}_stop': (dim, [coord_stop])})
            
            signal_out.attrs = signal_in.attrs.copy()
            return(signal_out)
        
        print('-----> Algorithm.__finalize__')
        
        if isinstance(result, list):
            output = [_numpy_to_dataarray(x, signal_in) for x in result]
        else:
            output = _numpy_to_dataarray(result, signal_in)
        print('<----- Algorithm.__finalize__')
        return(output)

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def algorithm(cls, signal222):
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
        self.name = 'Mean'
        self.dimensions = {'time':0}

    def algorithm(self, signal1):
        print('-----> Mean.algorithm')
        # mean = _np.mean(signal.values, keepdims=True)
        print(type(signal1))
        print(signal1)
        print('<----- Mean.algorithm')
        return signal1.values

#%%
result_mean = Mean()(SIGNAL)

#%%
class Normalize(Algorithm):
    def __init__(self, norm_method='standard', norm_bias=0, norm_range=1, **kwargs):
        assert norm_method in ['mean', 'standard', 'min', 'maxmin', 'custom'],\
            "norm_method must be one of 'mean', 'standard', 'min', 'maxmin', 'custom'"
        if norm_method == "custom":
            assert norm_range != 0, "norm_range must not be zero"
        Algorithm.__init__(self, norm_method=norm_method, norm_bias=norm_bias, norm_range=norm_range, **kwargs)

    def algorithm(self, signal):
        # from ..indicators.timedomain import Mean as _Mean, StDev as _StDev, Min as _Min, Max as _Max
        params = self._params
        method = params['norm_method']
        signal_values = signal.values
        if method == "mean":
            return signal_values - Mean()(signal).values
        elif method == "standard":
            return (signal_values - Mean()(signal).values + _np.random.uniform(5,10)) / StDev()(signal).values
        elif method == "custom":
            result = (signal_values - params['norm_bias']) / params['norm_range']
            return result
        else:
            raise ValueError



result = Normalize()(signal)

plt.plot(signal.p.main_signal.values[:, 0, 0])
plt.plot(signal.p.main_signal.values[:, 0, 1])

plt.plot(result.p.main_signal.values[:, 0, 0])
plt.plot(result.p.main_signal.values[:, 0, 1])

#%% SEGMENTERS
class Segment(object):
    """
    Base Segment, a time begin-end pair with a reference to the base signal and a name.
    """

    def __init__(self, begin, end, label=None, signal=None):
        """
        Creates a base Window
        @param begin: Begin sample index
        @param end: End sample index
        """
        self._begin = begin
        self._end = end
        self._label = label

    def get_begin_time(self):
        return self._begin

    def get_end_time(self):
        return self._end

    def get_label(self):
        return self._label

    def __call__(self, data=None):
        return data.p.segment_time(self.get_begin_time(), self.get_end_time())

class SegmentationIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, Segmenter)
        self._win = copy(win)

    def __next__(self):
        return self._win.next_segment()

    # Python 2 & users compatibility
    def next(self):
        return self.__next__()
    
class Segmenter(object):
    # Assumed: timeline signal extended over the end by holding the value

    def __init__(self, timeline=None, drop_cut=True, drop_mixed=True, **kwargs):
        self._params = {}
        self._params['drop_cut'] = drop_cut
        self._params['drop_mixed'] = drop_mixed
        self._params.update(kwargs)
        self.timeline = timeline
        self.reference = None
        

    def next_segment(self):
        assert self.reference is not None
        label = b = e = None
        while True:
            # break    ==> keep
            # continue ==> drop
            b, e, label = self._next_segment()
            #accoriding to segmentation method and params
            #b is None if the segment shold be discarded
            if b is None: 
                continue
            break

        s = Segment(b, e, label)
        return s
    
    def manage_drops(self, b, e):
        assert self.reference is not None
        #manage drop_cut
        
        
        if e >= self.reference.p.get_end_time():
            if self._params['drop_cut']:
                return([None, None, None])
        
        #manage labels, drop_mixed
        if self.timeline is not None:
            
            timeline_segment = self.timeline.p.segment_time(b, e).p.get_values()

            if (timeline_segment == timeline_segment[0]).all():
                #timeline values are the same within the segment
                label = _np.array(timeline_segment[0]).ravel()
                return([b, e, label])
            else:
                #timeline values change within the segment
                if self._params['drop_mixed']:
                    return([None, None, None])
                else:
                    return([b, e, _np.array([_np.nan])])
        else:
            return([b, e, _np.nan])
    
    def __call__(self, reference=None):
        if reference is not None:
            self.reference = reference
        else:
            assert self.timeline is not None
            self.reference = self.timeline
        
    @classmethod
    def _next_segment(self):
        pass

    def __iter__(self):
        return SegmentationIterator(self)

class FixedSegments(Segmenter):
    """
    Fixed length segments iterator, specifying step and width in seconds.

    A label signal from which to
    take labels can be specified.

    Parameters
    ----------
    step : float, >0
        time distance between subsequent segments.

    Optional parameters
    -------------------
    width : float, >0, default=step
        time distance between subsequent segments.
    start : float
        start time of the first segment
    labels : array
        Signal of the labels
    drop_mixed : bool, default=True
        In case labels is specified, whether to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Whether to drop segments that are shorter due to the crossing of the signal end.
    """

    def __init__(self, step, width=None, timeline=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(FixedSegments, self).__init__(timeline=timeline, drop_mixed=drop_mixed, drop_cut=drop_cut, **kwargs)
        assert step > 0
        assert width is None or width > 0
        
        self._step = step
        self._width = width if width is not None else step
        self._t = None
        
    def _next_segment(self):
        if self._t is None:
            self._t = self.reference.p.get_start_time()
        b = self._t
        
        self._t += self._step
        e = b + self._width
        
        if b >= self.reference.p.get_end_time():
            raise StopIteration()
        
        return self.manage_drops(b, e)

def fmap(segmenter, algorithms, signal):
   
    if segmenter.reference is None:
        segmenter(signal)
    
    result = []
    
    #for all algorithms
    for alg in algorithms:
        
        result_algorithm = []
        for i_seg, seg in enumerate(segmenter): #this generates segments from the segmenter
            signal_segment = seg(signal)
            
            res = alg(signal_segment, rename_signal=True)
            res = res.dropna(dim='time')
            res = res.assign_coords(label=('time', [seg.get_label()[0]]))
            
            result_algorithm.append(res)
        
        result.append(_xr.concat(result_algorithm, dim='time'))

    result = _xr.merge(result)
    return result

#%%
x = _np.zeros(10000)
x[2500:3000] = 1 
x[5000:8000] = 2

stim = create_signal(x, sampling_freq=1000, name = 'stimulus')

segmenter = FixedSegments(0.5, 2, timeline=stim, drop_mixed=False)

result = fmap(segmenter, [Mean(), StDev()], signal)

#%% SIGNAL QUALITY
class SignalQualityIndicator(Algorithm):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.

    Args:
        threshold (low, high): The range within which the sqi indicates good quality
    
    Returns:
        result (sqi, isgood): Tuple containing the value of the sqi and if it corresponds to good quality
    
    """
    def __init__(self, threshold, **kwargs):
        '''
        '''
        assert len(threshold)==2
        Algorithm.__init__(self, threshold=threshold, **kwargs)
    
    def is_good(self, sqi_values):
        # print('-----> is_good')
        params = self._params
        threshold = params['threshold']
        if sqi_values.ndim == 0:
            output = (sqi_values >= threshold[0]) & (sqi_values <= threshold[1])
            output = _np.array(output)
        else:
            output = _np.zeros_like(sqi_values)
            idx_good = _np.where((sqi_values >= threshold[0]) & (sqi_values <= threshold[1]))
            output[idx_good] = 1
        
        # print('<----- is_good')
        return(output)
        
    def __call__(self, signal, add_result=False, rename_signal=False):
        # print('-----> SQI.__call__()')
        values_out = super().__call__(signal)
        values_out = values_out.rename({signal.p.main_signal.name:self.name})
        isgood = self.is_good(values_out[self.name])

        #convert isgood to dataarray
        isgood_out = values_out[self.name].copy(data = isgood)
        isgood_name = self.name +'_isgood'
        isgood_out.name = isgood_name

        out = _xr.merge([values_out, isgood_out])
        # print('<----- SQI.__call__()')
        return(out)

class Kurtosis(SignalQualityIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        SignalQualityIndicator.__init__(self, threshold, **kwargs)
        self.name = 'Kurtosis'

    def algorithm(self, data):
        print('-----> Kurtosis')
        # print(type(data))
        # print(data.shape)
        k = sps.kurtosis(data.values.ravel())
        
        k_out = _np.array([[k]])
        # print(k_out.shape)
        
        print('<----- Kurtosis')
        return(k_out)

#%%
K = Kurtosis(threshold=[2, 10])
k_out = K(signal)

#%%
k_win = fmap(segmenter, [K, K], signal)

ttt = _xr.merge([signal, k_win, result])

plt.plot(ttt['random'].p.get_times(), ttt['random'].values[:,0,0])
plt.plot(ttt['random_Mean'].p.get_times(), ttt['random_Mean'].values[:,0,0], '.')

#%%
def get_portions(x):
    idx_changes = _np.where(_np.diff(x) != 0)[0]
    portions = _np.zeros(len(x))
    idx_start_portion = 0
    for id_portion, idx_end_portion in enumerate(idx_changes):
        portions[idx_start_portion:idx_end_portion] = id_portion
        idx_start_portion = idx_end_portion
    portions[idx_start_portion:] = portions[idx_start_portion-1]+1
    return(portions)

portions = get_portions(x)

portions = data.p.clone(portions, 'portions')

data = data.merge(stim, join='left')
data = data.merge(portions, join='left')
res = data.groupby('portions', restore_coord_dims=True).apply(Mean())

#%%
plt.plot(data.coords['time'], data.portions.values)
plt.plot(res.coords['time'], res.signal.values[:, 0,0], 'o-')
plt.grid()
