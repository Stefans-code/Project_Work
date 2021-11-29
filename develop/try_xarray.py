# coding=utf-8
import numpy as _np
import pandas as _pd
import numpy.ma as _ma
from copy import copy
# from scipy import interpolate as _interp
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
     tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
         xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

import xarray as _xr

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
    
    signal = _xr.DataArray(data, dims = dims,
                           coords = coords, 
                           attrs = info,
                           name = name)
    return signal

#%%
import matplotlib.pyplot as plt
data = _np.random.uniform(size = (10000, 5, 2))
sampling_freq = 1000
data = create_signal(data, sampling_freq=sampling_freq, name = 'random')
   
#%%     
@_xr.register_dataarray_accessor('p')
class Signal(object):
    def __init__(self, xdataarray):
        self.da = xdataarray
    
    def clone(self, values, name='signal'):
        assert values.shape[0] == self.da.values.shape[0]
        signal_clone = create_signal(values, times = self.da.coords['time'].values,
                                     name = name, info=self.da.attrs)
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
        return self.da.attr

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

print(data.p.get_sampling_freq())
# data.p.plot()

#%% ALGORITHMS
class Algorithm(object):

    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)  # already checked by __init__
        self.name = ''

    def __call__(self, signal):
        assert isinstance(signal, _xr.DataArray)
        sig_name = signal.name
        dimensions = list(signal.dims)

        if len(dimensions)>1:
            #if multidimensional
            #stack dimensions other than time
            signal_stacked = signal.stack(new=dimensions[1:]).transpose('new', ...)
        else:
            #else create a new fake dimension, for compatibility
            signal_stacked = signal.expand_dims('new', axis=0)
        
        #create a roller over new dimension (
        #so to process one component at a time)
        rr = signal_stacked.rolling(new=1)

        results = []
        for label, arr_window in rr:
            #apply to each component
            res_sig = self.algorithm(arr_window)
            
            #if we lose temporal consistency
            #(this is expected when we extract indicators)
            #save info about start and stop of the original signal
            if res_sig.shape[-1] != signal.shape[0]: 
                t_start = arr_window.coords['time'].values[0]
                t_end = arr_window.coords['time'].values[-1]
                res_sig = res_sig.assign_coords(time=[t_start])
                res_sig = res_sig.assign_coords(t_start=('time', [t_start]))
                res_sig = res_sig.assign_coords(t_end=('time', [t_end]))
            
            results.append(res_sig)
   
        #recompose
        signal_out = _xr.merge(results)
        signal_out = signal_out.transpose('time', ...)

        if len(dimensions)>1:
            signal_out = signal_out.unstack()
        else:
            signal_out = signal_out.squeeze('new')
        
        
        signal_out = signal_out.to_array(name=sig_name+'_'+self.name).squeeze('variable')
        return(signal_out)

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
        self.name = 'Normalize'

    def algorithm(self, signal):
        # print('Normalize.algorithm >>>')
        # print(type(signal))
        # print(signal.shape)
        # print(signal)
        # print('Normalize.algorithm <<<')
        return (signal - _np.mean(signal) +_np.random.uniform(0, 10)) / _np.std(signal)
    
class Mean(Algorithm):
    def __init__(self, **kwargs):
        Algorithm.__init__(self, **kwargs)
        self.name = 'Mean'

    def algorithm(self, signal):
        # print('Mean.algorithm >>>')
        # print(type(signal))
        # print(signal.shape)
        # print(signal)
        mean = _np.mean(signal, keepdims=True)
        
        # print(type(mean))
        # print(mean)
        # print('Mean.algorithm <<<')
        
        return mean

class Min(Algorithm):
    def __init__(self, **kwargs):
        Algorithm.__init__(self, **kwargs)
        self.name = 'Min'

    def algorithm(self, signal):
        # print('Mean.algorithm >>>')
        # print(type(signal))
        # print(signal.shape)
        # print(signal)
        mean = _np.min(signal, keepdims=True)
        
        # print(type(mean))
        # print(mean)
        # print('Mean.algorithm <<<')
        
        return mean
    
# class Test(Algorithm):
#     def __init__(self, **kwargs):
#         Algorithm.__init__(self, **kwargs)

#     def algorithm(self, signal):
#         print('Test.algorithm >>>')
#         print(type(signal))
#         return signal

#%%
rr = Normalize()(data)

plt.plot(data.values[:, 0, 0])
plt.plot(data.values[:, 0, 1])

plt.plot(rr.values[:, 0, 0])
plt.plot(rr.values[:, 0, 1])

result = Mean()(rr)

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
            print(i_seg)
            # result_segment = {'begin': seg.get_begin_time(),
            #                   'end': seg.get_end_time(),
            #                   'label': seg.get_label()}
            
            
            #when called on a signal, a segment returns a portion of the signal
            signal_segment = seg(signal)
            
            res = alg(signal_segment)

            res.assign_coords(label=seg.get_label()[0])
            
            result_algorithm.append(res)
        
        result.append(_xr.concat(result_algorithm, dim='time'))

    result = _xr.merge(result)
    return result

#%%
x = _np.zeros(10000)
x[2500:3000] = 1 
x[5000:8000] = 2

stim = data.p.clone(x, 'stimulus')

segmenter = FixedSegments(0.5, 2, timeline=stim, drop_mixed=False)

result = fmap(segmenter, [Mean(), Min()], data)

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
