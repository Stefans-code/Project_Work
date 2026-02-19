import numpy as _np
from copy import copy as _cpy
import xarray as _xr

class _Segment(object):
    """
    Represents a segment of data defined by a begin and end time.

    Parameters:
        begin (float): The begin time of the segment.
        end (float): The end time of the segment.
        label (optional): The label associated with the segment. Defaults to None.
        signal (optional): The signal associated with the segment. Defaults to None.

    Attributes:
        _begin (float): The begin time of the segment.
        _end (float): The end time of the segment.
        _label: The label associated with the segment.
    
    Methods:
        get_begin_time():
            Returns the begin time of the segment.

        get_end_time():
            Returns the end time of the segment.

        get_label():
            Returns the label associated with the segment.

        __call__(data=None):
            Returns a segment of data based on the begin and end times.

        __repr__():
            Returns a string representation of the segment object.
    """

    def __init__(self, begin, end, label=None, signal=None):
        """
        Initialize a _Segment object with the specified begin and end times.

        Parameters:
            begin (float): The begin time of the segment.
            end (float): The end time of the segment.
            label (optional): The label associated with the segment. Defaults to None.
            signal (optional): The signal associated with the segment. Defaults to None.
        """
        self._begin = begin
        self._end = end
        self._label = label

    def get_begin_time(self):
        """
        Returns the begin time of the segment.

        Returns:
            float: The begin time of the segment.
        """
        return self._begin

    def get_end_time(self):
        """
        Returns the end time of the segment.

        Returns:
            float: The end time of the segment.
        """
        return self._end

    def get_label(self):
        """
        Returns the label associated with the segment.

        Returns:
            float: The label associated with the segment.
        """
        return float(self._label)

    def __call__(self, data=None):
        """
        Returns a segment of data based on the begin and end times.

        Parameters:
            data (optional): The data to segment. Defaults to None.

        Returns:
            object: A segment of data defined by the begin and end times.
        """
        data_segment = data.p.segment_time(self.get_begin_time(), self.get_end_time())
        return data_segment

    def __repr__(self):
        """
        Returns a string representation of the segment object.

        Returns:
            str: A string representation of the segment object.
        """
        return '[%s:%s' % (str(self.get_begin_time()), str(self.get_end_time())) + (
            ":%s]" % self._label if self._label is not None else "]")

class _SegmentationIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, _Segmenter)
        self._win = _cpy(win)

    def __next__(self):
        return self._win.next_segment()

    # Python 2 & users compatibility
    def next(self):
        return self.__next__()
    
class _Segmenter(object):
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

        s = _Segment(b, e, label)
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
        return _SegmentationIterator(self)

    def __repr__(self):
        if self.reference is not None:
            message = self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']
            return message + " over\n" + str(self.reference)
        else:
            message = self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']
            return message

class FixedSegments(_Segmenter):
    """
    Segmenter that divides a signal into fixed-width segments.

    This class is a subclass of `_Segmenter` and implements a segmentation algorithm
    that divides a signal into fixed-width segments. The width of the segments is
    determined by the `width` parameter, or if not specified, it is set equal to the
    `step` parameter. The `step` parameter determines the distance between the
    starting points of consecutive segments.

    Parameters
    ----------
    step : float
        The distance between the starting points of consecutive segments.
        Must be greater than 0.
    width : float, optional
        The width of each segment. If not specified, it is set equal to `step`.
        Must be greater than 0 if provided.
    timeline : signal, optional
        The signal used to obtain the information about the timeline of the experiment. 
        If not provided, the reference timeline of the
        `_Segmenter` base class will be used.
    drop_mixed : bool, optional
        Whether to drop segments that contain mixed annotations.
        Default is True.
    drop_cut : bool, optional
        Whether to drop segments that have been manually cut by a user.
        Default is True.
    **kwargs : dict
        Additional keyword arguments to be passed to the base `_Segmenter` class.
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
        e = b + self._width - 0.0001
        
        if b >= self.reference.p.get_end_time():
            raise StopIteration()
        
        return self.manage_drops(b, e)

class CustomSegments(_Segmenter):
    """
    A class for segmenting data based on custom segment boundaries.

    This class extends the `_Segmenter` base class and allows segmentation of data based on specified segment boundaries.
    Each segment is defined by a beginning and an end index.

    Parameters
    ----------
    begins : list
        A list of integers representing the beginning indices of each segment.
    ends : list
        A list of integers representing the end indices of each segment.
    timeline : list, optional
        A list of labels for each segment. If provided, should have the same length as begins and ends.
    drop_mixed : bool, optional
        A boolean value indicating whether to drop mixed segments. Default is True.
    drop_cut : bool, optional
        A boolean value indicating whether to drop cut segments. Default is True.
    **kwargs
        Additional keyword arguments to be passed to the base class constructor.

    Raises
    ------
    AssertionError
        If the length of begins is not equal to the length of ends.

    """

    def __init__(self, begins, ends, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        #TODO: timeline can also be a list with labels of each segment
        super(CustomSegments, self).__init__(labels=labels, drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        
        assert len(begins) == len(ends), "The number of begins has to be equal to the number of ends :)"
        if (labels is not None):
            assert len(labels) == len(begins)
        self._i = -1
        self._b = begins
        self._e = ends
        self._labels = labels

    def _next_segment(self):
        self._i += 1
        if self._i < len(self._b):
            b = self._b[self._i]
            e = self._e[self._i]
            l = self._labels[self._i]
            # b, e, _ = self.manage_drops(b, e)
            # if (self._labels is not None):
            #     l = self._labels[self._i]
            return(b, e, l)
                
        else:
            raise StopIteration()

class LabelSegments(_Segmenter):
    """
    Segmenter class for creating custom segments frmo a signal.

    Parameters
    ----------
    timeline : signal
        A signal that represents the experimental timeline.
    drop_mixed : bool, optional
        Whether to drop segments with mixed values. Defaults to True.
    drop_cut : bool, optional
        Whether to drop segments that have been cut by previous segmenters. Defaults to True.
    **kwargs
        Additional keyword arguments to be passed to the base class.

    """
    def __init__(self, timeline, drop_mixed=True, drop_cut=True, **kwargs):
        super(LabelSegments, self).__init__(timeline=timeline, drop_mixed=drop_mixed, drop_cut=drop_cut, **kwargs)
        self._i = 0
        
    def _next_segment(self):
        timeline_values = self.timeline.p.main_signal.values
        if self._i >= len(timeline_values):
            raise StopIteration()
        end = self._i
        
        while end < len(timeline_values) and timeline_values[self._i] == timeline_values[end]:
            end += 1
        
        b = self.timeline.p.get_times()[self._i]
        e = self.timeline.p.get_times()[end-1]
        self._i = end
        return b, e, timeline_values[end-1]

class RandomFixedSegments(_Segmenter):
    """
    A class that generates random fixed-width segments from a reference signal or a timeline.

    Parameters:
    -----------
    N : int
        The number of segments to generate.
    width : float
        The width of each segment in time units.
    reference : signal, optional
        The reference signal from which to generate segments. Either `reference` or `timeline` should be provided.
    timeline : signal, optional
        A signal that represents the experimental timeline. Either `reference` or `timeline` should be provided.
    drop_mixed : bool, default=True
        Specifies whether to drop mixed segments that overlap with multiple labels.
    drop_cut : bool, default=True
        Specifies whether to drop cut segments that overlap with cut annotations.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the base `_Segmenter` class.

    Raises:
    -------
    AssertionError:
        - If `N` is not greater than 0.
        - If `width` is not greater than 0.
        - If neither `reference` nor `timeline` is provided.
    """

    def __init__(self, N, width, reference=None, timeline=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(RandomFixedSegments, self).__init__(timeline=timeline, drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        assert N > 0
        assert width > 0
        
        assert (reference is not None) or (timeline is not None), "Either a reference signal or a timeline should be provided"
        
        self._N = N
        self._width = width
        self._i = -1
        self.reference = reference
        self.timeline = timeline
        
        
                
        if reference is None:
            t_st = self.timeline.p.get_start_time()
            t_sp = self.timeline.p.get_end_time() - self._width
            tst = _np.random.uniform(t_st, t_sp, self._N)
        else:
            t_st = self.reference.p.get_start_time()
            t_sp = self.reference.p.get_end_time() - self._width
            tst = _np.random.uniform(t_st, t_sp, self._N)

        #timestamps should be strictly (--> _np.unique) monotonic
        self.tst = _np.unique(tst[_np.argsort(tst)])
            
    def _next_segment(self):
        self._i += 1
        if self._i < self._N:
            b = self.tst[self._i]
            e = b + self._width
            return self.manage_drops(b, e)
        else:
            raise StopIteration()
            
def fmap(segmenter, algorithms, signal):
    """
    Apply a series of algorithms to segments of a signal using a segmenter.

    Parameters
    ----------
    segmenter : Segmenter object
        An object responsible for segmenting the signal into segments.
        It should have a `reference` attribute that indicates whether the
        segmenter has been initialized or not. If `reference` is None,
        the segmenter will be applied to the signal to generate segments.
        The segmenter should support iteration, where each iteration
        generates a segment of the signal.

    algorithms : list
        A list of algorithm objects to be applied to each segment of the signal.
        Each algorithm should have a callable `__call__` method that takes
        a signal segment as input and returns a result.

    signal : Signal object
        The input signal to be segmented and processed.

    Returns
    -------
    result : xarray Dataset
        A merged xarray Dataset containing the results of applying the algorithms
        to the segments of the signal. Each algorithm's results are stored as
        separate variables in the Dataset. The time dimension is defined by the
        segments, and each segment is labeled with a unique identifier.

    Notes
    -----
    This function applies a series of algorithms to segments of a signal. It
    first checks if the segmenter has been initialized by examining the
    `reference` attribute. If `reference` is None, the segmenter is applied
    to the signal to generate segments.

    For each algorithm and segment combination, the function applies the algorithm
    to the corresponding segment of the signal. The resulting xarray Dataset
    is modified to remove the original signal variable, drop NaN values along
    the time dimension, assign segment labels, and store the result in a list.

    Finally, the function merges the results of all algorithms into a single
    xarray Dataset and returns it.
    """
    if segmenter.reference is None:
        segmenter(signal)
    
    result = []
    
    signal_name = signal.p.main_signal.name
    #for all algorithms
    for alg in algorithms:
        # print(alg.name)
        result_algorithm = []
        for i_seg, seg in enumerate(segmenter): #this generates segments from the segmenter
            # print(seg.get_begin_time())    
            signal_segment = seg(signal)
            if signal_segment.p.get_values().shape[0] > 0:
                res = alg(signal_segment, add_signal=True)
                res = res.drop(signal_name)
                
                res_out = res.copy()
                
                res_out = res_out.dropna(dim='time', 
                                         how='all', 
                                         subset=[f'{signal_name}_{alg.__repr__()}'])
                
                #if the result of the computation of the indicator is na
                if res_out.dims['time'] == 0:
                    res_out = res.isel({'time':[0]})
                    
            res_out = res_out.assign_coords(label=('time', [seg.get_label()]))
            result_algorithm.append(res_out)

        result.append(_xr.concat(result_algorithm, dim='time'))

    result = _xr.merge(result,compat='override')
    return result

def indicators2df(fmap_results):
    import pandas as _pd

    k = list(fmap_results.keys())

        
    ind_sample = fmap_results[k[0]]
    assert ind_sample.ndim <=3, "computed results have more than three dimensions"
    n_channels = ind_sample.p.get_nchannels()
    n_components = ind_sample.p.get_ncomponents()
    
    t = ind_sample.p.get_times()
    label = ind_sample['label'].p.get_values().ravel()
    
    df_all = []
    for i_comp in range(n_components):
        for i_chan in range(n_channels):
            
            indicator_df = {}
            indicator_df['time'] = t
            indicator_df['label'] = label
    
            for key in list(fmap_results.keys()):
                result_key = fmap_results[key]
                
                if ind_sample.ndim == 3:
                    indicator_df[key] = result_key.p.get_values()[:, i_chan, i_comp].ravel()
                    indicator_df['component'] = _np.repeat(i_comp+1, len(t))
                    indicator_df['channel'] = _np.repeat(i_chan+1, len(t))
                    
                else:
                    if ind_sample.ndim == 2:
                        indicator_df[key] = result_key[:, i_chan].p.get_values().ravel()
                        indicator_df['channel'] = _np.repeat(i_chan+1, len(t))
                    else:
                        indicator_df[key] =  result_key
            
            df_all.append(_pd.DataFrame(indicator_df))
    
    
    df_all = _pd.concat(df_all, axis = 0)
    return(df_all)
