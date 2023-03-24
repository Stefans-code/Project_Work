import numpy as _np
from copy import copy as _cpy
import xarray as _xr

class _Segment(object):
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
        return float(self._label)

    def __call__(self, data=None):
        data_segment = data.p.segment_time(self.get_begin_time(), self.get_end_time())
        return data_segment

    def __repr__(self):
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
        e = b + self._width - 0.0001
        
        if b >= self.reference.p.get_end_time():
            raise StopIteration()
        
        return self.manage_drops(b, e)

class CustomSegments(_Segmenter):
    """
    Custom segments iterator, specifying an array of begin times and an array of end times.

    Parameters
    ----------
    begins : array or list
        Array of the begin times of the segments to return.
    ends : array or list
        Array of the end times of the segments to return, of the same length of 'begins'.

    Optional parameters
    -------------------
    labels : array or list
        Signal of the labels
    drop_mixed : bool, default=True
        In case labels is specified, weather to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Weather to drop segments that are shorter due to the crossing of the signal end.
    """

    def __init__(self, begins, ends, timeline=None, drop_mixed=True, drop_cut=True, **kwargs):
        #TODO: timeline can also be a list with labels of each segment
        super(CustomSegments, self).__init__(timeline=timeline, drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        
        assert len(begins) == len(ends), "The number of begins has to be equal to the number of ends :)"
        self._i = -1
        self._b = begins
        self._e = ends

    def _next_segment(self):
        self._i += 1
        if self._i < len(self._b):
            b = self._b[self._i]
            e = self._e[self._i]
            return self.manage_drops(b, e)
        else:
            raise StopIteration()

class LabelSegments(_Segmenter):
    """
    Generates a list of segments from a label signal, allowing to collapse subsequent equal samples.

    Parameters
    ----------
    labels : array or list
        Signal of the labels

    Optional parameters
    -------------------
    drop_mixed : bool, default=True
        In case labels is specified, weather to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Weather to drop segments that are shorter due to the crossing of the signal end.
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
    Fixed length segments iterator, at random start timestamps, specifying step and width in seconds.

    A label signal from which to
    take labels can be specified.

    Parameters
    ----------
    width : float, >0
        time distance between subsequent segments.
    N : int, >0
        number of segments to be extracted.

    Optional parameters
    -------------------
    labels : array
        Signal of the labels
    drop_mixed : bool, default=True
        In case labels is specified, whether to drop segments with more than one label, if False the label of such
         segments is set to None.
    drop_cut : bool, default=True
        Whether to drop segments that are shorter due to the crossing of the signal end.
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
            res = alg(signal_segment, add_signal=True)
            res = res.drop(signal_name)
            res = res.dropna(dim='time', 
                             how='all', 
                             subset=[f'{signal_name}_{alg.name}'])
            res = res.assign_coords(label=('time', [seg.get_label()]))
            
            result_algorithm.append(res)

        result.append(_xr.concat(result_algorithm, dim='time'))

    result = _xr.merge(result)
    return result

#TODO: needed? if yes, fix--->
'''
def indicators2df(fmap_results):
    import pandas as _pd

    k = list(fmap_results.keys())[0]
    
    for k,v in fmap_results.items():
        assert isinstance(v, _Signal), 'Provided fmap_results should be all Signals'
        
    ind_sample = fmap_results[k]
    assert ind_sample.ndim <=3, "computed results have more than three dimensions"
    n_channels = ind_sample.get_nchannels()
    n_components = ind_sample.get_ncomponents()
    
    t = ind_sample.get_times()
    label = ind_sample.get_info()['label'].get_values().ravel()
    
    df_all = []
    for i_comp in range(n_components):
        for i_chan in range(n_channels):
            
            indicator_df = {}
            indicator_df['time'] = t
            indicator_df['label'] = label
    
            for key in list(fmap_results.keys()):
                result_key = fmap_results[key]
                
                if ind_sample.ndim == 3:
                    indicator_df[key] = result_key.get_values()[:, i_chan, i_comp].ravel()
                    indicator_df['component'] = _np.repeat(i_comp+1, len(t))
                    indicator_df['channel'] = _np.repeat(i_chan+1, len(t))
                    
                else:
                    if ind_sample.ndim == 2:
                        indicator_df[key] = result_key[:, i_chan].get_values().ravel()
                        indicator_df['channel'] = _np.repeat(i_chan+1, len(t))
                    else:
                        indicator_df[key] =  result_key
            
            df_all.append(_pd.DataFrame(indicator_df))
    
    
    df_all = _pd.concat(df_all, axis = 0)
    return(df_all)
'''