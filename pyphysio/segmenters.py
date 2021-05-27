# coding=utf-8
import numpy as _np
from copy import copy as _cpy
# from numpy import asarray as _asarray
# from ..Utility import abstractmethod as _abstract
from .signal import EvenlySignal as _EvenlySignal, Signal as _Signal, UnevenlySignal as _UnevenlySignal
# from numbers import Number as _Number
# __author__ = 'AleB'

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
        return data.segment_time(self.get_begin_time(), self.get_end_time())

    def __repr__(self):
        return '[%s:%s' % (str(self.get_begin_time()), str(self.get_end_time())) + (
            ":%s]" % self._label if self._label is not None else "]")

class SegmentationIterator(object):
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

        s = Segment(b, e, label)
        return s
    
    def manage_drops(self, b, e):
        assert self.reference is not None
        #manage drop_cut
        if e >= self.reference.get_end_time():
            if self._params['drop_cut']:
                return([None, None, None])
        
        #manage labels, drop_mixed
        if self.timeline is not None:
            timeline_segment = self.timeline.segment_time(b, e)
            
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
            assert isinstance(reference, _Signal), "The provided reference signal should be a Signal"
            self.reference = reference
        else:
            assert self.timeline is not None, "The timeline should be not None if not providing a reference signal"
            self.reference = self.timeline
        
    @classmethod
    def _next_segment(self):
        pass

    def __iter__(self):
        return SegmentationIterator(self)

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
        assert timeline is None or isinstance(timeline, _EvenlySignal),\
            "The parameter 'labels' should be an EvenlySignal."
        
        super(FixedSegments, self).__init__(timeline=timeline, drop_mixed=drop_mixed, drop_cut=drop_cut, **kwargs)
        assert step > 0
        assert width is None or width > 0
        
        self._step = step
        self._width = width if width is not None else step
        self._t = None
        
    def _next_segment(self):
        if self._t is None:
            self._t = self.reference.get_start_time()
        b = self._t
        self._t += self._step
        e = b + self._width
        
        if b >= self.reference.get_end_time():
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
        assert timeline is None or isinstance(timeline, _EvenlySignal),\
            "The parameter 'labels' should be an EvenlySignal."
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
        assert timeline is None or isinstance(timeline, _EvenlySignal),\
            "The parameter 'labels' should be an EvenlySignal."
        super(LabelSegments, self).__init__(timeline=timeline, drop_mixed=drop_mixed, drop_cut=drop_cut, **kwargs)
        self._i = 0
        
    def _next_segment(self):
        if self._i >= len(self.timeline):
            raise StopIteration()
        end = self._i
        while end < len(self.timeline) and self.timeline[self._i] == self.timeline[end]:
            end += 1
        
        b = self.timeline.idx2time(self._i)
        e = self.timeline.idx2time(end)
        self._i = end
        return b, e, self.timeline[end-1]

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
        assert timeline is None or isinstance(timeline, _EvenlySignal),\
            "The parameter 'labels' should be an EvenlySignal."
        super(RandomFixedSegments, self).__init__(timeline=timeline, drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        assert N > 0
        assert width > 0
        
        self._N = N
        self._width = width
        self._i = -1
        self.reference = reference
        
        if reference is None:
            print('\n\n >>> No reference signal provided: new random segments will be generated for each channel/component. Expect funny results')
            self.tst = None
        else:
            t_st = self.reference.get_start_time()
            t_sp = self.reference.get_end_time() - self._width
            tst = _np.random.uniform(t_st, t_sp, self._N)

            #timestamps should be strictly (--> _np.unique) monotonic
            self.tst = _np.unique(tst[_np.argsort(tst)])
            
    def _next_segment(self):
        
        if self.tst is None: #needs initialization
            t_st = self.reference.get_start_time()
            t_sp = self.reference.get_end_time() - self._width
            tst = _np.random.uniform(t_st, t_sp, self._N)

            #timestamps should be strictly (--> _np.unique) monotonic
            self.tst = _np.unique(tst[_np.argsort(tst)])
        
        self._i += 1
        if self._i < self._N:
            b = self.tst[self._i]
            e = b + self._width
            return self.manage_drops(b, e)
        else:
            raise StopIteration()
            
def fmap(segmenter, algorithms, signal):
    """
    Generates a list of a list of results for each segment.

    [[result for each algorithm] for each segment]
    :param segments: An iterable of segments (e.g. an initialized SegmentGenerator)
    :param algorithms: A list of algorithms
    :param alt_signal: The signal that will be used instead of the one referenced in the segments

    :return: values, col_names A tuple: matrix (segment x algorithms) containing a value for each
     algorithm, the list of the algorithm names.
    """

    if segmenter.reference is None:
        segmenter(signal)
    
    result = {}
    
    #for all algorithms
    for alg in algorithms:
        
        result_algorithm = {}
        for i_seg, seg in enumerate(segmenter): #this generates segments from the segmenter
            result_segment = {'begin': seg.get_begin_time(),
                              'end': seg.get_end_time(),
                              'label': seg.get_label()}
            
            #when called on a signal, a segment returns a portion of the signal
            signal_segment = seg(signal)
            
            result_segment['result'] = alg(signal_segment)
            result_algorithm[i_seg] = result_segment
        
        #the following is to prepare labels and t
        #that might be used later
        #(to avoid messy code)
        labels = []
        values = []
        t = []
        for k,v in result_algorithm.items():
            labels.append(v['label'])
            values.append(v['result'])
            t_ = v['begin'] + (v['end'] - v['begin']) / 2
            t.append(t_)
        
        
        #if no segments were processed
        if len(labels) == 0: 
            result[alg.__repr__()] = result_algorithm
        
        #if the algorithm returns a Signal
        elif isinstance(values[0], _Signal):
            print(1)
            result[alg.__repr__()] = result_algorithm
        
        #if the algorithm returns a numpy array
        #we create Signals
        elif isinstance(values[0], _np.ndarray):
            print(2)
            print(values[0].shape)
            
            values = _np.concatenate(values, axis=0)
            #BE CAREFUL HERE ABOUT THE NUMBER OF DIMS OF THE OUTPUT ARRAY
            print(values.shape)
            
            if isinstance(segmenter, FixedSegments):
                print(3)
                #since we used a FixedSegments, we can create an EvenlySignal
                fsamp = 1/segmenter._step
                
                info = {'label': _EvenlySignal(labels, fsamp, t[0]),
                        'name': alg.__repr__()}
                
                info.update(signal.get_info())
                
                result[alg.__repr__()] = _EvenlySignal(values, fsamp, t[0], info)
                
            else:
                print(4)
                fsamp = signal.get_sampling_freq()
                info = {'label': _UnevenlySignal(labels, fsamp,
                                                 x_values = _np.array(t),
                                                 x_type='instants'),
                        'name': alg.__repr__()}
                
                info.update(signal.get_info())
                
                result[alg.__repr__()] = _UnevenlySignal(values, fsamp, info=info,
                                                         x_values = _np.array(t),
                                                         x_type='instants')
        
        #if list or tuple of ndarrays
        #(it is a special case we can try to manage)
        #we create a list of Signals
        elif (isinstance(values[0], list) or isinstance(values[0], tuple)) and \
            sum([isinstance(x, _np.ndarray) for x in values[0]]) ==  len(values[0]):
            print(5)
            number_signals = len(values[0])
            signals_out = []
            for i_signal in range(number_signals):
                values_signal = []
                for v in values:
                    values_signal.append(v[i_signal])
                values_signal = _np.concatenate(values_signal, axis=0)
                
                if isinstance(segmenter, FixedSegments):
                    print(6)
                    #since we used a FixedSegments, we can create an EvenlySignal
                    fsamp = 1/segmenter._step
                    
                    info = {'label': _EvenlySignal(labels, fsamp, t[0]),
                            'name': alg.__repr__()}
                    
                    info.update(signal.get_info())
                    
                    signals_out.append(_EvenlySignal(values_signal, 
                                                     fsamp, 
                                                     t[0], 
                                                     info))
                    
                else:
                    print(7)
                    fsamp = signal.get_sampling_freq()
                    info = {'label': _UnevenlySignal(_np.array(labels), 
                                                     fsamp, 
                                                     x_values = _np.array(t),
                                                     x_type='instants'),
                            'name': alg.__repr__()}
                    
                    info.update(signal.get_info())
                    
                    signals_out.append(_UnevenlySignal(values_signal, 
                                                       fsamp, 
                                                       info=info,
                                                       x_values = _np.array(t),
                                                       x_type='instants'))
        
            result[alg.__repr__()] = signals_out
        
        #all other cases
        #just return the original dictionary
        else:
            print(8)
            result[alg.__repr__()] = result_algorithm
            
    return result

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
                    indicator_df[key] = result_key[:, i_chan, i_comp].ravel()
                    indicator_df['component'] = _np.repeat(i_comp+1, len(t))
                    indicator_df['channel'] = _np.repeat(i_chan+1, len(t))
                    
                else:
                    if ind_sample.ndim == 2:
                        indicator_df[key] = result_key[:, i_chan].ravel()
                        indicator_df['channel'] = _np.repeat(i_chan+1, len(t))
                    else:
                        indicator_df[key] =  result_key
            
            df_all.append(_pd.DataFrame(indicator_df))
    
    
    df_all = _pd.concat(df_all, axis = 0)
    return(df_all)