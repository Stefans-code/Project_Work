# coding=utf-8
import numpy as _np
from numpy import asarray as _asarray
# from ..Utility import abstractmethod as _abstract
from ..BaseSegmentation import SegmentsWithLabelSignal as _SegmentsWithLabelSignal, Segment
from ..Signal import Signal as _Signal

# __author__ = 'AleB'


class RandomFixedSegments(_SegmentsWithLabelSignal):
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

    def __init__(self, N, width, labels, drop_mixed=True, drop_cut=True, **kwargs):
        super(RandomFixedSegments, self).__init__(N=N, width=width, labels=labels, 
                                                  drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        assert N > 0
        assert width > 0
        assert isinstance(labels, _Signal), "The parameter 'labels' should be a Signal."
        self._N = N
        self._width = width
        self._i = -1

        tst = labels.get_start_time()
        tsp = labels.get_end_time() - width
        self._tst_randoms = _np.random.uniform(tst, tsp, self._N)
        self._labsig = labels
        
    def next_times(self):
        self._i +=1
        
        if self._i < len(self._tst_randoms):
            b = self._tst_randoms[self._i]
            e = b + self._width
            return b, e
        else:
            raise StopIteration()
            
class FixedSegments(_SegmentsWithLabelSignal):
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

    def __init__(self, step, width=None, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(FixedSegments, self).__init__(step=step, width=width, labels=labels, drop_mixed=drop_mixed,
                                            drop_cut=drop_cut, **kwargs)
        assert step > 0
        assert width is None or width > 0
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be a Signal."
        self._step = step
        self._width = width if width is not None else step
        self._t = None
        self._labsig = labels
        
    def next_times(self):
        assert self._labsig is not None
        if self._t is None:
            self._t = self._labsig.get_start_time()
        b = self._t
        self._t += self._step
        e = b + self._width
        if b >= self._labsig.get_end_time():
            raise StopIteration()
        return b, e

class CustomSegments(_SegmentsWithLabelSignal):
    #TODO: labels should be a list with labels of each segment, not a signal
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

    def __init__(self, begins, ends, labels=None, drop_mixed=True, drop_cut=True, **kwargs):
        super(CustomSegments, self).__init__(begins=begins, ends=ends, labels=labels, drop_cut=drop_cut,
                                             drop_mixed=drop_mixed, **kwargs)
        assert len(begins) == len(ends), "The number of begins has to be equal to the number of ends :)"
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be an Signal."
        self._i = -1
        self._b = begins
        self._e = ends
        self._labsig = labels

    
    def next_times(self):
        self._i += 1
        if self._i < len(self._b):
            return self._b[self._i], self._e[self._i]
        else:
            raise StopIteration()

class LabelSegments(_SegmentsWithLabelSignal):
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

    def __init__(self, labels, drop_mixed=True, drop_cut=True, **kwargs):
        super(LabelSegments, self).__init__(labels=labels, drop_mixed=drop_mixed, drop_cut=drop_cut, **kwargs)
        assert labels is None or isinstance(labels, _Signal),\
            "The parameter 'labels' should be a Signal."
        self._i = 0
        self._labsig = labels

    def next_times(self):
        if self._i >= len(self._labsig):
            raise StopIteration()
        end = self._i
        while end < len(self._labsig) and self._labsig[self._i] == self._labsig[end]:
            end += 1
        
        b = self._labsig.get_time_from_iidx(self._i)
        e = self._labsig.get_time_from_iidx(end)
        self._i = end
        return b, e

def fmap(segments, algorithms, alt_signal=None):
    # TODO : rename extract_indicators
    """
    Generates a list composed of a list of results for each segment.

    [[result for each algorithm] for each segment]
    :param segments: An iterable of segments (e.g. an initialized SegmentGenerator)
    :param algorithms: A list of algorithms
    :param alt_signal: The signal that will be used instead of the one referenced in the segments

    :return: values, col_names A tuple: matrix (segment x algorithms) containing a value for each
     algorithm, the list of the algorithm names.
    """

    
    seg_for = segments(alt_signal) if isinstance(segments, _SegmentsWithLabelSignal) else segments
    
    values = []
    for seg in seg_for:
        segment_data = _np.array([seg.get_begin_time(), seg.get_end_time(), seg.get_label()]).reshape(3,1)
        vals_segment = []
        for alg in algorithms:
            vals_alg = _np.array(alg(seg(alt_signal)))

            if not alt_signal.is_multi():
#                vals_alg = _np.expand_dims([vals_alg], 1)
                vals_alg = _np.array([vals_alg])
            vals_segment.append(vals_alg)
            
        vals_segment = _np.array(vals_segment)
        print(vals_segment.shape)
        print(vals_segment)
        seg_data_array = _np.repeat(segment_data, alt_signal.get_nchannels(), axis = 1)
        print(seg_data_array.shape)
        print(seg_data_array)
        vals_segment = _np.concatenate([seg_data_array, vals_segment], axis = 0)
        values.append(vals_segment)
    
    values = _np.array(values)
    
    #for compatibility
    if not alt_signal.is_multi():
        values = values[:,:,0]
        
    col_names = ["begin", "end", "label"] + [x.__repr__() for x in algorithms]
    
    return values, _asarray(col_names)