# coding=utf-8
# from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta
from copy import copy as _cpy
import numpy as _np
# from .Utility import PhUI as _PhUI
from .BaseAlgorithm import Algorithm as _Algorithm
from .Signal import Signal as _Signal

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
        self._signal = signal

    def get_begin_time(self):
        return self._begin

    def get_end_time(self):
        return self._end

    def get_begin(self):
        return self._signal.get_idx(self.get_begin_time())

    def get_end(self):
        return self._signal.get_idx(self.get_end_time()) if self.get_end_time() is not None else None

    def get_duration(self):
        return (self.get_end_time() - self.get_begin_time()) if self.get_end_time() is not None else None

    def get_label(self):
        return self._label

    def is_empty(self):
        return self._signal is None or self.get_begin_time() >= len(self._signal.get_end_time())

    def __call__(self, data=None):
        if data is None:
            data = self._signal
        return data.segment_time(self.get_begin_time(), self.get_end_time())

    def __repr__(self):
        return '[%s:%s' % (str(self.get_begin_time()), str(self.get_end_time())) + (
            ":%s]" % self._label if self._label is not None else "]")

class SegmentationIterator(object):
    """
    A generic iterator that is called from each WindowGenerator from the __iter__ method.
    """

    def __init__(self, win):
        assert isinstance(win, SegmentsWithLabelSignal)
        self._win = _cpy(win)
        self._win.init_segmentation()

    def __next__(self):
        return self._win.next_segment()

    # Python 2 & users compatibility
    def next(self):
        return self.__next__()
    

class SegmentsWithLabelSignal(_Algorithm):
    # Assumed: label signal extended over the end by holding the value

    def __init__(self, drop_cut=True, drop_mixed=True, **kwargs):
        _Algorithm.__init__(self, drop_cut=drop_cut, drop_mixed=drop_mixed, **kwargs)
        self._labsig = None

        pass

    def next_segment(self):
        assert self._labsig is not None, "Can't preview the segments without a signal here."
        #Use the syntax " + FixedSegments.__name__ + "(**params)(signal)")
        #raise StopIteration()

        b, e, label = self.next_segment_mix_labels()
        s = Segment(b, e, label, self._labsig)
        return s

    # @_abstract
    def next_times(self):
        pass

    # Algorithm Override, no cache
    def __call__(self, data=None):
        assert data is not None or self._labsig is not None, "No signal specified for " + self.__class__.__name__
        assert isinstance(data.get_values(), _np.ndarray), "The data must be a Signal (see class EvenlySignal and UnevenlySignal)."
        
        # params = self._params
        # o = self(**params)
        # o._signal = data
        self._labsig = data
        return self
    
        # # return self.run(data if data is not None else self._signal, self._params, use_cache=False)
        # values_out = _np.apply_along_axis(self.algorithm, 0, data)
        # return(values_out)

    def __iter__(self):
        return SegmentationIterator(self)

    # # @classmethod
    # def algorithm(self, data):
    #     params = self._params
    #     o = self(**params)
    #     o._signal = data
    #     return o

    def check_drop_and_range(self, s, b, e):
        drop = True, None, None

        # signal segment bounds

        # full under-range (empty) or full over-range (empty)
        if e < s.get_start_time() or b >= s.get_end_time():
            return drop

        # part before start: mixed and shorter (as partially before the first label's begin)
        if b < s.get_start_time():
            if self._params['drop_mixed'] or self._params['drop_cut']:
                # goto next segment (drop)
                return drop
            else:
                # cut to start
                b = s.get_start_time()

        # part after end: shorter but not mixed (as half after the last label's end)
        if e > s.get_end_time():
            if self._params['drop_cut']:
                # goto next segment (drop)
                return drop
            else:
                # cut to start
                e = s.get_end_time()

        # Don't drop, inclusive begin time, exclusive end time
        return False, b, e

    def next_segment_mix_labels(self):
        label = b = e = None
        while True:
            # break    ==> keep
            # continue ==> drop

            b, e = self.next_times()

            drop, b, e = self.check_drop_and_range(self._labsig, b, e)

            if drop:
                continue

            if not isinstance(self._labsig, _Signal):
                label = None
            else:

                drop, _, _ = self.check_drop_and_range(self._labsig, b, e)

                if drop:
                    # partially or completely out of labsig range and have to drop it
                    label = None
                    continue

                # labels segment bounds, may be < 0 (None < 0)
                first = self._labsig.get_iidx(b)
                last = self._labsig.get_iidx(e)
                
                if first == last:
                    last += 1

                lab_seg = self._labsig.segment_iidx(first, last)
                lab_first = lab_seg[0]

                if len(lab_seg) == 1 or (lab_seg[:1:-1] == lab_first).all(): ###[AB]
                    label = lab_first
                else:
                    if self._params['drop_mixed']:
                        continue
                    else:
                        label = None
            break

        return b, e, label

    def __repr__(self):
        if self._labsig is not None:
            return _Algorithm.__repr__(self) + " over\n" + str(self._labsig)
        else:
            return _Algorithm.__repr__(self)
