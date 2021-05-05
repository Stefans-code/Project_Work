# coding=utf-8
# from __future__ import division

import numpy as _np
from ..processing import Algorithm as _Algorithm

from ..processing.tools import Diff as _Diff
from ..signal import EvenlySignal as _EvenlySignal, Signal as _Signal


# __author__ = 'AleB'


class Mean(_Algorithm):
    """
    Compute the arithmetic mean of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        return _np.nanmean(signal.get_values())


class Min(_Algorithm):
    """
    Return minimum of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmin(data.get_values())


class Max(_Algorithm):
    """
    Return maximum of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        return _np.nanmax(signal.get_values())


class Range(_Algorithm):
    """
    Compute the range of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        return Max()(signal) - Min()(signal)


class Median(_Algorithm):
    """
    Compute the median of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        return _np.median(signal.get_values())


class StDev(_Algorithm):
    """
    Computes the standard deviation of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        return _np.nanstd(signal.get_values())


class Sum(_Algorithm):
    """
    Computes the sum of the values in the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        return _np.nansum(signal.get_values())


class AUC(_Algorithm):
    """
    Computes the Area Under the Curve of the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
            print('Calculating Area Under the Curve of an Unevenly signal!')
        fsamp = signal.get_sampling_freq()
        return (1. / fsamp) * Sum()(signal)
    
class DetrendedAUC(_Algorithm):
    """
    Computes the Area Under the Curve of the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
            print('Calculating Area Under the Curve of an Unevenly signal!')
        fsamp = signal.get_sampling_freq()
        
        #detrend
        t_signal = signal.get_times()
        intercept = signal[0]
        coeff = (signal[-1] - signal[0]) / signal.get_duration()
        baseline = coeff*(t_signal - t_signal[0]) + intercept
        
        signal_ = signal - baseline
        return (1. / fsamp) * Sum()(signal_)


class RMSSD(_Algorithm):
    """
    Compute the square root of the mean of the squared 1st order discrete differences.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        diff = _Diff()(signal)
        return _np.sqrt(_np.nanmean(_np.power(diff.get_values(), 2)))


class SDSD(_Algorithm):
    """
    Calculate the standard deviation of the 1st order discrete differences.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        diff = _Diff()(signal)
        return StDev()(diff)

# TODO: FIX Histogram missing
class Triang(_Algorithm):
    """
    Computes the HRV triangular index.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        step = 1000. / 128
        min_ibi = _np.min(signal)
        max_ibi = _np.max(signal)
        if (max_ibi - min_ibi) / step + 1 < 10:
            print("len(bins) < 10")
            return _np.nan
        else:
            bins = _np.arange(min_ibi, max_ibi, step)
            h, b = _np.histogram(signal, bins)
            return len(signal) / _np.max(h)


class TINN(_Algorithm):
    """
    Computes the triangular interpolation of NN interval histogram.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)

    def algorithm(self, signal):
        step = 1000. / 128
        min_ibi = _np.min(signal)
        max_ibi = _np.max(signal)
        if (max_ibi - min_ibi) / step + 1 < 10:
            print("len(bins) < 10")
            return _np.nan
        else:
            bins = _np.arange(min_ibi, max_ibi, step)
            h, b = _np.histogram(signal, bins)
            max_h = _np.max(h)
            hist_left = _np.array(h[0:_np.argmax(h)])
            ll = len(hist_left)
            hist_right = _np.array(h[_np.argmax(h):])
            rl = len(hist_right)
            y_left = _np.array(_np.linspace(0, max_h, ll))

            minx = _np.Inf
            pos = 0
            for i in range(1, len(hist_left) - 1):
                curr_min = _np.sum((hist_left - y_left) ** 2)
                if curr_min < minx:
                    minx = curr_min
                    pos = i
                y_left[i] = 0
                y_left[i + 1:] = _np.linspace(0, max_h, ll - i - 1)

            n = b[pos - 1]

            y_right = _np.array(_np.linspace(max_h, 0, rl))
            minx = _np.Inf
            pos = 0
            for i in range(rl - 1, 1, -1):
                curr_min = _np.sum((hist_right - y_right) ** 2)
                if curr_min < minx:
                    minx = curr_min
                    pos = i
                y_right[i - 1] = 0
                y_right[0:i - 2] = _np.linspace(max_h, 0, i - 2)

            m = b[_np.argmax(h) + pos + 1]
            return m - n
