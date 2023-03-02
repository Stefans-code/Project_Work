import numpy as _np
from .._base_algorithm import _Algorithm

from ..utils import Diff as _Diff

class Mean(_Algorithm):
    """
    Compute the arithmetic mean of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}
        
    def algorithm(self, signal):
        result = _np.mean(signal.values, keepdims=True)
        # print(result.shape)
        return result

class Min(_Algorithm):
    """
    Return minimum of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    @classmethod
    def algorithm(cls, signal):
        return _np.min(signal.values, keepdims=True)


class Max(_Algorithm):
    """
    Return maximum of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        return _np.max(signal.values, keepdims=True)


class Range(_Algorithm):
    """
    Compute the range of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        return Max()(signal).values - Min()(signal).values


class Median(_Algorithm):
    """
    Compute the median of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        return _np.median(signal.values, keepdims=True)


class StDev(_Algorithm):
    """
    Computes the standard deviation of the signal, ignoring any NaNs.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        return _np.std(signal.values, keepdims=True)


class Sum(_Algorithm):
    """
    Computes the sum of the values in the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        return _np.sum(signal.values, keepdims=True)


class AUC(_Algorithm):
    """
    Computes the Area Under the Curve of the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        fsamp = signal.p.get_sampling_freq()
        return Sum()(signal).values*(1./fsamp)
    
class DetrendedAUC(_Algorithm):
    """
    Computes the Area Under the Curve of the signal, treating Not a Numbers (NaNs) as zero.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        fsamp = signal.p.get_sampling_freq()
        
        signal_values = signal.values
        
        #detrend
        idx = _np.arange(len(signal_values))[:, _np.newaxis, _np.newaxis]
        intercept = signal_values[[0]]
        coeff = (signal_values[[-1]] - signal_values[[0]]) / len(signal_values)
        
        baseline = intercept + coeff*idx
        
        signal_ = signal_values - baseline
        
        auc = (1. / fsamp) * _np.sum(signal_, keepdims=True)
        return auc


class RMSSD(_Algorithm):
    """
    Compute the square root of the mean of the squared 1st order discrete differences.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        
        signal_values = signal.p.get_values()
        diff = _np.diff(signal_values, axis=0)
        return _np.sqrt(_np.mean(_np.power(diff, 2), keepdims=True))


class SDSD(_Algorithm):
    """
    Calculate the standard deviation of the 1st order discrete differences.
    """
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        diff = _Diff()(signal)
        return StDev()(diff)

'''
# I never really used these...

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
'''