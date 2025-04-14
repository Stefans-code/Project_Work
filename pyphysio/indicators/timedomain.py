import numpy as _np
from . import _Indicator

class Mean(_Indicator):
    """
    Compute the arithmetic mean of the signal, ignoring any NaNs.
    """
    
    def algorithm(self, signal):
        signal_values = signal.p.get_values()
        result = _np.array(_np.mean(signal_values))
        return result

class Min(_Indicator):
    """
    Return minimum of the signal, ignoring any NaNs.
    """
    
    def algorithm(cls, signal):
        signal_values = signal.p.get_values()
        return _np.array(_np.min(signal_values))


class Max(_Indicator):
    """
    Return maximum of the signal, ignoring any NaNs.
    """
    def algorithm(self, signal):
        signal_values = signal.p.get_values()
        return _np.array(_np.max(signal_values))


class Range(_Indicator):
    """
    Compute the range of the signal, ignoring any NaNs.
    """

    def algorithm(self, signal):
        signal_values = signal.p.get_values()
        return _np.array(_np.max(signal_values) - _np.min(signal_values))


class Median(_Indicator):
    """
    Compute the median of the signal, ignoring any NaNs.
    """
    
    def algorithm(self, signal):
        signal_values = signal.p.get_values()
        return _np.array(_np.median(signal_values))


class StDev(_Indicator):
    """
    Computes the standard deviation of the signal, ignoring any NaNs.
    """
    
    def algorithm(self, signal):
        signal_values = signal.p.get_values()
        return _np.array(_np.std(signal_values))


class Sum(_Indicator):
    """
    Computes the sum of the values in the signal, treating Not a Numbers (NaNs) as zero.
    """
    
    def algorithm(self, signal):
        signal_values = signal.p.get_values()
        return _np.array(_np.sum(signal_values))


class AUC(_Indicator):
    """
    Computes the Area Under the Curve of the signal.
    """
    def algorithm(self, signal):
        fsamp = signal.p.get_sampling_freq()
        signal_values = signal.p.get_values()
        return _np.array(_np.sum(signal_values)*(1./fsamp))
