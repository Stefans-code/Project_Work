# coding=utf-8
# from __future__ import division
# 
# from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta

import numpy as _np
#import xarray as _xr
from .._base_algorithm import _Algorithm
from ..utils import PeakDetection as _PeakDetection,\
    PeakSelection as _PeakSelection

# __author__ = 'AleB'

#TODO: create just one function for duration and slopes


def _get_idx_peaks(signal, params):
    delta = params['delta']
    peaks = _PeakDetection(delta = delta)(signal)
    idx_peaks = _np.where(~_np.isnan(peaks.p.get_values()))[0].ravel()
    return(idx_peaks)

def _get_st_sp_peaks(signal, idx_peaks, params):
    win_pre = params['win_pre']
    win_post = params['win_post']
    
    peaks_area = _PeakSelection(idx_peaks, win_pre, win_post)(signal).p.get_values().ravel()
    idx_start_peaks = _np.where(_np.diff(peaks_area) ==1)[0]
    idx_stop_peaks = _np.where(_np.diff(peaks_area) == -1)[0]
    
    if (len(idx_start_peaks) == 0) | (len(idx_stop_peaks) == 0):
        out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
        return out
    
    if idx_start_peaks[0] > idx_stop_peaks[0]: #start with a peak
        idx_start_peaks = _np.insert(idx_start_peaks, 0, 0)
        
    if idx_start_peaks[-1] > idx_stop_peaks[-1]: #stop with a peak
        idx_stop_peaks = _np.insert(idx_stop_peaks, 
                                    len(idx_stop_peaks), 
                                    len(peaks_area))
    return(idx_start_peaks, idx_stop_peaks)

def _get_durations(signal, idx_peaks, params):
    fsamp = signal.p.get_sampling_freq()
    idx_start_peaks, idx_stop_peaks = _get_st_sp_peaks(signal, idx_peaks, params)
    
    durations = []
    for i_st, i_sp in zip(idx_start_peaks, idx_stop_peaks):
        durations.append((i_sp - i_st)/fsamp)
    return durations


def _get_slopes(signal, idx_peaks, params):
    fsamp = signal.p.get_sampling_freq()
    idx_start_peaks, idx_stop_peaks = _get_st_sp_peaks(signal, idx_peaks, params)
    
    signal_values = signal.p.get_values().ravel()
    slopes = []
    for i_st, i_sp in zip(idx_start_peaks, idx_stop_peaks):
        if (i_sp-i_st)>1:
            slopes.append(fsamp*_np.max(_np.diff(signal_values[i_st:i_sp])))
            
    return slopes

class PeaksMax(_Algorithm):
    """
    Return the maximum amplitude of detected peaks.

    Parameters
    ----------
    peaks : numpy array
        values of the result of PeakSelection
        
    
    Returns
    -------
    mx : float
        Maximum amplitude of detected peaks
    
    """
    def __init__(self, delta, **kwargs):
        _Algorithm.__init__(self, delta=delta, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        idx_peaks = _get_idx_peaks(signal, self._params)
        signal_values = signal.p.get_values()
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal_values, axis=0, keepdims=True)
            return out
            
        return _np.max(signal_values[idx_peaks], keepdims=True)


class PeaksMin(_Algorithm):
    """
    Return the minimum amplitude of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    mn : float
        Minimum amplitude of detected peaks
    
    """
    def __init__(self, delta, **kwargs):
        _Algorithm.__init__(self, delta=delta, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        idx_peaks = _get_idx_peaks(signal, self._params)
        signal_values = signal.p.get_values()
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal_values, axis=0, keepdims=True)
            return out
            
        return _np.min(signal_values[idx_peaks], keepdims=True)


class PeaksMean(_Algorithm):
    """
    Return the average amplitude of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    av : float
        Average amplitude of detected peaks
    
    """
    def __init__(self, delta, **kwargs):
        _Algorithm.__init__(self, delta=delta, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        idx_peaks = _get_idx_peaks(signal, self._params)
        signal_values = signal.p.get_values()
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal_values, axis=0, keepdims=True)
            return out
        
        out = _np.mean(signal_values[idx_peaks], axis=0, keepdims=True)
        return out


class PeaksNum(_Algorithm):
    """
    Return the number of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    
    Returns
    -------
    n : float
        Number of detected peaks
    
    """
    def __init__(self, delta, **kwargs):
        _Algorithm.__init__(self, delta=delta, **kwargs)
        self.dimensions = {'time' : 1}
    
    def algorithm(self, signal):
        idx_peaks = _get_idx_peaks(signal, self._params)
        return _np.array([len(idx_peaks)])


class DurationMin(_Algorithm):
    """
    Return the minimum duration of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mn : float
        Minimum duration of detected peaks
    
    """
    def __init__(self, delta, win_pre, win_post, **kwargs):
        assert delta > 0, 'delta must be > 0'
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Algorithm.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)
        self.dimensions = {'time' : 1}
        
    def algorithm(self, signal):
        params = self._params
        
        idx_peaks = _get_idx_peaks(signal, params)
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
            return out
        
        durations = _get_durations(signal, idx_peaks, params)
        
        return(_np.array([_np.min(durations)]))


class DurationMax(_Algorithm):
    """
    Return the maximum duration of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mx : float
        Maximum duration of detected peaks
    
    """
    def __init__(self, delta, win_pre, win_post, **kwargs):
        assert delta > 0, 'Parameter delta, i.e. amplitude of the minimum peak, has to be > 0'
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Algorithm.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        params = self._params
        
        idx_peaks = _get_idx_peaks(signal, params)
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
            return out
        
        durations = _get_durations(signal, idx_peaks, params)
        
        return(_np.array([_np.max(durations)]))


class DurationMean(_Algorithm):
    """
    Return the average duration of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    av : float
        Average duration of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        assert delta > 0, 'Parameter delta, i.e. amplitude of the minimum peak, has to be > 0'
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Algorithm.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        params = self._params
        
        idx_peaks = _get_idx_peaks(signal, params)
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
            return out
        
        idx_start_peaks, idx_stop_peaks = _get_st_sp_peaks(signal, idx_peaks, params)
        
        durations = _get_durations(signal, idx_peaks, params)
        
        return(_np.array([_np.mean(durations)]))


class SlopeMin(_Algorithm):
    """
    Return the minimum slope of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mn : float
        Minimum slope of detected peaks
    
    """
    def __init__(self, delta, win_pre, win_post, **kwargs):
        assert delta > 0, 'Parameter delta, i.e. amplitude of the minimum peak, has to be > 0'
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Algorithm.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        params = self._params
        
        idx_peaks = _get_idx_peaks(signal, params)
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
            return out
        
        slopes = _get_slopes(signal, idx_peaks, params)
        
        return(_np.array([_np.min(slopes)]))


class SlopeMax(_Algorithm):
    """
    Return the maximum slope of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mx : float
        Maximum slope of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        assert delta > 0, 'Parameter delta, i.e. amplitude of the minimum peak, has to be > 0'
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Algorithm.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        params = self._params
        
        idx_peaks = _get_idx_peaks(signal, params)
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
            return out
        
        slopes = _get_slopes(signal, idx_peaks, params)            
        
        return(_np.array([_np.max(slopes)]))


class SlopeMean(_Algorithm):
    """
    Return the average slope of detected peaks.

    Parameters
    ----------
    delta : float, >0
        Minimum amplitude of peaks to be selected
    win_pre : float, >0, default=1
        Interval before a detected peak where to search the start of the peak
    win_post : float, >0, default=1
        Interval after a detected peak where to search the end of the peak
        
    Returns
    -------
    mx : float
        Maximum slope of detected peaks
    
    """
    def __init__(self, delta, win_pre=1, win_post=1, **kwargs):
        assert delta > 0, 'Parameter delta, i.e. amplitude of the minimum peak, has to be > 0'
        assert win_pre > 0, 'win_pre must be > 0'
        assert win_post > 0, 'win_post must be > 0'
        _Algorithm.__init__(self, delta=delta, win_pre=win_pre, win_post=win_post, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        params = self._params
        
        idx_peaks = _get_idx_peaks(signal, params)
        
        if len(idx_peaks) == 0:
            out = _np.nan * _np.mean(signal.p.get_values(), axis=0, keepdims=True)
            return out
        
        slopes = _get_slopes(signal, idx_peaks, params)            
        
        return(_np.array([_np.mean(slopes)]))
