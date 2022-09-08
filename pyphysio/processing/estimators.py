# coding=utf-8
# from __future__ import division
import numpy as _np
from . import Algorithm as _Algorithm
from ..signal import create_signal
from .filters import IIRFilter as _IIRFilter, DeConvolutionalFilter as _DeConvolutionalFilter, \
    ConvolutionalFilter as _ConvolutionalFilter
from .tools import SignalRange as _SignalRange, PeakDetection as _PeakDetection, Minima as _Minima, \
    PeakSelection as _PeakSelection, Diff as _Diff
import itertools as _itertools
    
class Energy(_Algorithm):
    """
    Estimate the local energy of the signal, by windowing

    Parameters
    ----------
    win_len : float, >0
        Length of the window in seconds
    win_step : float, >0
        Shift of the window to start the next window
        
    Optional parameters
    -------------------
    
    smooth : boolean, default = True
        Whether to convolve the result with a gaussian window

    Returns
    -------
    energy : numpy.array
        Local energy
    """

    def __init__(self, win_len, win_step, smooth=True):
        assert win_len > 0, "Window length has to be positive"
        assert win_step > 0, "Window step has to be positive"
        _Algorithm.__init__(self, win_len=win_len, win_step=win_step, smooth=smooth)

    def algorithm(self, signal):
        params = self._params
        win_len = params['win_len']
        win_step = params['win_step']
        smooth = params['smooth']

        fsamp = signal.get_sampling_freq()
        idx_len = win_len * fsamp
        idx_step = win_step * fsamp

        windows = _np.arange(0, len(signal) - idx_len + 1, idx_step)

        energy = _np.empty(len(windows) + 2)
        for i in range(1, len(windows) + 1):
            start = windows[i - 1]
            portion_curr = signal.segment_idx(start, start + idx_len)
            energy[i] = _np.nanmean(_np.power(portion_curr, 2))
        energy[0] = energy[1]
        energy[-1] = energy[-2]

        idx_interp = _np.r_[0, windows + round(idx_len / 2), len(signal)-1]
        energy_out = _Signal(energy, sampling_freq=signal.get_sampling_freq(), 
                             start_time = signal.get_start_time(), 
                             x_values=idx_interp,
                             x_type='indices').fill('linear')

        if smooth:
            energy_out = _ConvolutionalFilter(irftype='gauss', win_len=2, normalize=True)(energy_out)

        return energy_out
