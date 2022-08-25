# coding=utf-8
# from __future__ import division
import numpy as _np
from . import Algorithm as _Algorithm
from ..signal import create_signal
from .filters import IIRFilter as _IIRFilter, DeConvolutionalFilter as _DeConvolutionalFilter, \
    ConvolutionalFilter as _ConvolutionalFilter
from .tools import SignalRange as _SignalRange, PeakDetection as _PeakDetection, Minima as _Minima, \
    PeakSelection as _PeakSelection, Diff as _Diff

# __author__ = 'AleB'


# IBI ESTIMATION
class BeatFromBP(_Algorithm):
    """
    Identify the beats in a Blood Pulse (BP) signal and compute the IBIs.
    Optimized to identify the percussion peak.

    Optional parameters
    -------------------
    
    bpm_max : int, (1, 400], default=120
        Maximal expected heart rate (in beats per minute)
    win_pre : float, (0, 1], default=0.25
        Portion (in seconds) to consider before the candidate beat position where to look for the beat
    win_post : float, (0, 1], default=0.05
        Portion (in seconds) to consider after the candidate beat position where to look for the beat


    Returns
    -------
    ibi : UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
    Please cite:
        Bizzego, Andrea, and Cesare Furlanello. "DBD-RCO: Derivative Based Detection And Reverse Combinatorial Optimization To Improve Heart Beat Detection For Wearable Devices." bioRxiv (2017): 118943.
    """
    
    def __init__(self, bpm_max=120, win_pre=.25, win_post=.05):
        if not 10 < bpm_max < 400:
            self.warn("Parameter bpm_max out of reasonable range (10, 400)")
        assert 0 < win_pre <= 1, "Window pre peak value should be in (0 and 1]"
        assert 0 < win_post <= 1, "Window post peak value should be in (0 and 1]"
        _Algorithm.__init__(self, bpm_max=bpm_max, win_pre=win_pre, win_post=win_post)
        self.dimensions = {'time':0}

    def algorithm(self, signal):
        
        params = self._params
        fsamp = signal.p.get_sampling_freq()
        bpm_max = params["bpm_max"]
        #TODO account for bpm_max when assigning win_pre / win_post
        win_pre = params["win_pre"] * fsamp
        win_post = params["win_post"] * fsamp
        
        fmax = bpm_max / 60
        refractory = 1 / fmax

        times = signal.p.get_times()

        # STAGE 1 - EXTRACT BEAT POSITION SIGNAL
        # filtering
        signal_f = _IIRFilter(fp=1.2 * fmax, fs=3 * fmax, ftype='ellip')(signal)
        # find range for the adaptive peak detection
        delta = 0.5 * _SignalRange(win_len=1.5 / fmax, win_step=1 / fmax)(signal_f)
        
        delta = delta.values.ravel()

        #adjust for delta values equal to 0
        idx_delta_zeros = _np.where(delta==0)[0]
        idx_delta_nozeros = _np.where(delta>0)[0]
        delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])
        
        # detection of candidate peaks
        maxima = _PeakDetection(delta=delta, refractory=refractory, start_max=True, return_peaks=True)(signal_f)
        
        maxp = _np.where(~_np.isnan(maxima.values))[0].ravel()
        
        if maxp[0] == 0:
            maxp = maxp[1:]

        # STAGE 2 - IDENTIFY PEAKS using the signal derivative
        # compute the signal derivative
        dxdt = _Diff()(signal).values

        true_peaks = []
        # for each candidate peak find the correct peak
        for idx_beat in maxp:
            start_ = int(idx_beat - win_pre)
            if start_ < 0:
                start_ = 0

            stop_ = int(idx_beat + win_post)
            if stop_ > len(dxdt):
                stop_ = -1

            # select portion of derivative where to search
            obs = dxdt[start_:stop_]
            peak_obs = _np.argmax(obs)
            i_end = 1
            while peak_obs == (len(obs) - i_end):
                peak_obs = _np.argmax(obs[:-i_end])
                i_end +=1
                
            true_obs = dxdt[start_ + peak_obs: stop_]
            
            true_obs = create_signal(abs(true_obs), times = times[start_ + peak_obs: stop_])
            
            # find the 'first minimum' (zero) the derivative (peak)
            minima = _Minima(win_len=0.1, win_step=0.025, method='windowing')(true_obs)
                        
            idx_mins = _np.where(~_np.isnan(minima.p.main_signal.values))[0].ravel()

            if len(idx_mins) >= 1:
                peak = idx_mins[0]
                true_peaks.append(start_ + peak_obs + peak + 1)
            else:
                print('Peak not found; idx_beat: ' + str(idx_beat))
                pass

        # STAGE 3 - FINALIZE computing IBI
        t_ibi = true_peaks / fsamp
        v_ibi = _np.diff(t_ibi)
        v_ibi = _np.insert(v_ibi, 0, v_ibi[0])

        ibi_scaffold = _np.nan* _np.zeros(len(signal.values))
        ibi_scaffold[true_peaks] = v_ibi
        
        return ibi_scaffold


class BeatFromECG(_Algorithm):
    """
    Identify the beats in an ECG signal and compute the IBIs.

    Optional parameters
    -------------------
    
    bpm_max : int, (1, 400], default=120
        Maximal expected heart rate (in beats per minute)
    delta : float, >=0, default=0
        Threshold for the peak detection. By default it is computed from the signal (adaptive thresholding)
    k : float, (0,1), default=0.7
        Ratio at which the signal range is multiplied (when delta = 0)

    Returns
    -------
    ibi : UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
        This algorithms looks for maxima in the signal which are followed by values lower than a delta value. 
        The adaptive version estimates the delta value adaptively.
    """

    def __init__(self, bpm_max=120, delta=0, k=0.7):
        if not 10 < bpm_max < 400:
            self.warn("Parameter bpm_max out of reasonable range (10, 400)")
        assert delta >= 0, "Delta value should be positive (or equal to 0 if automatically computed)"
        assert 0 < k < 1, "K coefficient must be in the range (0,1)"
        _Algorithm.__init__(self, bpm_max=bpm_max, delta=delta, k=k)
        self.dimensions = {'time':0}

    def algorithm(self, signal):
        params = self._params
        bpm_max, delta, k = params["bpm_max"], params["delta"], params["k"]
        fmax = bpm_max / 60
        
        fsamp = signal.p.get_sampling_freq()
        
        if delta == 0:
            delta = k * _SignalRange(win_len=2 / fmax, win_step=0.5 / fmax, smooth=False)(signal)
            delta = _np.array(delta).ravel()
        
        #adjust for delta values equal to 0
        idx_delta_zeros = _np.where(delta==0)[0]
        idx_delta_nozeros = _np.where(delta>0)[0]
        delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])
        
        refractory = 1 / fmax
        
        #find beats
        maxp = _PeakDetection(delta=delta, refractory=refractory, start_max=True)(signal)
        maxp = _np.array(maxp).ravel()
        
        if maxp[0] == 0:
            maxp = maxp[1:]

        idx_beats = _np.where(~_np.isnan(maxp))[0]
        
        times_beats = idx_beats / fsamp
        
        ibi_values = _np.diff(times_beats)

        ibi_values = _np.insert(ibi_values, 0, ibi_values[0])
        
        ibi_scaffold = _np.nan* _np.zeros(len(signal.values))
        
        ibi_scaffold[idx_beats] = ibi_values

        return ibi_scaffold


# PHASIC ESTIMATION
class DriverEstim(_Algorithm):
    """
    Estimates the driver of an EDA signal according to (see Notes)

    The estimation uses a deconvolution using a Bateman function as Impulsive Response Function.
    The version of the Bateman function here adopted is:

    :math:`b = e^{-t/T1} - e^{-t/T2}`

    Optional parameters
    -------------------
    t1 : float, >0, default = 0.75
        Value of the T1 parameter of the bateman function
    t2 : float, >0, default = 2
        Value of the T2 parameter of the bateman function

    Returns
    -------
    driver : EvenlySignal
        The EDA driver function

    Notes
    -----
    Please cite:
        
    """
    #TODO: add citation

    def __init__(self, t1=.75, t2=2):
        assert t1 > 0, "t1 value has to be positive"
        assert t2 > 0, "t2 value has to be positive"
        _Algorithm.__init__(self, t1=t1, t2=t2)

    def algorithm(self, signal):
        params = self._params
        t1 = params['t1']
        t2 = params['t2']

        fsamp = signal.get_sampling_freq()
        bateman = DriverEstim._gen_bateman(fsamp, [t1, t2])
        idx_max_bat = _np.argmax(bateman)

        # Prepare the input signal to avoid starting/ending peaks in the driver
        bateman_first_half = bateman[0:idx_max_bat + 1]
        bateman_first_half = signal[0] * (bateman_first_half - _np.min(bateman_first_half)) / (
            _np.max(bateman_first_half) - _np.min(bateman_first_half))

        bateman_second_half = bateman[idx_max_bat:]
        bateman_second_half = signal[-1] * (bateman_second_half - _np.min(bateman_second_half)) / (
            _np.max(bateman_second_half) - _np.min(bateman_second_half))

        signal_in = _np.r_[bateman_first_half, signal.get_values(), bateman_second_half]
        signal_in = _Signal(signal_in, sampling_freq=fsamp)

        # deconvolution
        driver = _DeConvolutionalFilter(irf=bateman, normalize=True, deconv_method='fft')(signal_in)
        driver = driver[idx_max_bat + 1: idx_max_bat + len(signal)]

        # gaussian smoothing
        driver = _ConvolutionalFilter(irftype='gauss', win_len=_np.max([0.2, 1 / fsamp]) * 8, normalize=True)(driver)

        driver = _Signal(driver, sampling_freq=fsamp, start_time=signal.get_start_time(), info=signal.get_info())
        return driver

    @staticmethod
    def _gen_bateman(fsamp, par_bat):
        """
        Generates the bateman function:

        :math:`b = e^{-t/T1} - e^{-t/T2}`

        Parameters
        ----------
        fsamp : float
            Sampling frequency
        par_bat: list (T1, T2)
            Parameters of the bateman function

        Returns
        -------
        bateman : array
            The bateman function
        """

        idx_T1 = par_bat[0] * fsamp
        idx_T2 = par_bat[1] * fsamp
        len_bat = idx_T2 * 10
        idx_bat = _np.arange(len_bat)
        bateman = _np.exp(-idx_bat / idx_T2) - _np.exp(-idx_bat / idx_T1)

        # normalize
        bateman = fsamp * bateman / _np.sum(bateman)
        return bateman


class PhasicEstim(_Algorithm):
    """
    Estimates the phasic and tonic components of a EDA driver function.
    It uses a detection algorithm based on the derivative of the driver.

    
    Parameters:
    -----------
    delta : float, >0
        Minimum amplitude of the peaks in the driver
        
    Optional parameters
    -------------------
    grid_size : float, >0, default = 1
        Sampling size of the interpolation grid
    pre_max : float, >0, default = 2
        Duration (in seconds) of interval before the peak where to search the start of the peak
    post_max : float, >0, default = 2
        Duration (in seconds) of interval after the peak where to search the end of the peak

    Returns:
    --------
    phasic : EvenlySignal
        The phasic component
    tonic : EvenlySignal
        The tonic component
    driver_no_peak : EvenlySignal
        The "de-peaked" driver signal used to generate the interpolation grid
    
    Notes
    -----
    Please cite:
        
    """
    #TODO: add citation

    def __init__(self, delta, grid_size=1, win_pre=2, win_post=2):
        assert delta > 0, "Delta value has to be positive"
        assert grid_size > 0, "Step of the interpolation grid has to be positive"
        assert win_pre > 0,  "Window pre peak value has to be positive"
        assert win_post > 0, "Window post peak value has to be positive"
        _Algorithm.__init__(self, delta=delta, grid_size=grid_size, win_pre=win_pre, win_post=win_post)

    def algorithm(self, signal):
        params = self._params
        delta = params["delta"]
        grid_size = params["grid_size"]
        win_pre = params['win_pre']
        win_post = params['win_post']

        fsamp = signal.get_sampling_freq()

        # find peaks in the driver
        idx_max, idx_min, val_max, val_min = _PeakDetection(delta=delta, refractory=1, start_max=True)(signal)

        # identify start and stop of the peak
        idx_pre, idx_post = _PeakSelection(indices=idx_max, win_pre=win_pre, win_post=win_post)(signal)

        # Linear interpolation to substitute the peaks
        driver_no_peak = _np.copy(signal)
        for I in range(len(idx_pre)):
            i_st = idx_pre[I]
            i_sp = idx_post[I]

            if not _np.isnan(i_st) and not _np.isnan(i_sp):
                idx_base = _np.arange(i_sp - i_st)
                coeff = (signal[i_sp] - signal[i_st]) / len(idx_base)
                driver_base = idx_base * coeff + signal[i_st]
                driver_no_peak[i_st:i_sp] = driver_base

        # generate the grid for the interpolation
        idx_grid = _np.arange(0, len(driver_no_peak) - 1, grid_size * fsamp)
        idx_grid = _np.r_[idx_grid, len(driver_no_peak) - 1]

        driver_grid = _Signal(driver_no_peak[idx_grid], sampling_freq = fsamp, 
                              start_time= signal.get_start_time(), info=signal.get_info(),
                              x_values=idx_grid, x_type='indices')
        tonic = driver_grid.fill(kind='cubic')

        phasic = signal - tonic

        return phasic, tonic, driver_no_peak


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
