import numpy as _np
from ..._base_algorithm import _Algorithm
# from ...signal import create_signal
from ...filters import DeConvolutionalFilter as _DeConvolutionalFilter, \
    ConvolutionalFilter as _ConvolutionalFilter
from ...utils import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection

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

#%%    
