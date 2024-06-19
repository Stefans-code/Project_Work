import numpy as _np
import scipy.optimize as _opt
from ... import create_signal
from ..._base_algorithm import _Algorithm
# from ...signal import create_signal
from ...filters import DeConvolutionalFilter as _DeConvolutionalFilter, \
    ConvolutionalFilter as _ConvolutionalFilter, IIRFilter as _IIRFilter
from ...utils import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection

from ._presets import *


def _loss(t1, t2, signal):
    if t2<=t1:
        return(1000000)
    
    driver = DriverEstim(t1=t1, t2=t2, optim=False)(signal)
    # driver = _ConvolutionalFilter('rect', 1)(driver, add_signal=False)
    
    driver_f = _IIRFilter([0.01, 0.05], btype='bandpass')(driver).p.get_values().ravel()
    
    # driver_diff = driver_v - driver_f_v
    driver_f[_np.where(driver_f>0)[0]] = _np.nan
    loss_out = abs(_np.nanmean(driver_f))
    
    return(loss_out)

def optimize_T1_T2(signal, bayesian, optim_bounds):
    minT1 = optim_bounds[0]
    maxT1 = optim_bounds[1]
    minT2 = optim_bounds[2]
    maxT2 = optim_bounds[3]
    
    if bayesian:
        from bayes_opt import BayesianOptimization
        
        def loss(t1, t2):
            loss_out = _loss(t1, t2, signal)
            return(-loss_out)
    
        optimizer = BayesianOptimization(loss, 
                                         pbounds = {'t1': (minT1, maxT1), 
                                                    't2': (minT2, maxT2)},
                                         verbose=False,
                                         allow_duplicate_points=True)
        optimizer.maximize(init_points=100, n_iter=100)
        
        t1 = optimizer.max['params']['t1']
        t2 = optimizer.max['params']['t2']
        res = {'x': [t1, t2]}
    else:
        def loss(pars):
            t1 = pars[0]
            t2 = pars[1]
            loss_out = _loss(t1, t2, signal)
            return(loss_out)

        res = _opt.differential_evolution(loss, bounds=[(minT1, maxT1), (minT2, maxT2)],
                                          maxiter=100,
                                          polish=True)
    return(res)

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

    def __init__(self, t1=.75, t2=2, optim=False, rescale_driver=True, optim_bayes=True, optim_bounds = (0.01, 1.99, 0.01, 20)):
        assert t1 > 0, "t1 value has to be positive"
        assert t2 > 0, "t2 value has to be positive"
        _Algorithm.__init__(self, t1=t1, t2=t2, optim=optim, 
                            rescale=rescale_driver,
                            optim_bayes=optim_bayes,
                            optim_bounds = optim_bounds)
        self.dimensions = {'time': 0}
        
    def algorithm(self, signal):
        optim = self._params['optim']
        if optim:
            optim_bayes = self._params['optim_bayes']
            optim_bounds = self._params['optim_bounds']
            pars = optimize_T1_T2(signal, 
                                  optim_bayes, 
                                  optim_bounds)
            
            self._params['t1'] = pars['x'][0]
            self._params['t2'] = pars['x'][1]
            
        fsamp = signal.p.get_sampling_freq()
        bateman = self._gen_bateman(fsamp)
        rescale = self._params['rescale']
        
            
        driver = _DeConvolutionalFilter(irf=bateman, normalize=False, deconv_method='fft')(signal)

        driver_values = driver.p.get_values()
        if rescale:
            driver_values = driver_values*_np.max(bateman)*fsamp
        
        return driver_values

    def _gen_bateman(self, fsamp):
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
        params = self._params
        t1 = params['t1']
        t2 = params['t2']
        
        idx_T1 = t1 * fsamp
        idx_T2 = t2 * fsamp
        len_bat = idx_T2 * 10
        idx_bat = _np.arange(len_bat)
        bateman = _np.exp(-idx_bat / idx_T2) - _np.exp(-idx_bat / idx_T1)

        # normalize
        bateman = bateman / (_np.sum(bateman) * len(bateman) / fsamp)
        
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

    def __init__(self, amplitude=0.01, win_pre=2, win_post=2, polyfit=True, return_phasic=True):
        assert amplitude > 0, "Amplitude value has to be positive"
        assert win_pre > 0,  "Window pre peak value has to be positive"
        assert win_post > 0, "Window post peak value has to be positive"
        _Algorithm.__init__(self, amplitude=amplitude, win_pre=win_pre, win_post=win_post, polyfit=polyfit, return_phasic=return_phasic)
        self.dimensions = {'time': 0}

    def algorithm(self, signal):
        params = self._params
        amplitude = params["amplitude"]
        # grid_size = params["grid_size"]
        win_pre = params['win_pre']
        win_post = params['win_post']
        polyfit = params['polyfit']
        return_phasic = params['return_phasic']

        fsamp = signal.p.get_sampling_freq()
        signal_values = signal.p.get_values().ravel()
        
        # find peaks in the driver
        maxima = _PeakDetection(delta=amplitude, refractory=1, return_peaks=True)(signal)
        idx_maxp = _np.where(~_np.isnan(maxima.p.main_signal.values))[0].ravel()
        # print(idx_maxp)
        
        # identify start and stop of the peaks
        peaks = _PeakSelection(indices=idx_maxp, win_pre=win_pre, win_post=win_post)(signal)
        
        # find tonic component (= portion outside the peaks ==> peaks == 0)
        idx_tonic = _np.where(peaks.p.get_values().ravel() == 0)[0]
        
        #first and last sample should be included
        if idx_tonic[0] != 0:
            idx_tonic = _np.insert(idx_tonic, 0, 0)
        
        if idx_tonic[-1] != (len(signal_values) - 1):
            idx_tonic = _np.insert(idx_tonic, len(idx_tonic), len(signal_values) - 1)
        
        
        tonic_interp = signal_values[idx_tonic]

        if polyfit:
            z = _np.polyfit(_np.arange(len(tonic_interp)), tonic_interp, 10)
            p = _np.poly1d(z)
            tonic_interp = p(_np.arange(len(tonic_interp)))
            
        tonic = create_signal(tonic_interp, times = idx_tonic/fsamp + signal.p.get_start_time())
        tonic = tonic.interp({'time': signal.p.get_times()}, 'cubic')
        tonic_values = tonic.p.get_values().ravel()
        
        # fitter = _np.poly1d(_np.polyfit(idx_tonic, driver_interp, 10))
        # tonic = fitter(_np.arange(len(signal_values)))

        if not return_phasic:
            return tonic_values
        
        phasic_values = signal_values - tonic.p.get_values().ravel()
       
        return phasic_values
    
        # # Linear interpolation to substitute the peaks
        # driver_no_peak = _np.copy(signal)
        # for I in range(len(idx_pre)):
        #     i_st = idx_pre[I]
        #     i_sp = idx_post[I]

        #     if not _np.isnan(i_st) and not _np.isnan(i_sp):
        #         idx_base = _np.arange(i_sp - i_st)
        #         coeff = (signal[i_sp] - signal[i_st]) / len(idx_base)
        #         driver_base = idx_base * coeff + signal[i_st]
        #         driver_no_peak[i_st:i_sp] = driver_base

        # # generate the grid for the interpolation
        # idx_grid = _np.arange(0, len(driver_no_peak) - 1, grid_size * fsamp)
        # idx_grid = _np.r_[idx_grid, len(driver_no_peak) - 1]

        # driver_grid = _Signal(driver_no_peak[idx_grid], sampling_freq = fsamp, 
        #                       start_time= signal.get_start_time(), info=signal.get_info(),
        #                       x_values=idx_grid, x_type='indices')
        # tonic = driver_grid.fill(kind='cubic')

        # phasic = signal - tonic
    




#%%    
