import numpy as _np
import scipy.optimize as _opt
from ... import create_signal
from ..._base_algorithm import _Algorithm
# from ...signal import create_signal
from ...filters import DeConvolutionalFilter as _DeConvolutionalFilter, \
    ConvolutionalFilter as _ConvolutionalFilter, IIRFilter as _IIRFilter, \
    KalmanFilter as _KalmanFilter, _Filter
from ...utils import PeakDetection as _PeakDetection, PeakSelection as _PeakSelection

# from ._presets import *


def _loss(t1, t2, signal, amplitude):
    if t2<=t1:
        return(_np.sum(abs(signal.p.get_values().ravel())))
    
    driver = DriverEstim(t1=t1, t2=t2, optim=False)(signal)
    # signal_f = _IIRFilter(0.05, btype='lowpass')(driver).p.get_values()
    
    # driver_diff = driver.p.get_values() - signal_f
    
    phasic_values = PhasicEstimKalman(amplitude=amplitude)(driver)

    phasic_values[_np.where(phasic_values>0)[0]] = _np.nan
    loss_out = abs(_np.nanmean(phasic_values))
    
    return(loss_out)

def optimize_T1_T2(signal, bayesian, optim_bounds, amplitude):
    minT1 = optim_bounds[0]
    maxT1 = optim_bounds[1]
    minT2 = optim_bounds[2]
    maxT2 = optim_bounds[3]
    
    if bayesian:
        from bayes_opt import BayesianOptimization
        _np.random.seed(1234)
        def loss(t1, t2):
            loss_out = _loss(t1, t2, signal, amplitude)
            return(-loss_out)
        print(loss(0.75, 2))
        optimizer = BayesianOptimization(loss, 
                                         pbounds = {'t1': (minT1, maxT1), 
                                                    't2': (minT2, maxT2)},
                                         verbose=False,
                                         allow_duplicate_points=True)
        optimizer.maximize(init_points=100, n_iter=100)
        
        t1 = optimizer.max['params']['t1']
        t2 = optimizer.max['params']['t2']
        print(optimizer.max['target'])
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
class DriverEstim(_Filter):
    """
    Estimates the driver of an EDA signal according to (see Notes)

    The estimation uses a deconvolution using a Bateman function as Impulsive Response Function.
    The version of the Bateman function here adopted is:

    :math:`b = e^{-t/T1} - e^{-t/T2}`

    Optional parameters
    -------------------
    t1 : float, >0, default = 0.96 
        Value of the T1 parameter of the bateman function. 
        The default value is the average value found in Benedek and Kaernback, 2010)
    t2 : float, >0, default = 3.76
        Value of the T2 parameter of the bateman function
        The default value is the average value found in Benedek and Kaernback, 2010)

    Returns
    -------
    driver : EvenlySignal
        The EDA driver function

    Notes
    -----
    Please cite:
        
    """
    #TODO: add citation

    def __init__(self, t1=.96, t2=3.76, rescale_driver=True,
                 optim=False, optim_bayes=True, optim_bounds = (0.05, 3, 0.3, 15),
                 amplitude=0.01):
        assert t1 > 0, "t1 value has to be positive"
        assert t2 > 0, "t2 value has to be positive"
        _Filter.__init__(self, t1=t1, t2=t2,
                            rescale=rescale_driver,
                            optim=optim, 
                            optim_bayes=optim_bayes,
                            optim_bounds = optim_bounds,
                            amplitude=amplitude)
        
    def algorithm(self, signal):
        optim = self._params['optim']
        if optim:
            optim_bayes = self._params['optim_bayes']
            optim_bounds = self._params['optim_bounds']
            amplitude = self._params['amplitude']
            pars = optimize_T1_T2(signal, 
                                  optim_bayes, 
                                  optim_bounds,
                                  amplitude)
            
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

class PhasicEstim(_Filter):
    """
    Estimates the phasic and tonic components of a EDA driver function.
    
    Notes
    -----
    Please cite:
        
    """
    #TODO: add citation

    def __init__(self, amplitude=0.01, win_pre=2, win_post=2, polyfit=True, return_phasic=True):
        assert amplitude > 0, "Amplitude value has to be positive"
        assert win_pre > 0,  "Window pre peak value has to be positive"
        assert win_post > 0, "Window post peak value has to be positive"
        _Filter.__init__(self, amplitude=amplitude, win_pre=win_pre, win_post=win_post, polyfit=polyfit, return_phasic=return_phasic)

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
        signal['peaks'] = maxima
        
        peaks = _PeakSelection(win_pre=win_pre, win_post=win_post)(signal)
        
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
    
class PhasicEstimKalman(_Algorithm):
    """
    """
    
    def __init__(self, amplitude=0.01, 
                 return_phasic=True):
        assert amplitude > 0, "Amplitude value has to be positive"
        _Algorithm.__init__(self, amplitude=amplitude,
                            return_phasic=return_phasic)

    def algorithm(self, signal):
        params = self._params
        amplitude = params["amplitude"]
        
        return_phasic = params['return_phasic']

        signal_values = signal.p.get_values().ravel()

        R = _np.var(signal_values)
        
        signal_f = _IIRFilter(0.01, btype='highpass')(signal).p.get_values().ravel()
        Q = _np.var(_np.diff(signal_f))

        signal_k = _KalmanFilter(R, Q)(signal)
        
        driver_diff = abs(signal.p.get_values().ravel() - signal_k.p.get_values().ravel())
        
        idx_10 = create_signal((driver_diff < amplitude/2).astype(int), sampling_freq=signal.p.get_sampling_freq())
        idx_tonic = _np.where(idx_10.p.get_values().ravel() == 1)[0]
        
        if idx_tonic[0] != 0:
            idx_tonic = _np.insert(idx_tonic, 0, 0)
        
        if idx_tonic[-1] != (len(driver_diff) - 1):
            idx_tonic = _np.insert(idx_tonic, len(idx_tonic), len(driver_diff)-1)
        
        tonic = create_signal(signal_k.p.get_values().ravel()[idx_tonic], 
                              times = signal_k.p.get_times()[idx_tonic])
        
        tonic = tonic.interp({'time': signal.p.get_times()},
                             method='linear')
        
        tonic_values = tonic.p.get_values().ravel()

        if not return_phasic:
            return tonic_values
        
        phasic_values = signal_values - tonic_values
       
        return phasic_values

#%%    
