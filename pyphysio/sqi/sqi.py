# coding=utf-8
import numpy as _np
import numpy.ma as _ma
from . import SignalQualityIndicator as _SignalQualityIndicator
from ..indicators.frequencydomain import PowerInBand as _PowerInBand
import scipy.stats as _sps
from ..processing.filters import ImputeNAN as _ImputeNAN
from ..processing.tools import Diff as _Diff

class Kurtosis(_SignalQualityIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)
        self.dimensions = {'time':1}

    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        k = _sps.kurtosis(signal_values)
        k_out = _np.array([[k]])
        return(k_out)

class Entropy(_SignalQualityIndicator):
    def __init__(self, threshold, nbins=25, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, nbins=nbins, **kwargs)
        self.dimensions = {'time':1}
    
    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        params = self._params
        nbins=params['nbins']
        p_data = _np.histogram(signal_values.reshape(-1,1), bins=nbins)[0]/len(signal_values) # calculates the probabilities
        entropy = _sps.entropy(_np.array(p_data))  # input probabilities to get the entropy 
        entropy_out = _np.array([[entropy]])
        return entropy_out

class DerivativeEnergy(_SignalQualityIndicator):
    """
    Compute the Derivative Energy

    """
    def __init__(self, threshold, dt=0.01, **kwargs):
        assert dt>0
        _SignalQualityIndicator.__init__(self, threshold, dt = dt, **kwargs)
        self.dimensions = {'time':1}
    
    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        degree = int(signal.p.get_sampling_freq()*self.params['dt'])
        de = _np.sqrt(_np.mean(_np.power(_Diff(degree=degree)(signal).values, 2)))
        de_out = _np.array([[de]])
        return de_out
        
class SpectralPowerRatio(_SignalQualityIndicator):
    """
    Compute the Spectral Power Ratio

    """
    def __init__(self, threshold, method='ar', bandN=[5,14], bandD=[5,50],**kwargs):
        _SignalQualityIndicator.__init__(self, threshold, method=method, bandN=bandN, bandD=bandD, **kwargs)
        self.dimensions = {'time':1}

    
    def algorithm(self, signal):
        params = self._params
        bandN = params['bandN']
        bandD = params['bandD']
        assert bandD[1] < signal.p.get_sampling_freq()/2, 'The higher frequency in bandD is greater than fsamp/2: cannot compute power' # CHECK: check sampling frequency of the signal (e.g. <=128)
        p_N = _PowerInBand(bandN[0], bandN[1], params['method'])(signal)
        p_D = _PowerInBand(bandD[0],bandD[1], params['method'])(signal)
        return(_np.array(p_N/p_D))

class CVSignal(_SignalQualityIndicator):
    """
    Compute the Coefficient of variation of the signal
    
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859838/
    And Morais et al. 2018

    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)
        self.dimensions = {'time':1}

    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        mean = _np.mean(signal_values)
        sd = _np.std(signal_values)
        cv = float(100*sd/mean)
        
        return _np.array([cv])

class PercentageNAN(_SignalQualityIndicator):
    """
    Compute the Percentage of NaNs

    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)
        self.dimensions = {'time':1}

    def algorithm(self, signal):
        signal_values = signal.values
        n_nan = _np.sum(_np.isnan(signal_values))
        perc = 100*n_nan/len(signal_values)
        return _np.array([[perc]])
