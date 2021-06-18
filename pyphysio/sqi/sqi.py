# coding=utf-8
import numpy as _np
import numpy.ma as _ma
from . import SignalQualityIndicator as _SignalQualityIndicator
from ..indicators.frequencydomain import PowerInBand as _PowerInBand
import scipy.stats as _sps
from ..processing.filters import ImputeNAN as _ImputeNAN


class Kurtosis(_SignalQualityIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)

    def algorithm(self, data):
        k = _sps.kurtosis(data.get_values().ravel())
        return(k)

class Entropy(_SignalQualityIndicator):
    def __init__(self, threshold, nbins=25, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, nbins=nbins, **kwargs)
    
    def algorithm(self, data):
        params = self._params
        nbins=params['nbins']
        p_data = _np.histogram(data.get_values().ravel().reshape(-1,1), bins=nbins)[0]/len(data) # calculates the probabilities
        entropy = _sps.entropy(_np.array(p_data))  # input probabilities to get the entropy 
        return entropy

class DerivativeEnergy(_SignalQualityIndicator):
    """
    Compute the Derivative Energy

    """
    def __init__(self, threshold, **kwargs):
        #TODO, use Diff with custom spacing
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)
    
    def algorithm(self, data):
        x = data.get_values().ravel()
        de = _np.sqrt(_np.nanmean(_np.power(_np.diff(x), 2)))
        return(de)
        
class SpectralPowerRatio(_SignalQualityIndicator):
    """
    Compute the Spectral Power Ratio

    """
    def __init__(self, threshold, method='ar', bandN=[5,14], bandD=[5,50],**kwargs):
        _SignalQualityIndicator.__init__(self, threshold, method=method, bandN=bandN, bandD=bandD, **kwargs)

    
    def algorithm(self, data):
        params = self._params
        bandN = params['bandN']
        bandD = params['bandD']
        assert bandD[1] < data.get_sampling_freq()/2, 'The higher frequency in bandD is greater than fsamp/2: cannot compute power' # CHECK: check sampling frequency of the signal (e.g. <=128)
        p_N = _PowerInBand(bandN[0], bandN[1], params['method'])(data)
        p_D = _PowerInBand(bandD[0],bandD[1], params['method'])(data)
        return(p_N/p_D)

class CVSignal(_SignalQualityIndicator):
    """
    Compute the Coefficient of variation of the signal
    
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859838/
    And Morais et al. 2018

    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)

    def algorithm(self, data):
        data_values= data.get_values()
        mean = _ma.mean(data_values)
        sd = _ma.std(data_values)
        cv = 100*sd/mean
        return(cv)

class PercentageNAN(_SignalQualityIndicator):
    """
    Compute the Percentage of NaNs

    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold, **kwargs)

    def algorithm(self, data):
        n_nan = _np.sum(_np.isnan(data))
        data = _ImputeNAN()(data)
        return(100*n_nan/len(data))
