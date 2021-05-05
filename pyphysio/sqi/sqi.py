# coding=utf-8
import numpy as _np

from ..processing import Algorithm as _Algorithm
from ..indicators.frequencydomain import PowerInBand as _PowerInBand
import scipy.stats as _sps
from ..processing.filters import ImputeNAN as _ImputeNAN
from ..signal import Signal as _Signal

class SignalQualityIndicator(_Algorithm):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.
    """
    def __init__(self, threshold, **kwargs):
        assert len(threshold)==2
        _Algorithm.__init__(self, threshold=threshold, **kwargs)
        
    def is_good(self, sqi_values):
        params = self._params
        threshold = params['threshold']
        
        output = _np.zeros_like(sqi_values)
        idx_good = _np.where((sqi_values >= threshold[0]) & (sqi_values <= threshold[1]))
        output[idx_good] = 1
        return(output)
        
    def __call__(self, data):
        """
        Executes the algorithm using the parameters saved by the constructor.
        @param data: The data.
        @type data: TimeSeries
        @return: The result.
        """
        
        assert isinstance(data, _Signal), "The data must be a Signal."
            
        values_out = _np.apply_along_axis(self.algorithm, 0, data)
        isgood = self.is_good(values_out)
        return(values_out, isgood)

class Kurtosis(SignalQualityIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        SignalQualityIndicator.__init__(self, threshold, **kwargs)

    def algorithm(self, data):
        k = _sps.kurtosis(data.get_values())
        return(k)

class Entropy(SignalQualityIndicator):
    def __init__(self, threshold, nbins=25, **kwargs):
        SignalQualityIndicator.__init__(self, threshold, nbins=nbins, **kwargs)
    
    def algorithm(self, data):
        params = self._params
        if _np.isnan(data).all():
            return(_np.nan)
        nbins=params['nbins']
        p_data = _np.histogram(data, bins=nbins)[0]/len(data) # calculates the probabilities
        entropy = _sps.entropy(p_data)  # input probabilities to get the entropy 
        return(entropy)

class DerivativeEnergy(SignalQualityIndicator):
    """
    Compute the Derivative Energy

    """
    def __init__(self, threshold, **kwargs):
        #TODO, use Diff with custom spacing
        SignalQualityIndicator.__init__(self, threshold, **kwargs)
    
    def algorithm(self, data):
        x = data.get_values()
        de = _np.sqrt(_np.nanmean(_np.power(_np.diff(x), 2)))
        return(de)
        
class SpectralPowerRatio(SignalQualityIndicator):
    """
    Compute the Spectral Power Ratio

    """
    def __init__(self, threshold, method='ar', bandN=[5,14], bandD=[5,50],**kwargs):
        SignalQualityIndicator.__init__(self, threshold, method=method, bandN=bandN, bandD=bandD, **kwargs)

    
    def algorithm(self, data):
        params = self._params
        bandN = params['bandN']
        bandD = params['bandD']
        assert bandD[1] < data.get_sampling_freq()/2, 'The higher frequency in bandD is greater than fsamp/2: cannot compute power' # CHECK: check sampling frequency of the signal (e.g. <=128)
        p_N = _PowerInBand(bandN[0], bandN[1], params['method'])(data)
        p_D = _PowerInBand(bandD[0],bandD[1], params['method'])(data)
        return(p_N/p_D)

class CVSignal(SignalQualityIndicator):
    """
    Compute the Coefficient of variation of the signal
    
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859838/

    """
    def __init__(self, threshold, **kwargs):
        SignalQualityIndicator.__init__(self, threshold, **kwargs)

    def algorithm(self, data):
        mean = _np.nanmean(data)
        sd = _np.nanstd(data)
        cv = sd/mean
        return(cv)

class PercentageNAN(SignalQualityIndicator):
    """
    Compute the Percentage of NaNs

    """
    def __init__(self, threshold, **kwargs):
        SignalQualityIndicator.__init__(self, threshold, **kwargs)

    def algorithm(self, data):
        n_nan = _np.sum(_np.isnan(data))
        data = _ImputeNAN()(data)
        return(100*n_nan/len(data))
