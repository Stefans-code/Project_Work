# coding=utf-8
import numpy as _np
from .indicators.frequencydomain import PowerInBand as _PowerInBand
from .filters import IIRFilter as _IIRFilter, Normalize as _Normalize
import scipy.stats as _sps
from ._base_algorithm import _Algorithm
from .utils import PSD as _PSD

class _SQIIndicator(_Algorithm):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.

    Args:
        threshold (low, high): The range within which the sqi indicates good quality
    
    Returns:
        result : xarray dataarrat with the new variables: the sqi and 'is_good'
    
    """
    def __init__(self, threshold, **kwargs):
        assert len(threshold)==2
        _Algorithm.__init__(self, threshold=threshold, **kwargs)
        
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        t_out = signal['time'][0]
        template = self.__compute_template__(signal, {'time': [t_out], 
                                                      'is_good': 2})
        return(chunk_dict, template)
        
    def __check_good__(self, sqi_indicator, signal):
        params = self._params
        threshold = params['threshold']
        
        is_good = (sqi_indicator >= threshold[0]) & (sqi_indicator <= threshold[1])
        
        _, template = self.__get_template__(signal)
        out_shape = [template.sizes[d] for d in signal.dims]
        
        sqi_indicator = _np.array(sqi_indicator).reshape(out_shape)
        is_good = _np.array(is_good).reshape(out_shape)
        
        out = _np.stack([sqi_indicator, is_good], axis = -1)
        return(out)

class Kurtosis(_SQIIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        _SQIIndicator.__init__(self, threshold, **kwargs)
        self.required_dims = ['time']
    
    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        k = _sps.kurtosis(signal_values)
        
        k_out = self.__check_good__(k, signal)
        return k_out

class Entropy(_SQIIndicator):
    def __init__(self, threshold, nbins=25, **kwargs):
        _SQIIndicator.__init__(self, threshold, nbins=nbins, **kwargs)
        self.required_dims = ['time']
    
    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        params = self._params
        nbins=params['nbins']
        p_data = _np.histogram(signal_values.reshape(-1,1), bins=nbins)[0]/len(signal_values) # calculates the probabilities
        entropy = _sps.entropy(_np.array(p_data))  # input probabilities to get the entropy 
        entropy_out = self.__check_good__(entropy, signal)
        return entropy_out
        
class SpectralPowerRatio(_SQIIndicator):
    """
    Compute the Spectral Power Ratio

    """
    def __init__(self, threshold, method='ar', bandN=None, bandD=None, **kwargs):
        _SQIIndicator.__init__(self, threshold, method=method, bandN=bandN, bandD=bandD, **kwargs)
        self.required_dims = ['time']
    
    def algorithm(self, signal):
        params = self._params
        method = params['method']
        bandN = params['bandN']
        bandD = params['bandD']
        assert bandD[1] < signal.p.get_sampling_freq()/2, 'The higher frequency in bandD is greater than fsamp/2: cannot compute power' # CHECK: check sampling frequency of the signal (e.g. <=128)
        
        psd = _PSD(method=method)(signal)
        
        power_N = psd.query(freq=f"freq >= {bandN[0]} & freq <= {bandN[1]}").sum(dim='freq')
        power_D = psd.query(freq=f"freq >= {bandD[0]} & freq <= {bandD[1]}").sum(dim='freq')
        
        
        spr = float(power_N)/float(power_D)
        spr_out = self.__check_good__(spr, signal)
        return spr_out
    
class CVSignal(_SQIIndicator):
    """
    Compute the Coefficient of variation of the signal
    
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859838/
    And Morais et al. 2018

    """
    def __init__(self, threshold, **kwargs):
        _SQIIndicator.__init__(self, threshold, **kwargs)
        self.required_dims = ['time']

    def algorithm(self, signal):
        signal_values = signal.values.ravel()
        mean = _np.mean(signal_values)
        sd = _np.std(signal_values)
        cv = float(100*sd/mean)
        cv_out = self.__check_good__(cv, signal)
        return cv_out

class PercentageNAN(_SQIIndicator):
    """
    Compute the Percentage of NaNs

    """
    def __init__(self, threshold, **kwargs):
        _SQIIndicator.__init__(self, threshold, **kwargs)
        self.required_dims = ['time']

    def algorithm(self, signal):
        signal_values = signal.values
        n_nan = _np.sum(_np.isnan(signal_values))
        perc = 100*n_nan/len(signal_values)
        perc_out = self.__check_good__(perc, signal)
        return perc_out
