# coding=utf-8
import numpy as _np

from ..BaseIndicator import Indicator as _Indicator
from ..BaseAlgorithm import Cache as _Cache
from ..indicators.FrequencyDomain import PowerInBand as _PowerInBand
import scipy.stats as _sps
from ..filters.Filters import ImputeNAN as _ImputeNAN
from ..Utility import PhUI as _PhUI
from ..Signal import EvenlySignal as _EvenlySignal

__author__ = 'AleB'

class SignalQualityIndicator(_Indicator):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.
    """
    def __init__(self, threshold, **kwargs):
        assert len(threshold)==2
        _Indicator.__init__(self, threshold=threshold, **kwargs)
    
    @classmethod
    def is_good(cls, output, params):
        # params = cls._params
        threshold = params['threshold']
        return(output >= threshold[0] and output <= threshold[1])
        
    @classmethod
    def run(cls, data, params=None, use_cache=False, **kwargs):
        if type(params) is dict:
            kwargs.update(params)
        if not isinstance(data.get_values(), _np.ndarray):
            _PhUI.w("The data must be a Signal (see class EvenlySignal and UnevenlySignal).")
            use_cache = False
        if use_cache is True:
            _Cache.cache_check(data)
            # noinspection PyTypeChecker
            return _Cache.run_cached(data, cls, kwargs)
        else:            
            if not data.is_multi():
                output = cls.algorithm(data, kwargs)
                isgood = cls.is_good(output, kwargs)
                return(output, isgood)
            else:
                data_values = data.get_values()
                values_out = []
                isgood_out = []
                for i_ch in range(data.get_nchannels()):
                    channel_ph = _EvenlySignal(data_values[:,i_ch], data.get_sampling_freq(), data.get_start_time())
                    output_ph = cls.algorithm(channel_ph, kwargs)
                    isgood_ph = cls.is_good(output_ph)
                    values_out.append(output_ph)
                    isgood_out.append(isgood_ph)
        
                # if output are signals, compose a multimodal instance
                if isinstance(values_out[0], _EvenlySignal):
                    values_out_np = _np.stack([x.get_values() for x in values_out], axis=1)
                    output = data.clone_properties(values_out_np)
                    
                    isgood_out_np = _np.stack([x.get_values() for x in isgood_out], axis=1)
                    isgood = data.clone_properties(isgood_out_np)
                    return(output, isgood)
                else:
                    return(values_out, isgood_out)


class Kurtosis(SignalQualityIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        SignalQualityIndicator.__init__(self, threshold, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        k = _sps.kurtosis(data.get_values())
        return(k)

class Entropy(_Indicator):
    def __init__(self, nbins=25, **kwargs):
        _Indicator.__init__(self, nbins=nbins, **kwargs)
    
    @classmethod
    def algorithm(cls, data, params):
        if _np.isnan(data).all():
            return(_np.nan)
        nbins=params['nbins']
        p_data = _np.histogram(data.get_values(), bins=nbins)[0]/len(data) # calculates the probabilities
        entropy = _sps.entropy(p_data)  # input probabilities to get the entropy 
        return(entropy)

class DerivativeEnergy(_Indicator):
    """
    Compute the Derivative Energy

    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)
    
    @classmethod
    def algorithm(cls, data, params):
        x = data.get_values()
        de = _np.sqrt(_np.nanmean(_np.power(_np.diff(x), 2)))
        return(de)
        
class SpectralPowerRatio(_Indicator):
    """
    Compute the Spectral Power Ratio

    """
    def __init__(self, method='ar', bandN=[5,14], bandD=[5,50],**kwargs):
        _Indicator.__init__(self, method=method, bandN=bandN, bandD=bandD, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        bandN = params['bandN']
        bandD = params['bandD']
        assert bandD[1] < data.get_sampling_freq()/2, 'The higher frequency in bandD is greater than fsamp/2: cannot compute power' # CHECK: check sampling frequency of the signal (e.g. <=128)
        p_N = _PowerInBand(bandN[0], bandN[1], params['method'])(data)
        p_D = _PowerInBand(bandD[0],bandD[1], params['method'])(data)
        return(p_N/p_D)

class CVSignal(_Indicator):
    """
    Compute the Coefficient of variation of the signal
    
    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859838/

    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        mean = _np.nanmean(data)
        sd = _np.nanstd(data)
        cv = sd/mean
        return(cv)

class PercentageNAN(_Indicator):
    """
    Compute the Percentage of NaNs

    """
    def __init__(self, **kwargs):
        _Indicator.__init__(self, **kwargs)

    @classmethod
    def algorithm(cls, data, params):
        n_nan = _np.sum(_np.isnan(data))
        data = _ImputeNAN()(data)
        return(100*n_nan/len(data))
