# coding=utf-8
import numpy as _np
from .indicators.frequencydomain import PowerInBand as _PowerInBand
import scipy.stats as _sps
from .utils import Diff as _Diff
import xarray as _xr
from ._base_algorithm import _Algorithm




class _SignalQualityIndicator(_Algorithm):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.

    Args:
        threshold (low, high): The range within which the sqi indicates good quality
    
    Returns:
        result (sqi, isgood): Tuple containing the value of the sqi and if it corresponds to good quality
    
    """
    def __init__(self, threshold, **kwargs):
        '''
        '''
        assert len(threshold)==2
        _Algorithm.__init__(self, threshold=threshold, **kwargs)
    
    def is_good(self, sqi_dataarray):
        # print('-----> is_good')
        sqi_values = sqi_dataarray.values
        params = self._params
        threshold = params['threshold']
        
        if sqi_values.ndim == 0:
            output = (sqi_values >= threshold[0]) & (sqi_values <= threshold[1])
            output = _np.array(output)
        else:
            output = _np.zeros_like(sqi_values)
            idx_good = _np.where((sqi_values >= threshold[0]) & (sqi_values <= threshold[1]))
            output[idx_good] = 1
            #propagate nans
            idx_nan = _np.where(_np.isnan(sqi_values))
            output[idx_nan] = _np.nan
        
        # print('<----- is_good')
        return(output)
        
    def __call__(self, signal, add_signal=True, dimensions=None):
        # print('-----> SQI.__call__()')
        values_out = super().__call__(signal, add_signal=add_signal, 
                                      dimensions=dimensions)
        
        signal_name = signal.p.main_signal.name
        if add_signal:
            indicator_name = signal_name+'_'+self.name
        else:
            indicator_name = signal_name
        
        #for SQI that are called from within other algorithms
        if isinstance(values_out, _xr.Dataset):
            isgood = self.is_good(values_out[indicator_name])
            #convert isgood to dataarray
            isgood_out = values_out[indicator_name].copy(data = isgood)
        else:
            values_out.name = indicator_name
            isgood = self.is_good(values_out)
            #convert isgood to dataarray
            isgood_out = values_out.copy(data = isgood)
        
        
        isgood_name = indicator_name +'_isgood'
        isgood_out.name = isgood_name
        
        out = _xr.merge([values_out, isgood_out])
        # print('<----- SQI.__call__()')
        return(out)

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
        k_out = _np.array([k])
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
        # signal_values = signal.values.ravel()
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