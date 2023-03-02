import numpy as _np
from .._base_algorithm import _Algorithm
from ..utils import PSD as PSD

# __author__ = 'AleB'

class PowerInBand(_Algorithm):
    """
    Estimate the power in given frequency band

    Parameters
    ----------
    freq_min : float, >0
        Left bound of the frequency band
    freq_max : float, >0
        Right bound of the frequency band
    method : 'ar', 'welch' or 'fft'
        Method to estimate the PSD
        
    Additional parameters
    ---------------------
    For the PSD (see pyphysio.tools.Tools.PSD):
        
    interp_freq : float, >0
        Frequency used to (re-)interpolate the signal

    Returns
    -------
    power : float
        Power in the frequency band
    """

    def __init__(self, freq_min, freq_max, method, **kwargs):
        _Algorithm.__init__(self, freq_min=freq_min, freq_max=freq_max, method=method, **kwargs)
        self.dimensions = {'time' : 1}

    def algorithm(self, signal):
        params = self._params
        fsamp = signal.p.get_sampling_freq()
        psd = PSD(scaling='density', **params)(signal)
        freq = psd.coords['freq'].values
        power = psd.values
        i_min = _np.searchsorted(freq, params["freq_min"])
        i_max = _np.searchsorted(freq, params["freq_max"])
        result = _np.sum(power[i_min:i_max]*fsamp, axis=0, keepdims=True)
        return result

class PeakInBand(_Algorithm):
    """
    Estimate the peak frequency in a given frequency band

    Parameters
    ----------
    freq_min : float, >0
        Left bound of the frequency band
    freq_max : float, >0
        Right bound of the frequency band
    method : 'ar', 'welch' or 'fft'
        Method to estimate the PSD
        
    Additional parameters
    ---------------------
    For the PSD (see pyphysio.tools.Tools.PSD):
        
    interp_freq : float, >0
        Frequency used to (re-)interpolate the signal

    Returns
    -------
    peak : float
        Peak frequency
    """

    def __init__(self, freq_min, freq_max, method, **kwargs):
        _Algorithm.__init__(self, freq_min=freq_min, freq_max=freq_max, method=method, **kwargs)
        self.dimensions = {'time' : 1}
    
    def algorithm(self, signal):
        
        params = self._params
        
        psd = PSD(**params)(signal)
        
        freq = psd.coords['freq'].values
        power = psd.values.ravel()
        
        i_min = _np.searchsorted(freq, params["freq_min"])
        i_max = _np.searchsorted(freq, params["freq_max"])
        
        f_band = freq[i_min:i_max]
        p_band = power[i_min:i_max]
        f_peak = f_band[_np.argmax(p_band)]
        
        return _np.array([f_peak])

