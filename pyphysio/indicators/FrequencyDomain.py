# coding=utf-8
# from __future__ import division

from ..BaseAlgorithm import Algorithm as _Algorithm
from ..tools.Tools import PSD as PSD
import numpy as _np

# __author__ = 'AleB'


class InBand(_Algorithm):
    """
    Extract the PSD of a given frequency band
    

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
    For the PSD (see pyphysio.tools.Tools.PSD), for instance:
        
    interp_freq : float, >0
        Frequency used to (re-)interpolate the signal

    Returns
    -------
    freq : numpy array
        Frequencies in the frequency band
    psd : float
        Power Spectrum Density in the frequency band
    """

    def __init__(self, freq_min, freq_max, method, **kwargs):
        _Algorithm.__init__(self, freq_min=freq_min, freq_max=freq_max, method=method, **kwargs)

    def algorithm(self, signal):
        params = self._params
        freq, spec = PSD(**params)(signal)
        # freq is sorted so
        i_min = _np.searchsorted(freq, params["freq_min"])
        i_max = _np.searchsorted(freq, params["freq_max"])
        return freq[i_min:i_max], spec[i_min:i_max]


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

    def algorithm(self, signal):
        params = self._params
        freq, powers = InBand(**params)(signal)
        return _np.sum(powers)


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
    
    def algorithm(self, signal):
        params = self._params
        freq, power = InBand(**params)(signal)
        return freq[_np.argmax(power)]

