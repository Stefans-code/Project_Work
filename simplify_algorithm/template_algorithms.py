from simplify_algorithm import _Algorithm
import numpy as _np
import xarray as _xr
import pywt as _pywt
from scipy.signal import detrend as _detrend
    

# class NoRolling(_Algorithm):
#     """ WIP
#     """

#     def __init__(self, **kwargs):
#         _Algorithm.__init__(self, **kwargs)
#         self.required_dims = []
        
#     def algorithm(self, signal, **kwargs):
#         return(_np.ones(shape=(1,1,10)))
    
#     def __get_template__(self, signal):
#         chunk_dict = {}
#         template = self.__compute_template__(signal)
        # return(chunk_dict, template)
    
class SimpleFilter(_Algorithm):
    """
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = ['time']
    
    def __get_template__(self, signal):
        return(self.__get_template_timeonly__(signal))
    
    def algorithm(self, signal, **kwargs):
        in_shape = signal.values.shape
        out = _np.zeros(in_shape[0])
        print(out.shape)
        return(out)
    
    
class SimpleIndicator(_Algorithm):
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = ['time']
        
    def algorithm(self, signal, **kwargs):
        out = _np.array(0)      
        print(out.shape)
        return(out)
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal, {'time': 1})
        return(chunk_dict, template)


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
        
    def __check_good__(self, sqi_indicator, signal):
        params = self._params
        threshold = params['threshold']
        
        is_good = (sqi_indicator >= threshold[0]) & (sqi_indicator <= threshold[1])
        
        out_shape = _np.array(signal.shape)
        out_shape[0] = 1
        
        sqi_indicator = _np.array(sqi_indicator).reshape(out_shape)
        is_good = _np.array(is_good).reshape(out_shape)
        
        out = _np.stack([sqi_indicator, is_good], axis = - 1)
        
        return(out)
    
class SimpleSQIIndicator(_SignalQualityIndicator):
    """
    Compute the Kurtosis of the signal
    
    """
    def __init__(self, threshold, **kwargs):
        _SignalQualityIndicator.__init__(self, threshold=threshold, **kwargs)
        self.required_dims = ['time']
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal, {'time': 1, 
                                                      'is_good': 2})
        return(chunk_dict, template)
     
    def algorithm(self, signal):
        sqi_indicator = 3
        sqi_out = self.__check_good__(sqi_indicator, signal)
        
        return(sqi_out)
    
class AddDimension(_Algorithm):
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = ['time']
        
    def algorithm(self, signal, **kwargs):
        size_new_dim = self._params['size_new_dim']
        out_dims = [x for x in signal.shape]
        out_dims.append(size_new_dim)
        out_zeros = _np.zeros(out_dims)
        return(out_zeros)
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal, {'new_dim': self._params['size_new_dim']})
        return(chunk_dict, template)
    
class ShrinkDimension(_Algorithm):
    '''
    Keeps size in time but reduces the number of channels
    '''
    def __init__(self, n_channels_out, **kwargs):
        _Algorithm.__init__(self, n_channels_out=n_channels_out, **kwargs)
        self.required_dims = ['time', 'channel']
        
    def algorithm(self, signal, **kwargs):
        n_channels_out = self._params['n_channels_out']
        assert n_channels_out <= signal.shape[1]
        
        out_dims = [x for x in signal.shape]
        out_dims[1] = n_channels_out
        out_zeros = _np.zeros(out_dims)
        
        return(out_zeros)
    
    def __get_template__(self, signal):
        
        #new size of dim should be derived knowing the algorithm params
        chunk_dict = self.__compute_chunk_dict__(signal)
        n_channels_out = self._params['n_channels_out']
        template = self.__compute_template__(signal, {'channel': n_channels_out})
        return(chunk_dict, template)
    
class WhateverDimension(_Algorithm):
    '''
    Assume that a signal (timepoints x channels x ... ) is processed to obtain a
    timepoints x channels x new_component_dims x new_dimension
    
    We dont alter the size of channels to test the rolling mechanism; 
    chunk_dict will be {'channel':1}
    '''
    
    def __init__(self, n_components_out, n_newdim_out, **kwargs):
        _Algorithm.__init__(self,
                            n_components_out=n_components_out,
                            n_newdim_out=n_newdim_out,
                            **kwargs)
        self.required_dims = ['time', 'component']
        
    def algorithm(self, signal, **kwargs):
        
        n_times_out = signal.shape[0]
        n_channels_out = signal.shape[1]
        n_components_out = self._params['n_components_out']
        n_newdim_out = self._params['n_newdim_out']
        
        out_data = _np.zeros((n_times_out, 
                              n_channels_out, 
                              n_components_out,
                              n_newdim_out))
        return(out_data)
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        n_components_out = self._params['n_components_out']
        n_newdim_out = self._params['n_newdim_out']
        
        template = self.__compute_template__(signal, {'component': n_components_out,
                                                      'new_dim': n_newdim_out})
        
        return(chunk_dict, template)
    
class Wavelet(_Algorithm):
    """

    """
    def __init__(self, wtype = 'cmor_1.15-1.0',
                 freqs = None,
                 minScale = 2,
                 nNotes = 12,
                 detrend=True,
                 normalize=False,
                 compute_coi=False):
            
        
        _Algorithm.__init__(self, wtype = wtype, freqs = freqs,
                            minScale = minScale, nNotes = nNotes,
                            detrend=detrend, normalize=normalize,
                            compute_coi=compute_coi)
        
        self.required_dims = ['timel']
        
    def _compute_coi(self, W):
        N = W.shape[1]
        freqs_nyq = self._params['freqs_nyq']
        coif_ = 1/(2*_np.arange(1, N//2))

        # coif_ = fsamp/(2*np.arange(1, N//2))
        min_coif = coif_[-1]
        coif = _np.zeros(N) + min_coif
        coif[:len(coif_)] = coif_
        coif[-len(coif_):] = coif_[::-1]
        
        for i in range(W.shape[1]):
            idx_na = _np.where(freqs_nyq < coif[i])[0]
            W[idx_na, i] = _np.nan
        return(W)
            
    
    def _compute_scales(self, signal):
        params = self._params
        freqs = params['freqs']
        wtype = params['wtype']
        
        signal_values = signal.p.get_values()
        fsamp = signal.p.get_sampling_freq()
        
        if freqs is None: #users want the algoritm to compute the scales
            minScale = params['minScale']
            nNotes = params['nNotes']
            
            # The scales as of Mallat 1999
            # minScale = 2 # / wavelet.flambda()
            N = signal_values.shape[0]
            nOctaves = int(_np.round(_np.log2(N/2) / (1/nNotes)))
            scales = minScale * 2 ** (_np.arange(0, nOctaves + 1) * (1/nNotes))
            freqs_nyq = _pywt.scale2frequency(wtype, scales)
            
        else: #user provided the frequencies
            freqs = _np.array(freqs)
            #check correct order of frequencies
            assert freqs[0]>freqs[-1]
            assert (_np.diff(freqs)<0).all()
            freqs_nyq = freqs/fsamp
            scales = _pywt.frequency2scale(wtype, freqs_nyq)
            
        scales = _np.sort(scales)
        
        self._params['freqs_nyq'] = freqs_nyq
        self._params['scales'] = scales
        
    
    def algorithm(self, signal):
        if 'scales' not in self._params:
            self._compute_scales(signal)
        
        params = self._params
        #get signal values and info
        signal_values = signal.p.get_values().ravel()
        fsamp = signal.p.get_sampling_freq()
        N = len(signal_values)
        
        #remove linear drift
        detrend = params['detrend']
        if detrend:
            signal_values = _detrend(signal_values, type='linear')
        
        #compute wavelet
        wtype = params['wtype']
        scales = params['scales']
        
        W, freqs_nyq = _pywt.cwt(signal_values, scales, wavelet=wtype)
        
        
        freqs=freqs_nyq*fsamp
        self._params['freqs_nyq'] = freqs_nyq
        
        #normalize computed W
        normalize = params['normalize']
        if normalize:
            scaleMatrix = _np.ones([1, N]) * scales[:, None]
            W = W**2 / scaleMatrix
        
        #compute coi and assign na outside
        compute_coi = params['compute_coi']
        if compute_coi:
            W = self._compute_coi(W)
        
        W = W.T
        W = W[:, _np.newaxis, _np.newaxis, :]
        out = signal.copy(deep=True)
        
        out = out.expand_dims({'freq':freqs.astype(_np.float64)}, axis=-1)
        
        out.values = W
        return out
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        
        self._compute_scales(signal)
        fsamp = signal.p.get_sampling_freq()
        freqs = self._params['freqs_nyq']*fsamp
        
        template = self.__compute_template__(signal, {'freq': freqs})
        
        return(chunk_dict, template)