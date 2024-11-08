from simplify_algorithm import _Algorithm
import numpy as _np
import xarray as _xr
import pywt as _pywt
from scipy.signal import detrend as _detrend
    

class NoRolling(_Algorithm):
    """
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.chunk_dict = {}
        
    def algorithm(self, signal, **kwargs):
        return(_np.ones(shape=(1,1,10)))
    
    def __get_template__(self, signal):
        # template = self.__compute_template__(signal)
        # return(self.chunk_dict, template)
        pass

    def __finalize__(self, result_numpy, signal):
        result_out = self.__compute_template__(signal,
                                               out_dims={'time': 1,
                                                         'channel': 1,
                                                         'component': 10})
        result_out.values = result_numpy
        return(result_out)
    
class SimpleFilter(_Algorithm):
    """
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.chunk_dict = {'channel': 1, 'component': 1}
    
    def __get_template__(self, signal):
        template = self.__compute_template__(signal)
        return(self.chunk_dict, template)
    
    def algorithm(self, signal, **kwargs):
        signal_values = signal.values
        return(_np.zeros_like(signal_values))
    
    
    
class SimpleIndicator(_Algorithm):
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.chunk_dict = {'channel': 1, 'component': 1}
        
    def algorithm(self, signal, **kwargs):
        return(_np.zeros((1, signal.shape[1], signal.shape[2])))
    
    def __get_template__(self, signal):
        template = self.__compute_template__(signal, {'time': 1})
        return(self.chunk_dict, template)
    
class AddDimension(_Algorithm):
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.chunk_dict = {'channel': 1, 'component': 1}
        
    def algorithm(self, signal, **kwargs):
        size_new_dim = self._params['size_new_dim']
        signal_values = signal.values
        out_zeros = _np.zeros((signal_values.shape[0], 1, 1, 3))
        return(out_zeros)
    
    def __get_template__(self, signal):
        template = self.__compute_template__(signal, {'new_dim': [4, 7, 90]})
        return(self.chunk_dict, template)
    
class ShrinkDimension(_Algorithm):
    '''
    Keeps size in time but reduces the number of channels
    '''
    def __init__(self, n_channels_out, **kwargs):
        _Algorithm.__init__(self, n_channels_out=n_channels_out, **kwargs)
        self.chunk_dict = {'component': 1}
        
    def algorithm(self, signal, **kwargs):
        n_channels_out = self._params['n_channels_out']
        assert n_channels_out <= signal.shape[1]
        out_zeros = _np.zeros((signal.shape[0], 
                               n_channels_out,
                               signal.shape[2]))
        return(out_zeros)
    
    def __get_template__(self, signal):
        
        #new size of dim should be derived knowing the algorithm params
        n_channels_out = self._params['n_channels_out']
        template = self.__compute_template__(signal, {'channel': n_channels_out})
        
        return(self.chunk_dict, template)
    
class WhateverDimension(_Algorithm):
    '''
    Assume that a signal (timepoints x components) is processed to obtain a
    timepoints x new_component_dims x new_dimension
    
    (We dont work on component to test the rolling mechanism)
    '''
    def __init__(self, n_components_out, n_newdim_out, **kwargs):
        _Algorithm.__init__(self,
                            n_components_out=n_components_out,
                            n_newdim_out=n_newdim_out,
                            **kwargs)
        self.chunk_dict = {'channel': 1}
        
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
        n_components_out = self._params['n_components_out']
        n_newdim_out = self._params['n_newdim_out']
        
        template = self.__compute_template__(signal, {'component': n_components_out,
                                                      'new_dim': n_newdim_out})
        
        return(self.chunk_dict, template)
    
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
        
        self.chunk_dict = {'channel':1, 'component':1}
        
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
        self._compute_scales(signal)
        fsamp = signal.p.get_sampling_freq()
        freqs = self._params['freqs_nyq']*fsamp
        
        template = self.__compute_template__(signal, {'freq': freqs})
        
        return(self.chunk_dict, template)