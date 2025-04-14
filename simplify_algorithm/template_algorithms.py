from simplify_algorithm import _Algorithm
import numpy as _np
import xarray as _xr
import pywt as _pywt
from scipy.signal import detrend as _detrend
    

class NoRolling(_Algorithm):
    """
    To avoid the rolling mechanism, required_dims should be an empty list.
    
    The input signal will be processed as is by the algorithm function.
    
    To refine the output, you can write the function __finalize__.
    The function __finalize__ will be called after the algorithm function, with the following parameters:
    - result_algorithm: the result from the call of the function algorithm
    - signal: the input signal
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = []
        
    def algorithm(self, signal, **kwargs):
        return(_np.ones(shape=(1,1,10)))
    
    def __finalize__(self, result_algorithm, signal):
        '''
        This function is called after the algorithm function.
        It can be used to refine the output.
        
        Args:
            result_algorithm: the result from the call of the function algorithm
            signal: the input signal
        '''
        #do something with the result_algorithm
        # and the signal
        signal_out = _xr.DataArray(result_algorithm, coords = {'time': [0], 
                                                               'channel': [0],
                                                               'component': _np.arange(10)},)
        
        return(signal_out)
    
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
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal, {'time': 1, 
                                                      'is_good': 2})
        return(chunk_dict, template)
    
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

class AlgorithmUsingSupportingSignal(_Algorithm):
    '''
    First, add the supporting signal should be added to the signal to process
    as a new coordinate.
    The supporting signal should be a signal with the same coordinates and dimensions of the signal to process.
    For instance a signal resulting from the application of another algorithm to the same signal to process
    
    signal['new_coord'] = supporting_signal
    
    The new coordinate will be also divided into chunks to allow processing in parallel.
    But the new coordinate should be added to the result 
    before the results from the different chunks are merged.
    This is because the template is created based on the signal with the additional coordinate.
    
    To this aim, it is necessary to define the __mapper__function__ properly.
    See example below.
    '''
    
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = []
        
    def algorithm(self, signal):
        assert 'new_coord' in signal.coords
        supporting_signal = signal['new_coord'];
        #do something using the signal to process (signal)
        # and the supporting signal
        # note that the result will be a numpy array
        return(signal.values)

    def __mapper_func__(self, signal_in, **kwargs):
        out = super().__mapper_func__(signal_in, **kwargs)
        out['new_coord'] = signal_in['new_coord']
        return(out)
