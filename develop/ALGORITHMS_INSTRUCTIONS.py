'''
TODO: Explain call procedure with all steps

from . import Algorithm as _Algorithm

class NewAlgorithm(_Algorithm):
    def __init__(self, parameters (with default values)):
        
        #assert section to validate the parameter values
        
        #superclass initialization
        _Algorithm.__init__(self, 
                            param_name1=param_name1,
                            param_name2=param_name2,
                            ...,
                            **kwargs)
        
        
        #operating dimensions to regulate the "rolling" mechanism
        #all dimensions not specified are "rolled" (executed in parallel)
        
        self.dimensions= {}
        
        # dictionary where:
        #     key is the dimensions along which the algorithm operates.
        #     item is an int that defines the new size of the dimension;
        #         0 means that the size is the same as the input
        #         1 means that the output is a scalar (e.g indicator, mean)
        # OR 'none' to avoid the "rolling" mechanism
        
        # Examples:
        #     {'time': 0}: operate along each individual 1d signal 
        #                  (rolls on channels x components)
        #     {'time': 1}: indicator: computes a scalar from 1d signal.
        #     {'time': X}: some special cases may occurr when the output has 
        #                   some other fixed or known size
        #     {'time': X, 'channel': Y}: uses information across channels
        #     {'time': X, 'components': Y}: uses information across components
        #     #TODO: other examples!!!
        
    
    def __get_template(self, signal_in):
        
    def algorithm(cls, signal_in):
        
        # signal is a xarray.DataArray
        # three dimensions (time, channel, component)
                
        
        #extract methods
        params = self._params
        param_name1 = params['param_name1']
        
        #typically we want to operate on numpy arrays 
        #representing signal values
        signal_values = signal_in.values
        
        #DO SMTH WITH signal_values
        result = ...(signal_values)
        
        # result should be a 3d numpy array
        # size of each dimension should comply with what specified in the
        # self.dimensions dictionary (__init__)
        
        return result
    
    def __finalize__(self, result, signal_in, dimensions='none'):
        Can be implemented in special cases
        to give a coherent output
'''
#%%
from pyphysio.signal import create_signal
import numpy as np

from pyphysio.processing import Algorithm as _Algorithm


class VerboseAlgorithm(_Algorithm):
    def __init__(self, **kwargs):
        print('__init__')
        _Algorithm.__init__(self, **kwargs)
    
    def __mapper_func__(self, signal_in):
        print('>>> __mapper_func__')
        print(type(signal_in))
        print(signal_in.shape)
        result = _Algorithm.__mapper_func__(self, signal_in)
        print(type(result))
        print(result.shape)
        print('<<< __mapper_func__')
        return(result)

    def __call__(self, signal_in, add_signal=True, dimensions=None):
        print('>>> __call__')
        print(type(signal_in))
        print(signal_in.dims)
        result = _Algorithm.__call__(self, signal_in, add_signal, dimensions=dimensions, scheduler='synchronous')
        print(type(result))
        print(result.dims)
        print('<<< __call__')
        return(result)
    
    def __finalize__(self, result, signal_in, dimensions='none'):
        print('>>> __finalize__')
        print(type(result))
        print(result.shape)
        
        print(type(signal_in))
        print(signal_in.shape)
        
        result = _Algorithm.__finalize__(self, result, signal_in, dimensions)
        print(type(result))
        print(result.shape)
        print('<<< __finalize__')
        return(result)
    
# CREATE SAMPLE SIGNAL
n_ch = 3
n_cp = 2
data = np.random.uniform(size = (10000, n_ch, n_cp)) + np.random.uniform(0, 10, size = (1,n_ch,n_cp))
sampling_freq = 1000
signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')


#%% SIMPLE 1D SIGNAL PROCESSING STEP
class Simple1d(VerboseAlgorithm):
    """
    simple algorithm that adds a value to each channel
    
    Parameters
    -------------------
    number_to_add : float, default = 2.0        
    """

    def __init__(self, number_to_add=2.0, **kwargs):
        assert isinstance(number_to_add, float)
        VerboseAlgorithm.__init__(self, number_to_add=number_to_add, **kwargs)
        
        self.dimensions = {'time' : 0}

    def algorithm(self, signal_in):
        print('>>> algorithm')
        print(type(signal_in))
        print(signal_in.shape)
        
        params = self._params
        number_to_add = params['number_to_add']
        signal_values = signal_in.values
        
        result = signal_values + number_to_add
        print(type(result))
        print(result.shape)
        print('<<< algorithm')
        return result

signal_ = Simple1d()(signal)

#%% SIMPLE INDICATOR
class Mean(VerboseAlgorithm):
    """
    Simple algorithm that computes an indicator (mean) on 1d signals
    
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1} #output size is 1

    def algorithm(self, signal_in):
        # print('>>> algorithm')
        # print(type(signal_in))
        # print(signal_in.shape)
        signal_values = signal_in.values
        
        result = np.mean(signal_values, keepdims = True) #see the use of keepdims!
        # print(type(result))
        # print(result.shape)
        # print('<<< algorithm')
        return result

signal_ = Mean()(signal)

#%% Signal processing across components/channels
class AverageComponents(VerboseAlgorithm):
    """
    Compute the average across components
    
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 0,
                           'component': 1} #average across components reduces the number of components to 1

    def algorithm(self, signal_in):
        print('>>> algorithm')
        print(type(signal_in))
        print(signal_in.shape)
        signal_values = signal_in.values
        
        result = np.mean(signal_values, axis = 2, keepdims=True)
        print(type(result))
        print(result.shape)
        print('<<< algorithm')
        return result

signal_ = AverageComponents()(signal)

#%%
class AverageComponentsClone(VerboseAlgorithm):
    """
    Compute the average across components, and
    replicate/clone the result to obtain 
    the same number of components as the input
    
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 0,
                           'component': 0} #average across components reduces the number of components to 1

    def algorithm(self, signal):
        signal_values = signal.values
        
        n_components = signal.values.shape[2]
        
        mean_components = np.mean(signal_values, axis = 2, keepdims=True)
        
        #cloning the average to have the same number of components as the input
        output = np.repeat(mean_components, n_components, axis=2)
        print(output.shape)
        return output

signal_ = AverageComponentsClone()(signal)

#%%
class AverageComponentsChannels(VerboseAlgorithm):
    """
    Compute the timepoint-by-timepoint average across channelsXcomponents
    
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'channel' : 1,
                           'component': 1} #average across components reduces the number of components to 1

    def algorithm(self, signal):
        signal_values = signal.values
        # print(signal_values.shape)
        
        mean_components = np.mean(signal_values, keepdims=True)
        # print(mean_components.shape)
        return mean_components

signal_ = AverageComponentsChannels()(signal)

#%%
class MeanComponents(_Algorithm):
    """
    Compute the mean of the signal, across all components
    
    """

    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time' : 1,
                           'component': 1} 

    def algorithm(self, signal):
        signal_values = signal.values
        # print(signal_values.shape)
        
        mean_components = np.mean(signal_values, keepdims=True)
        # print(mean_components.shape)
        return mean_components

signal_ = MeanComponents()(signal)

#%%