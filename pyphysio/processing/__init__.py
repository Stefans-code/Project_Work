from ..signal import Signal as _Signal
import numpy as _np


def apply_on_signals(alg, signal):
    print(alg)
    print(type(signal))
    
    signal_out = []
    
    for i_component in range(signal.get_ncomponents()):
        component_out = []
        
        for i_channel in range(signal.get_nchannels()):
            channel_data = signal[:, i_channel, i_component]
            print(channel_data.shape)
            channel_out = alg(channel_data)
            if channel_out.ndim == 0:
                print('dims 0')
                #result is a scalar
                channel_out = _np.array([channel_out]).reshape(1,1,1)
            print(channel_out.shape)
            component_out.append(channel_out)
        
        component_out = _np.concatenate(component_out, axis = 1)
        print(component_out.shape)
        signal_out.append(component_out)
    
    signal_out = _np.concatenate(signal_out, axis = 2)
    # print(signal_out.shape)
    return(signal_out)
            
class Algorithm(object):
    """
    This is the algorithm container super class. It should be used only to be extended.
    """

    def __init__(self, **kwargs):
        """
        Incorporates the parameters and saves them in the instance.
        @param params: Dictionary of string-value parameters passed by the user.
        @type params: dict
        @param _kwargs: Internal channel for subclasses kwargs parameters.
        @type _kwargs: dict
        @param kwargs: kwargs parameters to pass to the feature extractor.
        @type kwargs: dict
        """
        self._params = {}
        self.set_params(**kwargs)  # already checked by __init__

    def __call__(self, data):
        """
        Executes the algorithm using the parameters saved by the constructor.
        @param data: The data.
        @type data: TimeSeries
        @return: The result.
        """
        
        assert isinstance(data, _Signal), "The data must be a Signal (see class EvenlySignal and UnevenlySignal)."
        
        #when applying non numpy methods (is it true?)
        #apply_along_axis generates arrays with shape (1,:,n_samples)
        #where the number of dimensions is the number of samples of the original signal
        #this would fuck up everithing and we need to manage this
        ndims_in = data.ndim
        shape_in = data.shape
        values_out = apply_on_signals(self.algorithm, data)
        
        ndims_out = values_out.ndim
        shape_out = values_out.shape
        
        if ndims_in != ndims_out:
            #TODO: check that shape is the same (except axis 0)
            values_out = _np.expand_dims(values_out, 0)
            
        signal_out = data.clone_properties(values_out)
        return(signal_out)

    def __repr__(self):
        return self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def set(self, **kwargs):
        kk = self.get()
        kk.update(kwargs)
        self.__init__(**kk)

    def get(self, param=None):
        """
        Placeholder for the subclasses
        @return
        """
        if param is None:
            return self._params
        else:
            return self._params[param]


    def algorithm(cls, data):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        pass

def algo(function, **kwargs):
    """
    Builds on the fly a new algorithm class using the passed function and params if passed.
    :param function: function(data, params) to be called
    :param kwargs: parameters to pass to the function.
    :return: An algorithm class if params is None else a parametrized algorithm instance.
    """

    class Custom(Algorithm):
        def __init__(self, **kwargs):
            Algorithm.__init__(self, **kwargs)

        def algorithm(self, signal):
            params = self._params
            return function(signal, params)

    if len(kwargs) == 0:
        return Custom
    else:
        return Custom(**kwargs)