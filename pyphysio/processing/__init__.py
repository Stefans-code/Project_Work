from ..signal import Signal as _Signal
import numpy as _np

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
        values_out = _np.apply_along_axis(self.algorithm, 0, data)
        
        ndims_out = values_out.ndim
        shape_out = values_out.shape
        
        if ndims_in != ndims_out:
            #TODO: check that shape is the same (except axis 0)
            values_out = _np.expand_dims(values_out, 0)
        return(values_out)

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