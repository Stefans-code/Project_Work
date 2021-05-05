# coding=utf-8
# from abc import abstractmethod as _abstract, ABCMeta as _ABCMeta
from .Signal import Signal as _Signal
# from .Utility import PhUI as _PhUI #was pyphysio.Utility
import numpy as _np
# __author__ = 'AleB'


class Algorithm(object):
    """
    This is the algorithm container super class. It (is abstract) should be used only to be extended.
    """
    # __metaclass__ = _ABCMeta

    # _log = None

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
            
        values_out = _np.apply_along_axis(self.algorithm, 0, data)
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


    # @classmethod
    # @_abstract
    # def is_compatible(cls, signal):
    #     """
    #     Placeholder for the subclasses
    #     :returns: Weather nature is compatible or not
    #     @raise NotImplementedError: Ever
    #     """
    #     pass

    # @classmethod
    # @_abstract
    def algorithm(cls, data):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        pass