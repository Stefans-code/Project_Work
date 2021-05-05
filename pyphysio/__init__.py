# coding=utf-8
# from __future__ import division

from numpy import array as _array
from .tools.Tools import *
import numpy as _np
from .filters import Filters
from .segmentation import SegmentsGenerators
from .indicators import FrequencyDomain
from .indicators import NonLinearDomain
from .indicators import PeaksDescription
from .indicators import TimeDomain
from .BaseSegmentation import Segment
from .Signal import EvenlySignal, UnevenlySignal, from_pickle, from_pickleable
from .interactive import Annotate
# BE CAREFUL with NAMES!!!
from .estimators.Estimators import *
from .filters.Filters import *

from .sqi.SignalQuality import *
#from .tests import TestData
from .segmentation.SegmentsGenerators import *
from .Signal import Signal

#TODO: all signals as N_SAMPLES x N_CH, with N_CH =1 for non MultiEvenly

print("Please cite:")
print("Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', SoftwareX")

__author__ = "AleB"

def nature2type(data):
    data.ph['signal_type'] = data.ph['signal_nature']
    #
#    if isinstance(data, Signal):
#        stim = data.get_stim()
#        stim.ph['signal_type'] = stim.ph['signal_nature']
#        data.set_stim(stim)
    return(data)


def algo(function, **kwargs):
    """
    Builds on the fly a new algorithm class using the passed function and params if passed.
    :param function: function(data, params) to be called
    :param kwargs: parameters to pass to the function.
    :return: An algorithm class if params is None else a parametrized algorithm instance.
    """

    from .BaseAlgorithm import Algorithm

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


def test():
    from pytest import main as m
    from os.path import dirname as d
    m(['-x', d(__file__)])
