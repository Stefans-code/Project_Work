# coding=utf-8
# from __future__ import division

# from numpy import array as _array
# from .tools.Tools import *
import numpy as _np
import os as _os
from .processing.tools import *
from .processing.filters import *
from .processing.estimators import *

from .indicators.timedomain import *
from .indicators.frequencydomain import *
from .indicators.peaks import *
from .indicators.nonlinear import *

from .segmenters import *
from .signal import *
from .interactive import Annotate

from .sqi import *
from .sqi.sqi import *
#from .tests import TestData

print("Please cite:")
print("Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', SoftwareX")

# __author__ = "AleB"
    
def update_signal(signal):
    signal_type = signal.ph['signal_type']
    info = {'signal_type': signal_type}
    signal.ph['info'] = info
    return(signal)

class TestData(object):
    _sing = None
    _path = _os.path.join(_os.path.dirname(__file__), '..', 'test', "data")
    _file = "medical.txt.bz2"

    @classmethod
    def get_data(cls):
        if TestData._sing is None:
            TestData._sing = _np.genfromtxt(_os.path.join(TestData._path, TestData._file), delimiter="\t")
        return TestData._sing

    # The following methods return an array to make it easier to test the Signal wrapping classes

    @classmethod
    def ecg(cls):
        return TestData.get_data()[:, 0]

    @classmethod
    def eda(cls):
        return TestData.get_data()[:, 1]

    @classmethod
    def bvp(cls):
        return TestData.get_data()[:, 2]

    @classmethod
    def resp(cls):
        return TestData.get_data()[:, 3]
    
def test():
    from pytest import main as m
    from os.path import dirname as d
    m(['-x', d(__file__)])
