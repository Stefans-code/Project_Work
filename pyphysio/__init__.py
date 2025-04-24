import numpy as _np
import xarray as _xr
import os as _os
from .signal import create_signal, load
from .loaders import load_xrnirs

_xr.set_options(keep_attrs=True)

try:
    from dask import __name__ as _
    scheduler = 'threads'
    # available schedulers:
    # #distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads
    print('Using dask. Scheduler: threads')
except:
    scheduler = 'single-threaded'
    
print("Please cite:")
print("Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', SoftwareX")


#namespace
from .signal import *


class TestData(object):
    _sing = None
    _path = _os.path.join(_os.path.dirname(__file__), 'test_data')
    _file = "medical.txt.bz2"
    _fsamp_medical = 2048

    # The following methods return an array to make it easier to test the Signal wrapping classes

    @classmethod
    def get_data(cls):
        if TestData._sing is None:
            TestData._sing = _np.genfromtxt(_os.path.abspath(_os.path.join(TestData._path, TestData._file)), delimiter="\t")
        return TestData._sing

    @classmethod
    def ecg(cls, return_signal=False):
        values = TestData.get_data()[:, 0]
        if return_signal:
            signal = create_signal(values, sampling_freq=cls._fsamp_medical)
            return signal
        else:
            return values

    @classmethod
    def eda(cls, return_signal=False):
        values = TestData.get_data()[:, 1]
        if return_signal:
            signal = create_signal(values, sampling_freq=cls._fsamp_medical)
            return signal
        else:
            return values

    @classmethod
    def bvp(cls, return_signal=False):
        values = TestData.get_data()[:, 2]
        if return_signal:
            signal = create_signal(values, sampling_freq=cls._fsamp_medical)
            return signal
        else:
            return values
    

    @classmethod
    def resp(cls, return_signal=False):
        values = TestData.get_data()[:, 3]
        if return_signal:
            signal = create_signal(values, sampling_freq=cls._fsamp_medical)
            return signal
        else:
            return values
    
    @classmethod
    def fnirs(cls, return_signal=False):
        signal = load_xrnirs(_os.path.abspath(_os.path.join(TestData._path, "fnirs.ph")))
        if return_signal:
            return signal
        else:
            return signal.values
        
    @classmethod
    def tapping(cls, return_signal=False):
        signal = load_xrnirs(_os.path.abspath(_os.path.join(TestData._path, "fnirs_tapping.ph")))
        if return_signal:
            return signal
        else:
            return signal.values
    
        
def test():
    from pytest import main as m
    from os.path import dirname as d
    m(['-x', d(__file__)])