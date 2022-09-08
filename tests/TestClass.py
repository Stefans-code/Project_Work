import os as _os
import numpy as _np

class TestData(object):
    _sing = None
    _path = _os.path.join(_os.path.dirname(__file__), '..', 'test_data')
    _file = "medical.txt.bz2"

    @classmethod
    def get_data(cls):
        if TestData._sing is None:
            TestData._sing = _np.genfromtxt(_os.path.abspath(_os.path.join(TestData._path, TestData._file)), delimiter="\t")
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
