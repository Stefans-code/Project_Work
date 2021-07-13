# coding=utf-8
import pyphysio as ph
import numpy as np
np.random.seed(1234)

import pytest

@pytest.mark.parametrize("size", [(1000), (1000, 5), (1000, 5, 3)])
@pytest.mark.parametrize("f", [10, 0.1, 10.1])
@pytest.mark.parametrize("t", [-100, 0, 0.00001])
def test_simple_signal(size, f, t):
    if isinstance(size, int):
        size = (size,)
        
    values = np.random.uniform(size=size)
    s = ph.Signal(values=values,
                  sampling_freq=f, 
                  start_time=t)
    
    assert s.ndim == len(size)
    assert s.ndim == values.ndim
    assert len(s) == size[0]
    
    if values.ndim > 1:
        assert s.get_nchannels() == values.shape[1], s.get_nchannels()
        
    if values.ndim > 2:
        assert s.get_ncomponents() == values.shape[2], s.get_ncomponents()
        
    assert s.get_sampling_freq() == f
    assert s.get_start_time() == t

    nsampl = len(s)
    
    assert s.get_end_time() == t + nsampl/f
    
    indices = s.get_indices()
    values = s.get_values()
    assert len(indices) == len(values)

    t = s.get_times()
    assert len(s.get_values()) == len(t)  # length
    assert len(np.where(np.diff(t) <= 0)[0]) == 0  # strong monotonicity

    # pickleability
    ps = ph.from_pickleable(s.pickleable)
    assert ps.get_start_time() == s.get_start_time()
    assert ps.get_sampling_freq() == s.get_sampling_freq()
    assert ps.get_end_time() == s.get_end_time()
    assert len(ps) == len(s)
    
    s.set_start_time(12)
    assert s.get_start_time() == 12
    
    s.set_sampling_freq(12)
    assert s.get_sampling_freq() == 12   
    
    #clone_properties
    s_ = s.clone_properties(values*0.5)
    assert s.ndim == s_.ndim
    assert len(s) == len(s_)
    
    assert s.get_sampling_freq() == s_.get_sampling_freq()
    assert s.get_start_time() == s_.get_start_time()
    
    