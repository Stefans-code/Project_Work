import pyphysio as ph
import numpy as np

def generate_evenly(shape=(1000, 5, 3), fsamp=100, starttime=0):
    signal = ph.Signal(np.random.uniform(size=shape), fsamp, starttime)
    if isinstance(shape, int):
        shape = [shape]
    check(signal, shape, fsamp, starttime)
    return(signal)
    
# TODO-AI [PRIORITY: MEDIUM]: Make this helper deterministic and expose as
# a pytest fixture in `conftest.py`.
# - Use a `rng` fixture or accept a `seed` parameter instead of global RNG.

def check(signal, shape=(1000, 5, 3), fsamp=100, starttime=0):
    assert len(signal) == shape[0]
    assert signal.get_start_time() == starttime
    assert signal.get_end_time() == starttime + len(signal)/fsamp
    assert signal.time2idx(starttime) == 0
    assert signal.idx2time(0) == starttime

def info(s):
    print(type(s))
    print('shape:', s.shape)
    print('ph:', s.ph)
    print('values:', s.get_values()[:1])
    print('values shape:', s.get_values().shape)
    
    print('times:', s.get_times()[:3], '...', s.get_times()[-3:], ',  ', s.get_duration())
    
    print('indices:', s.get_indices()[:3], '...', s.get_indices()[-3:])
    
    print('is 1-dim - chan \ comp', s.is_onedim(), s.get_nchannels(), s.get_ncomponents())
    
    