import pyphysio as ph
import numpy as np

def generate_evenly(shape=(1000, 5, 3), fsamp=100, starttime=0):
    signal = ph.EvenlySignal(np.random.uniform(size=shape), fsamp, starttime)
    if isinstance(shape, int):
        shape = [shape]
    check(signal, shape, fsamp, starttime)
    return(signal)

def check(signal, shape=(1000, 5, 3), fsamp=100, starttime=0):
    assert len(signal) == shape[0]
    assert signal.get_start_time() == starttime
    assert signal.get_end_time() == starttime + len(signal)/fsamp
    assert signal.time2idx(starttime) == 0
    assert signal.idx2time(0) == starttime


def info(s):
    print(type(s))
    print('shape\n', s.shape)
    print('dfsamp \n', s.get_sampling_freq())
    
    print('values \n', s.get_values()[:2])
    print('values shape\n', s.get_values().shape)

def info_evenly(s):
    print(type(s))
    print('shape\n', s.shape)
    print('dfsamp \n', s.get_sampling_freq())
    
    print('info \n', s.get_info())
    print('ph \n', s.ph)
    print('values \n', s.get_values()[:2])
    
    print('first 10 times  \n', s.get_times()[:10])
    print('last 10 times \n', s.get_times()[-10:])
    print('starttime \n', s.get_start_time())
    print('endtime \n', s.get_end_time())
    print('duration \n', s.get_duration())
    
    
    print('has chan \n', s.has_multi_channels())
    print('n chan \n', s.get_nchannels())
    print('has comp \n', s.has_multi_components())
    print('n comp \n', s.get_ncomponents())
    print('is 1-dim \n', s.is_onedim())
    print('has good \n', s.has_good())

def info_unevenly(s):
    print(type(s))
    print('shape\n', s.shape)
    print('dfsamp \n', s.get_sampling_freq())
    
    print('info \n', s.get_info())
    print('ph \n', s.ph)
    print('values \n', s.get_values()[:2])
    
    print('first 10 times  \n', s.get_times()[:10])
    print('last 10 times \n', s.get_times()[-10:])
    print('starttime \n', s.get_start_time())
    print('endtime \n', s.get_end_time())
    print('duration \n', s.get_duration())
    
    print('first 10 indices \n', s.get_indices()[:10])
    print('last 10 indices \n', s.get_indices()[-10:])
        
    print('has chan \n', s.has_multi_channels())
    print('n chan \n', s.get_nchannels())
    print('has comp \n', s.has_multi_components())
    print('n comp \n', s.get_ncomponents())
    print('is 1-dim \n', s.is_onedim())
    print('has good \n', s.has_good())
    