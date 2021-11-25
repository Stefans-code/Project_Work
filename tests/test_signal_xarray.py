#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:18:36 2021

@author: bizzego
"""
import numpy as np
from pyphysio.signal import create_signal
from pyphysio.processing.filters import Normalize

import xarray as xr

# def normalize(x):
#     return (x- np.mean(x))/np.std(x)

data = np.random.uniform(size = (1000, 10,5))

# s = xr.DataArray(data, dims = ['time', 'channels', 'components'],
#                  coords = {'time': np.arange(1000)}, 
#                  name = 'signal').to_dataset()


#%%
sampling_freq = 1000
s = create_signal(data, sampling_freq=sampling_freq)

print(s.signal.shape)
print(s.ph.get_values().shape)
print(s.ph.get_times().shape)
print(s.ph.get_start_time())
print(s.ph.get_end_time())
print(s.ph.get_sampling_freq())
print(s.ph.get_duration())
print(s.ph.has_multi_channels())
print(s.ph.get_nchannels())
print(s.ph.has_multi_components())
print(s.ph.get_ncomponents())
print(s.ph.get_duration())
print(s.ph.get_info())

#%% test segment_time
result = Normalize()(s)
