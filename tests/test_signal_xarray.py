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

#%%
# def normalize(x):
#     return (x- np.mean(x))/np.std(x)

data = np.random.uniform(size = (1000, 10,5))

# s = xr.DataArray(data, dims = ['time', 'channels', 'components'],
#                  coords = {'time': np.arange(1000)}, 
#                  name = 'signal').to_dataset()


#%%
sampling_freq = 7.81
s = create_signal(data, sampling_freq=sampling_freq)

print(s.signal.shape)
print(s.p.get_values().shape)
print(s.p.get_times().shape)
print(s.p.get_start_time())
print(s.p.get_end_time())
print(s.p.get_sampling_freq())
print(s.p.get_duration())
print(s.p.get_duration())
print(s.p.get_info())

#%% test segment_time
result = Normalize()(s)

#%%
data = np.random.uniform(size = (1000, 1,1))
sampling_freq = 7.81
s = create_signal(data, sampling_freq=sampling_freq)

s_ = s.p.resample(10)

s.p.plot()
s_.p.plot('.')

#%%
da = s.p.main_signal
da_ = da.p.resample(20)

da.p.plot()
da_.p.plot('.')