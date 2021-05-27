import pyphysio as ph
import numpy as np
from test_utils import info_evenly, info_unevenly
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

#%
signal = ph.EvenlySignal(values, fsamp, start_time)
print(info_evenly(signal))

#%%
signal_ = signal[:100]
print(info_evenly(signal_))

#%%
signal_ = signal[:100,0]
print(info_evenly(signal_))

#%%
signal_ = signal[:100,0,0]
print(info_evenly(signal_))

#%%
signal_ = signal[100:480,0,0]
print(info_evenly(signal_))

#%%
signal_ = signal[100,0,0]
print(type(signal_))

#%%
signal_ = signal[100]
print(info_evenly(signal_))

#%%
signal_ = signal[100, 3, :]
print(info_evenly(signal_))

#%%
import pyphysio as ph
import numpy as np
from test_utils import info_evenly, info_unevenly
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

x_indices = np.arange(1000)
x_values = x_indices/fsamp

signal = ph.UnevenlySignal(values, fsamp, start_time,
                           x_values = x_indices,
                           x_type = 'indices')
print(info_unevenly(signal))

#%%
signal_ = signal[:100]
print(info_unevenly(signal_))

#%%
signal_ = signal[:100,0]
print(info_unevenly(signal_))

#%%
signal_ = signal[:100,0,0]
print(info_evenly(signal_))

#%%
signal_ = signal[100:480,0,0]
print(info_evenly(signal_))

#%%
signal_ = signal.segment_time(55.3, 80)
print(info_evenly(signal_))

#%%
signal_ = signal.segment_idx(55, 80)
print(info_evenly(signal_))