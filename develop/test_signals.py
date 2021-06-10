import pyphysio as ph
import numpy as np
from test_utils import info_evenly, info_unevenly
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

#%
signal = ph.Signal(values, fsamp, start_time)
print(info_unevenly(signal))

#%%
sig_ = signal.resample(100,'cubic')

#%%
signal_ = signal[:100]
print(info_evenly(signal_))

#%%
signal_ = signal[:100,0]
print(info_evenly(signal_))

#%%
signal_ = signal[::2,0]
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
signal_ = signal[100, 3]
print(info_evenly(signal_))

#%%
import pyphysio as ph
import numpy as np
from test_utils import info_evenly, info_unevenly, info_ue
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

x_indices = np.arange(0, 2000, 2)

signal = ph.Signal(values, fsamp, start_time,
                   x_values = x_indices,
                   x_type = 'indices')

sig = signal.fill()

#%%
print(info_unevenly(signal))

#%%
signal_ = signal[:100]
print(info_ue(signal_))

#%%
signal_ = signal[:100,0]
print(info_ue(signal_))

#%%
signal_ = signal[:100,0,0]
print(info_ue(signal_))

#%%
signal_ = signal[100:480,0,0]
print(info_ue(signal_))

#%%
signal_ = signal.segment_time(55.3, 80)
print(info_unevenly(signal_))

#%%
signal_ = signal.segment_IDX(55, 80)
print(info_ue(signal_))