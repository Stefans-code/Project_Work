import pyphysio as ph
import numpy as np
from test_utils import info
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

#%
signal = ph.Signal(values, fsamp, start_time)
info(signal)

#%%
sig_ = signal.resample(100,'cubic')
info(sig_)

#%%
signal_ = signal[:100]
info(signal_)

#%%
signal_ = signal[:100,0]
info(signal_)

#%%
signal_ = signal[::2,0]
info(signal_)

#%%
signal_ = signal[:100,0,0]
info(signal_)

#%%
signal_ = signal[100:480,0,0]
info(signal_)

#%%
signal_ = signal[100,0,0]
print(type(signal_))

#%%
signal_ = signal[100]
info(signal_)

#%%
signal_ = signal[100, 3]
info(signal_)

#%%
import pyphysio as ph
import numpy as np
from test_utils import info
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

x_indices = np.arange(0, 2000, 2)

signal = ph.Signal(values, fsamp, start_time,
                   x_values = x_indices,
                   x_type = 'indices')
info(signal)

#%%
sig = signal.fill()
info(sig)


#%%
signal_ = signal[:100]
info(signal_)

#%%
signal_ = signal[:100,0]
info(signal_)

#%%
signal_ = signal[:100,0,0]
info(signal_)

#%%
signal_ = signal[100:480,0,0]
info(signal_)

#%%
signal_ = signal.segment_time(55.3, 80)
info(signal_)

#%%
signal_ = signal.segment_IDX(0, 80)
info(signal_)