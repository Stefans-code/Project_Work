# %reload_ext autoreload
# %autoreload 2

import numpy as np
from pyphysio.signal import create_signal
import xarray as xr
# import matplotlib.pyplot as plt
# plt.ion()    

sizes = [1000, (1000), (1000,2), (1000,2,3)]

sampling_freqs = [0.01, 1, 1./3]

for size in sizes:
    for sampling_freq in sampling_freqs:
        data = np.random.uniform(size = size)
        signal = create_signal(data, sampling_freq=sampling_freq)
        # plt.figure()
        signal.p.plot()
        # plt.show()

sampling_freq = 10
size = 1000

data = np.random.uniform(size = size)
signal = create_signal(data, sampling_freq=sampling_freq)
signal.p.plot()

data = np.ones(shape = 1000)
signal = create_signal(data, sampling_freq=sampling_freq)
signal.p.plot()

data = np.random.uniform(size = (1000, 2, 3))
signal = create_signal(data, sampling_freq=sampling_freq)
signal.p.plot()

data = np.array([10, 20, 30])
signal = create_signal(data, times = [10,20,30])
signal.p.plot('|', color = 'r')
# plt.show()


data = np.random.uniform(size = 1000)
signal = create_signal(data, sampling_freq=10)
signal = signal.p.segment_time(10, 20)
assert signal.p.get_start_time() == 10
# plt.show()

data = np.random.uniform(size = (1000, 2))

data[20:50, 0] = np.nan
data[100:200, 1] = np.nan
signal = create_signal(data, sampling_freq=10)
signal_ = signal.p.process_na(na_action='impute')
# plt.show()


#%%
# print(signal.p.get_sampling_freq())
# data.p.plot()
