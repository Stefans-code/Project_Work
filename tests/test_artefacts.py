import numpy as np
from pyphysio.signal import create_signal
import xarray as xr

import pyphysio.artefacts as art

size = (1000, 5, 2, 2)

sampling_freq = 10

data = np.random.uniform(size = size)
data[500:, :2, :] = data[500:, :2, :] + 1
data[500:, 2:, 0] = data[500:, 2:, 0] + 1
signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')

#%%
MA_none = art.DetectMA(fuse=None)(signal)
MA_all = art.DetectMA(fuse='all')(signal)
MA_component = art.DetectMA(fuse='component')(signal)

#%%
signal_none = art.MARA(MA_none)(signal, scheduler='single-threaded')
signal_all = art.MARA(MA_all)(signal, scheduler='single-threaded')
signal_component = art.MARA(MA_component)(signal, scheduler='single-threaded')

#%%
signal_ = art.WaveletFilter()(signal)
