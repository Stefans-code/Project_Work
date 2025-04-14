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
signal['MA'] = MA_none
signal_none = art.MARA()(signal, scheduler='single-threaded')
signal = signal.drop_vars('MA')

MA_all = art.DetectMA(fuse='all')(signal)
signal['MA'] = MA_all
signal_all = art.MARA()(signal, scheduler='single-threaded')
signal = signal.drop_vars('MA')


MA_component = art.DetectMA(fuse='component')(signal)
signal['MA'] = MA_component
signal_component = art.MARA()(signal, scheduler='single-threaded')
signal = signal.drop_vars('MA')


#%%
signal_ = art.WaveletFilter()(signal)
