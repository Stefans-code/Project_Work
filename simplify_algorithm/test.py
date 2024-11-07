from template_algorithms import NoRolling, SimpleFilter, SimpleIndicator, AddDimension, \
    ShrinkDimension, WhateverDimension, Wavelet
    
import pyphysio as ph
import numpy as _np
import xarray as _xr

signal = ph.create_signal(_np.ones(shape = (1000, 9, 2)),
                          sampling_freq=4,
                          start_time=100,
                          name='signal')
print(signal)

#%%
no_rolling = NoRolling()
signal_out = no_rolling(signal)
print(signal_out)

#%%
simple_filter = SimpleFilter()
signal_out = simple_filter(signal)
print(signal_out)

#%%
simple_indicator = SimpleIndicator()
signal_out = simple_indicator(signal)
print(signal_out)

#%%
add_dimension = AddDimension(size_new_dim=10)
signal_out = add_dimension(signal)
print(signal_out)

#%%
shrink_dimension = ShrinkDimension(n_channels_out=5)
signal_out = shrink_dimension(signal)
print(signal_out)

#%%
whatever_dimension = WhateverDimension(n_components_out=10, n_newdim_out=4)
signal_out = whatever_dimension(signal)
print(signal_out)

#%%
wavelet = Wavelet()
_, wavelet_template = wavelet.__get_template__(signal.p.main_signal)
signal_out = wavelet(signal)
print(signal_out)

#%%
target_freqs = _np.arange(0.01, 0.21, 0.01)[::-1]
wavelet = Wavelet(freqs=target_freqs)
_, wavelet_template = wavelet.__get_template__(signal.p.main_signal)
signal_out = wavelet(signal)
print(signal_out)

#%%
from pyphysio.loaders import load_xrnirs

# target_freqs = np.around(np.arange(0.01, 0.21, 0.01)[::-1], decimals=2)
nirs_A_wav = load_xrnirs('/home/bizzego/tmp/A_3min_wav_prew')
nirs_A_wav = nirs_A_wav.isel({'channel': [0], 'component': [0]})

nirs_A_wav = nirs_A_wav['signal_WaveletFilter_Raw2Oxy_Prewhitening']
ww = Wavelet(freqs=target_freqs)
W_A = ww(nirs_A_wav)
