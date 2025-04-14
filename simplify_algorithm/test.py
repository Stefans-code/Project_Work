from template_algorithms import NoRolling, SimpleFilter, SimpleIndicator, AddDimension, \
    ShrinkDimension, WhateverDimension, SimpleSQIIndicator, AlgorithmUsingSupportingSignal
    
import pyphysio as ph
import numpy as _np
import xarray as _xr

signal = ph.create_signal(_np.ones(shape = (1000, 5, 2)),
                          sampling_freq=4,
                          start_time=100,
                          name='signal')
# print(signal)

#%%
no_rolling = NoRolling()
signal_out = no_rolling(signal)
print(signal_out)

#%%
simple_filter = SimpleFilter()
signal_out = simple_filter(signal)
# print(signal_out)

#%%
simple_indicator = SimpleIndicator()
signal_out = simple_indicator(signal)
print(signal_out)

#%%
simple_sqi_indicator = SimpleSQIIndicator(threshold = [0, 10])
signal_out = simple_sqi_indicator(signal)
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

supp_algorithm = AlgorithmUsingSupportingSignal()
signal_in = signal.copy(deep=True)
signal_in['new_coord'] = signal
signal_out = supp_algorithm(signal_in)
print(signal_out)