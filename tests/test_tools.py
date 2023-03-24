import numpy as np
from pyphysio.signal import create_signal
import xarray as xr

import pyphysio.utils as utils

#%%
# tools = [tool.PSD('welch'),
#          tool.Diff(),
#          tool.Maxima(),
#          tool.Minima()]
         

# def test_tools(signal):
#     assert isinstance(signal, xr.Dataset)
#     assert isinstance(signal.p.main_signal, xr.DataArray)
    
#     for t in tools:
#         print(t)
#         result = t(signal)
#         # print(f.__name__())
#         # assert result.p.main_signal.values.ndim == signal.p.get_values().ndim
        

# sizes = [1000, (1000), (1000,1), (1000,1,1),
#          (1000, 5), (1000, 5, 2)]

# sampling_freqs = [100]

# for size in sizes:
#     for sampling_freq in sampling_freqs:
#         data = np.random.uniform(size = size)
#         signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')
#         test_tools(signal)

#%%
def check_shape(signal, result, all_dims=True):
    assert len(result.p.get_values().shape) == len(signal.p.get_values().shape)
    
    for i in range(len(result.p.get_values().shape)):
        if i!=0 or all_dims:
            assert result.p.get_values().shape[i] == signal.p.get_values().shape[i]

#%% test diff        
sampling_freq = 100
data = np.ones((1000, 3, 2))
signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')

result = utils.Diff()(signal)
check_shape(signal, result)
assert result.p.get_values().sum() == 0

result = utils.Diff(degree = 3)(signal)
check_shape(signal, result)
assert result.p.get_values().sum() == 0

#%% test peakdetection
freqs = np.arange(0,10)
t = np.arange(0, 20, 0.05)
data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T

signal = create_signal(data, sampling_freq=20, name = 'random')

res = utils.PeakDetection(0.1)(signal)

for i in np.arange(1, len(freqs)):
    res_ch = res.sel(channel=i).dropna(dim = 'time')
    t_max = res_ch.p.main_signal.p.get_times()

    assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'
    
#%% test signalrange
ampl = np.arange(1,11)
t = np.arange(0, 20, 0.05)
data = np.array([A*np.sin(2*np.pi*t) for A in ampl]).T

signal = create_signal(data, sampling_freq=20, name = 'random')

res = utils.SignalRange(1, 0.5)(signal)

for i in np.arange(1, len(ampl)):
    res_ch = res.sel(channel=i).dropna(dim = 'time')
    assert abs(np.max(res_ch.p.main_signal.values) - 2*ampl[i]) < 0.001, print(i)

#%% test PSD
freqs = np.arange(0,10)
t = np.arange(0, 20, 0.05)
data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T

signal = create_signal(data, sampling_freq=20, name = 'random')


pwd = utils.PSD('welch')(signal)

for i in np.arange(1, len(freqs)):
    idx_max = np.argmax(pwd.p.get_values()[:,i])
    assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01

pwd = utils.PSD('welch')(signal)

for i in np.arange(1, len(freqs)):
    idx_max = np.argmax(pwd.p.get_values()[:,i])
    assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01

# pwd = tool.PSD('ar')(signal)

# for i in np.arange(1, len(freqs)):
#     idx_max = np.argmax(pwd.p.get_values()[:,i])
#     assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01

#%% test wavelet
fsamp = 10
freqs = np.arange(0,10)
t = np.arange(0, 20, 1/fsamp)
data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T

signal = create_signal(data, sampling_freq=fsamp, name = 'random')

wavelet = utils.Wavelet()(signal)

#TODO: create assert here

#%%
fsamp = 10
f = 0.5
t = np.arange(0, 20, 1/fsamp)
data = np.sin(2*np.pi*f*t)

signal = create_signal(data, sampling_freq=fsamp, name = 'random')

wavelet = utils.Wavelet(freqs = np.array([5,3,2,0.5]))(signal)

#TODO: create assert here
    
#%% test maxima
freqs = np.arange(0,5)
t = np.arange(0, 20, 0.05)
data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T

signal = create_signal(data, sampling_freq=20, name = 'random')

res = utils.Maxima()(signal)

for i in np.arange(1, len(freqs)):
    res_ch = res.sel(channel=i).dropna(dim = 'time')
    t_max = res_ch.p.main_signal.p.get_times()

    assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'

#%% test minima
freqs = np.arange(0,5)
t = np.arange(0, 20, 0.05)
data = -1*np.array([np.sin(2*np.pi*x*t) for x in freqs]).T

signal = create_signal(data, sampling_freq=20, name = 'random')

res = utils.Minima()(signal)

for i in np.arange(1, len(freqs)):
    res_ch = res.sel(channel=i).dropna(dim = 'time')
    t_max = res_ch.p.main_signal.p.get_times()

    assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'   