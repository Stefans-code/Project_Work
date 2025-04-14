import numpy as np
from pyphysio.signal import create_signal
import pyphysio.utils as utils
import matplotlib.pyplot as plt

sizes = [1000, (1000), (1000,1), (1000,1,1),
         (1000, 5), (1000, 5, 2), (1000, 2,3,5)]

# size = (1000, 5,6,7)

# data = np.random.uniform(size = size)
# signal = create_signal(data, sampling_freq=100)

# pca = utils.PCA(dimension='dimension_4')
# result = pca(signal)

freqs = [0.2, 1.5]# np.arange(1,2, 0.5)
t = np.arange(0, 20, 0.05)
data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T

signal = create_signal(data, sampling_freq=20)

pd = utils.PeakDetection(0.1)(signal)

signal['peaks'] = pd

ps = utils.PeakSelection(win_pre=1, win_post=1)(signal)

signal.p.plot()
pd.p.plot('.')
ps.p.plot()
plt.show()
print(ps)

# maxx = utils.Minima()(signal)

# for i in np.arange(1, len(freqs)):
#     res_ch = maxx.sel(channel=i).dropna(dim = 'time')
#     t_max = res_ch.p.get_times()

#     assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'


# psd = utils.PSD('welch')

# pwd = psd(signal)

# for i in np.arange(1, len(freqs)):
#     idx_max = np.argmax(pwd.p.get_values()[:,i])
#     assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01

# pwd = utils.PSD('period')(signal)

# for i in np.arange(1, len(freqs)):
#     idx_max = np.argmax(pwd.p.get_values()[:,i])
#     assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01

# peaks = utils.PeakDetection(0.1, return_peaks=True)(signal)

# for i in np.arange(1, len(freqs)):
#     peaks_ch = peaks.sel(channel=i).dropna(dim = 'time')
#     t_max = peaks_ch.p.get_times()
#     assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'


#%% test wavelet
# wavelet = utils.Wavelet()
# W = wavelet(signal)

# freqs_w = W.freq.values
# coi = wavelet._compute_coi(W)

# for i in np.arange(1, len(freqs)):
#     www = abs(coi.sel({'channel':[i]}).p.get_values()[:,0,:])
#     www_avg = np.nanmean(www, axis=0)
#     www_avg = www_avg[:45]
#     idx_max = np.nanargmax(www_avg)
    
#     assert abs(idx_max - np.argmin(abs(freqs_w - i))) < 2, i

'''
ampl = np.arange(1,11)
t = np.arange(0, 20, 0.05)
data = np.array([A*np.sin(2*np.pi*t) for A in ampl]).T

signal = create_signal(data, sampling_freq=20)

sigrange = utils.SignalRange(1, 0.5)(signal)

for i in np.arange(1, len(ampl)):
    sigrange_ch = sigrange.sel(channel=i).dropna(dim = 'time')
    assert abs(np.max(sigrange_ch.p.get_values()) - 2*ampl[i]) < 0.001, print(i)


for size in sizes:
    for sampling_freq in [100]:
        data = np.random.uniform(size = size)
        signal = create_signal(data, sampling_freq=sampling_freq)
        pwd = utils.PSD('welch')(signal)
        assert pwd.p.get_values().ndim == signal.p.get_values().ndim + 1
        diff = utils.Diff()(signal)
        assert diff.p.get_values().ndim == signal.p.get_values().ndim
        peaks = utils.PeakDetection(0.1, return_peaks=True)(signal)
        assert peaks.p.get_values().ndim == signal.p.get_values().ndim
        sigrange = utils.SignalRange(1, 0.5)(signal)
        assert sigrange.p.get_values().ndim == signal.p.get_values().ndim


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

res = utils.PeakDetection(0.1, return_peaks=True)(signal)

for i in np.arange(1, len(freqs)):
    res_ch = res.sel(channel=i).dropna(dim = 'time')
    t_max = res_ch.p.main_signal.p.get_times()

    assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'
    
#%% test signalrange


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

wavelet = utils.Wavelet()
W = wavelet(signal)
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
    '''