import numpy as np
from pyphysio.signal import create_signal
import pyphysio.utils as utils
import matplotlib.pyplot as plt

# sampling_freq = 100
# size = 1000
# data = np.random.uniform(size = size)
# signal = create_signal(data, sampling_freq=sampling_freq)
# pwd = utils.Wavelet()(signal)
# assert pwd.p.get_values().ndim == signal.p.get_values().ndim + 1


#%% test wavelet
size = (1000,2)

#PCA
data = np.random.uniform(size = size)
signal = create_signal(data, sampling_freq=100)

wavelet = utils.Wavelet()
W = wavelet(signal)

freqs_w = W.freq.values
coi = wavelet._compute_coi(W)

#%%

size = (1000,5,6,7)

#PCA
data = np.random.uniform(size = size)
signal = create_signal(data, sampling_freq=100)

pca = utils.PCA(dimension='dimension_4')
result = pca(signal)

#PeakDetection
freqs = np.arange(0, 10)
t = np.arange(0, 20, 0.05)
data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
signal = create_signal(data, sampling_freq=20)
peaks = utils.PeakDetection(0.1, return_peaks=True)(signal)

for i in np.arange(1, len(freqs)):
    peaks_ch = peaks.sel(channel=i).dropna(dim = 'time')
    t_max = peaks_ch.p.get_times()
    assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'

#PeakSelection
signal['peaks'] = peaks
ps = utils.PeakSelection(win_pre=1, win_post=1)(signal)

signal = signal.drop_vars('peaks')
#Maxima
maxx = utils.Maxima()(signal)
for i in np.arange(1, len(freqs)):
    res_ch = maxx.sel(channel=i).dropna(dim = 'time')
    t_max = res_ch.p.get_times()
    assert len(t_max) == int(20*freqs[i]), f'{len(t_max)}, {freqs[i]}, {i}'

#PSD
psd = utils.PSD('welch')
pwd = psd(signal)
for i in np.arange(1, len(freqs)):
    idx_max = np.argmax(pwd.p.get_values()[:,i])
    pwd.coords['freq'].values[idx_max]
    assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01

pwd = utils.PSD('period')(signal)

for i in np.arange(1, len(freqs)):
    idx_max = np.argmax(pwd.p.get_values()[:,i])
    assert abs((pwd.coords['freq'].values[idx_max] - i)) < 0.01


#%% test wavelet
wavelet = utils.Wavelet()
W = wavelet(signal)

freqs_w = W.freq.values
coi = wavelet._compute_coi(W)

for i in np.arange(1, len(freqs)):
    www = abs(coi.sel({'channel':[i]}).p.get_values()[:,0,:])
    www_avg = np.nanmean(www, axis=0)
    www_avg = www_avg[:45]
    idx_max = np.nanargmax(www_avg)
    
    assert abs(idx_max - np.argmin(abs(freqs_w - i))) < 2, i

#SignalRange
ampl = np.arange(1,11)
t = np.arange(0, 20, 0.05)
data = np.array([A*np.sin(2*np.pi*t) for A in ampl]).T

signal = create_signal(data, sampling_freq=20)

sigrange = utils.SignalRange(1, 0.5)(signal)

for i in np.arange(1, len(ampl)):
    sigrange_ch = sigrange.sel(channel=i).dropna(dim = 'time')
    assert abs(np.max(sigrange_ch.p.get_values()) - 2*ampl[i]) < 0.001, print(i)


#%% try sizes
sizes = [1000, (1000), (1000,1), (1000,1,1),
         (1000, 5), (1000, 5, 2), (1000, 2,3,5)]
         
for size in sizes:
    
    for sampling_freq in [100]:
        data = np.random.uniform(size = size)
        signal = create_signal(data, sampling_freq=sampling_freq)
        pwd = utils.PSD('welch')(signal)
        assert pwd.p.get_values().ndim == signal.p.get_values().ndim + 1
        diff = utils.Diff()(signal)
        assert diff.p.get_values().ndim == signal.p.get_values().ndim
        maxx = utils.Maxima()(signal)
        assert maxx.p.get_values().ndim == signal.p.get_values().ndim
        peaks = utils.PeakDetection(0.1, return_peaks=True)(signal)
        assert peaks.p.get_values().ndim == signal.p.get_values().ndim
        sigrange = utils.SignalRange(1, 0.5)(signal)
        assert sigrange.p.get_values().ndim == signal.p.get_values().ndim
        pwd = utils.Wavelet()(signal)
        assert pwd.p.get_values().ndim == signal.p.get_values().ndim + 1
