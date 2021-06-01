# import packages
import numpy as np
import matplotlib.pyplot as plt
import pyphysio as ph
from test_utils import info
# create two signals
fsamp = 10

signal = ph.EvenlySignal(np.random.uniform(size=(100,2)), fsamp)

#%%
signal_ = ph.Mean()(signal)

print(signal_.shape)
print(type(signal_))

#%%
signal_ = ph.Normalize(norm_method='custom')(signal)
info(signal_)

#%%
signal.plot()
signal_.plot()
print(signal_.shape)

#%%
signal_ = ph.SignalRange(win_len=1, win_step=0.5, smooth=True)(signal)
print(signal_.shape)

plt.plot(signal[:,0])
plt.plot(signal_[:,0])

#%%
signal_vals = np.stack([np.sin(2*np.pi*x*np.arange(0, 10, 0.05)) for x in np.arange(10)], axis=1)

signal = ph.EvenlySignal(signal_vals, 20)
freq, pwd = ph.PSD('fft')(signal)
plt.plot(freq[:,0], pwd)

#%%
import pyphysio as ph
import numpy as np
from pyphysio import TestData
from pyphysio import EvenlySignal

ecg_data = TestData.ecg()
# eda_data = TestData.eda()

# create two signals
fsamp = 2048
tstart_ecg = 15
# tstart_eda = 5

ecg_ch = np.stack([ecg_data, 1+ecg_data, 2+ecg_data], 1)
ecg_comp = np.stack([ecg_ch, 2*ecg_ch], 2)

ecg = EvenlySignal(values = ecg_comp, 
                   sampling_freq = fsamp, 
                   start_time = tstart_ecg)

ecg_filtered = ph.Normalize()(ecg)


#%%
info(ecg_filtered)

#%%
ecg.plot()
ecg_filtered.plot()

#%%
ecg = EvenlySignal(values = ecg_data, 
                   sampling_freq = fsamp, 
                   start_time = tstart_ecg)

ibi_ecg = ph.BeatFromECG()
ibi = ibi_ecg(ecg)

#%%
rmssd = ph.RMSSD()
HF = ph.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15, method = 'ar')

rmssd_ = rmssd(ibi)
HF_ = HF(ibi.resample(4)) #resampling is needed to compute the Power Spectrum Density

print(rmssd_)
print(HF_)