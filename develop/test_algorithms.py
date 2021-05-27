# import packages
import numpy as np
import matplotlib.pyplot as plt
import pyphysio as ph

# create two signals
fsamp = 10

signal = ph.EvenlySignal(np.random.uniform(size=(10)), fsamp)

#%%
signal_ = ph.Mean()(signal)

#print(signal_.shape)

#%%
signal_ = ph.Normalize()(signal)

#%%
signal.plot()
signal_.plot()
print(signal_.shape)

#%%
signal_ = ph.SignalRange(win_len=1, win_step=0.5, smooth=True)(signal)
print(signal_.shape)

plt.plot(signal[:,0, 2])
plt.plot(signal_[:,0, 2])

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

# ecg = ecg[:10000]

# In[ ]:
# create two signals
# fsamp = 10

# signal = ph.EvenlySignal(np.random.uniform(size=(10)), fsamp)
ibi = ph.BeatFromECG()(ecg)




# In[3]:


# apply a Filter
ecg_filtered = ph.Normalize()(ecg)


# In[4]:


ecg_filtered.shape


# In[ ]:


ecg.plot()


# In[ ]:


ecg_filtered.shape


# In[ ]:


#plot
ecg.plot()
ecg_filtered.plot()


# ### 2.2 Estimators
# Estimators are algorithms which aim at extracting the information of interest from the input signal, thus returning a new signal which has a different **`signal_nature`**. 
# 
# The name *`Estimators`* recalls the fact that the information extraction depends on the value of the algorithm parameters which might not be known *a-priori*. Thus the result should be considered as an estimate of the real content of information of the input signal.

# In[ ]:


# create an Estimator
import pyphysio.processing.estimators as est

ibi_ecg = est.BeatFromECG()


# In[ ]:


# check parameters
ibi_ecg


# In[ ]:


# apply an Estimator
ibi = ibi_ecg(ecg)


# In[ ]:


# plot
ax1 = plt.subplot(211)
ecg.plot()

plt.subplot(212, sharex=ax1)
ibi.plot()


# ### 2.3 Indicators
# 
# Indicators are algorithm which extract a metrics (scalar value) from the input signal, for instance a statistic (average).
# 
# Three types of indicators are provided in **`pyphysio`**:
# * Time domain indicators: comprising simple statistical indicators and other metrics that can be computed on the signal values;
# * Frequency domain indicators: metrics that are computed on the Power Spectrum Density (PSD) of the signal;
# * Non-linear indicators: complex indicators that are computed on the signal values (e.g. Entropy).

# In[ ]:


# create an Indicator
import pyphysio.indicators.timedomain as td_ind
import pyphysio.indicators.frequencydomain as fd_ind


# In[ ]:


rmssd = td_ind.RMSSD()
HF = fd_ind.PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15, method = 'ar')


# In[ ]:


# check parameters
print(rmssd)
print(HF)


# In[ ]:


# apply an Indicator
rmssd_ = rmssd(ibi)
HF_ = HF(ibi.resample(4)) #resampling is needed to compute the Power Spectrum Density

print(rmssd_)
print(HF_)


# In[ ]:


# check output type
print(type(rmssd_))
print(type(HF_))


# ### 2.4 Signal Quality Indicators
# 
# _TODO_

# ### 2.5 Tools
# 
# This is a collection of useful algorithms that can be used for signal processing. 
# 
# These algorithms might return scalar values or numpy arrays.

# In[ ]:


# create a Tool
import pyphysio.processing.tools as tll

compute_psd = tll.PSD(method='ar', interp_freq = 4)


# In[ ]:


# check parameters
compute_psd


# In[ ]:


# apply a Tool
frequencies, power = compute_psd(ibi.resample(4))

plt.plot(frequencies, power)
plt.show()


# In[ ]:




