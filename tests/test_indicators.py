# import packages
import numpy as np
import matplotlib.pyplot as plt
from pyphysio import TestData
from pyphysio import create_signal

ecg_data = TestData.ecg()

# create two signals
fsamp = 2048
tstart_ecg = 15

ecg = create_signal(data = ecg_data, sampling_freq = fsamp, start_time = tstart_ecg)

#%%
import pyphysio.processing.estimators as est

ibi_ecg = est.BeatFromECG()
# apply an Estimator
ibi = ibi_ecg(ecg, add_signal=False)

#%%
import pyphysio.indicators.frequencydomain as fd_ind

ibi_ = ibi.dropna('time') #dropna is needed to correctly resample
ibi_ = ibi_.p.resample(4) #resampling is needed to compute the Power Spectrum Density


HF = fd_ind.PowerInBand(freq_min=0.15, freq_max=0.4, method = 'welch')
HF_ = HF(ibi_, add_signal=False) 

print(HF_.dropna('time'))

#%%
import pyphysio.indicators.timedomain as td_ind

rmssd = td_ind.RMSSD()
rmssd_ = rmssd(ibi.dropna('time'), add_signal=False)
print(rmssd_)