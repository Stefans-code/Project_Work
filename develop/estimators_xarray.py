# import packages
import numpy as np
import numpy as _np
import matplotlib.pyplot as plt
import pyphysio.processing.estimators as est

# import data from included examples
from pyphysio import TestData
from pyphysio import create_signal

%clear
#%
ecg_data = TestData.ecg()

fsamp = 2048
tstart_ecg = 15
ecg = create_signal(data = ecg_data, sampling_freq = fsamp, start_time = tstart_ecg)

#%
ibi_ecg = est.BeatFromECG()

ibi = ibi_ecg(ecg, scheduler='synchronous')

#%%


bpm_max = 120
delta = 0
k = 0.7

fmax = bpm_max / 60

import pyphysio.processing.tools as tools

delta = k * tools.SignalRange(win_len=2/fmax, win_step=0.5/fmax, smooth=False)(ecg)
delta = _np.array(delta.p.main_signal)

# print('delta')
#adjust for delta values equal to 0
idx_delta_zeros = _np.where(delta==0)[0]
idx_delta_nozeros = _np.where(delta>0)[0]
delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])

refractory = 1 / fmax

# print(signal.shape)
maxp = tools.PeakDetection(delta=delta.ravel(), refractory=refractory, start_max=True)(ecg)
maxp = _np.array(maxp.p.main_signal).ravel()
