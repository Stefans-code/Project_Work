import numpy as np
from pyphysio.loaders import load_nirx2
from pyphysio.specialized.fnirs import SignalQualityDeepLearning
from pyphysio.segmenters import FixedSegments, fmap
import pyphysio.artefacts as artefacts
from pyphysio.specialized.fnirs import Raw2Oxy, NegativeCorrelationFilter, SDto1darray
import pyphysio.filters as filters

import matplotlib.pyplot as plt

#%% load data
DATA_FOLDER = '/home/bizzego/UniTn/data/fnirs_technical_validation/hyper/pilot'
nirs = load_nirx2(f'{DATA_FOLDER}/TN001/TN001_base/tn001fa_001') 

#%%
signal_values = nirs.p.main_signal.values[:,1,0]
iqr = 1.5
import numpy as _np
import pywt
from copy import deepcopy as copy

#%% remove nans, if present
if np.sum(np.isnan(nirs.p.main_signal.values)) > 0:
    nirs = nirs.process_na('impute')
    
plt.figure()
nirs.p.plot(sharey=False)

#%% SQI using Deep Learning
ratio_min_good = 0.8 #at least 80% of the windows should have a good value

segmenter = FixedSegments(10, 20)
indicators = [SignalQualityDeepLearning()]

sqi = fmap(segmenter, indicators, nirs)
sqi = sqi.drop_vars(['component_start', 'component_stop', 'label'])
sqi = sqi.sel(component = [0])

# select channels
isgood_vals = sqi['nirs_SignalQualityDeepLearning_isgood'].values[:,:,0]
isgood_ratios = np.sum(isgood_vals, axis=0)/isgood_vals.shape[0]
id_good_channels = np.where(isgood_ratios >= ratio_min_good)[0]

print(" good channels: ", id_good_channels)

sqi.p.plot()

#%% preprocessing

#remove MA with splines
nirs_noMA = artefacts.MARA()(nirs)

plt.figure()
nirs.p.plot(sharey=False)
nirs_noMA.p.plot(sharey=False)

#remove MA with wavelet
nirs_wav = artefacts.WaveletFilter()(nirs)

plt.figure()
nirs_noMA.p.plot(sharey=False)
nirs_wav.p.plot(sharey=False)

#%% convert to hb
hb = Raw2Oxy(age=21)(nirs_wav)

plt.figure()
hb.p.plot(sharey=False)

#%% filtering

#frequency band / moving average
# hb_f = filters.IIRFilter(fp = [0.01, 0.2], order=3, ftype='ellip')(hb)
# hb_f = filters.ConvolutionalFilter('rect', win_len=1)(hb)
order = 30

hb_f = filters.FIRFilter(fp = [0.01, 0.5], order=order, btype='bandpass')(hb)
# 
plt.figure()
# plt.plot(hb.p.main_signal.values[:,0,0])
# plt.plot(hb_f.p.main_signal.values[:,0,0])
hb.p.plot(sharey=False)
hb_f.p.plot(sharey=False)
# 
#%%
#negative correlation filter
hb_nc = NegativeCorrelationFilter()(hb_f)

plt.figure()
hb_f.p.plot(sharey=False)
hb_nc.p.plot(sharey=False)

hb_nc.attrs['good_channels'] = id_good_channels

#%% save
OUT_FOLDER = '/home/bizzego/tmp'

hb_nc = SDto1darray(hb_nc)
hb_nc.to_netcdf(f'{OUT_FOLDER}/hb')
