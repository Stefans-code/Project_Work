import numpy as np
from pyphysio.loaders import load_nirx2
from pyphysio.specialized.fnirs import SignalQualityDeepLearning
from pyphysio.segmenters import FixedSegments, fmap
import pyphysio.artefacts as artefacts
from pyphysio.specialized.fnirs import Raw2Oxy, NegativeCorrelationFilter, SDto1darray
import pyphysio.filters as filters

import matplotlib.pyplot as plt

ratio_min_good = 0.8 #at least 80% of the windows should have a good value

#%%
nirs = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/hyper/pilot/TN001/TN001_base/tn001fa_001') 

#%% remove nans
if np.sum(np.isnan(nirs.p.main_signal.values)) > 0:
    nirs = nirs.process_na('impute')
    
#%%
plt.figure()
nirs.p.plot(sharey=False)

#%% SQI using Deep Learning
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

#%% preprocessing
#remove MA with splines
nirs_noMA = artefacts.MARA()(nirs)

plt.figure()
nirs.p.plot(sharey=False)
nirs_noMA.p.plot(sharey=False)

#remove MA with wavelet
nirs_wav = artefacts.WaveletFilter()(nirs_noMA)

plt.figure()
nirs_noMA.p.plot(sharey=False)
nirs_wav.p.plot(sharey=False)

#%% convert to hb
hb = Raw2Oxy(age=21)(nirs_wav)

plt.figure()
hb.p.plot(sharey=False)

#%% filters
# hb_f = filters.FIRFilter([0.01, 0.2], [0.001, 2])(hb)
hb_f = filters.ConvolutionalFilter('rect', 1)(hb)
# hb_f = filters.IIRFilter(fp = [0.01, 0.2], fs=[0.001, 2], ftype='ellip')(hb)
hb_nc = NegativeCorrelationFilter()(hb_f)

plt.figure()
hb.p.plot(sharey=False)
hb_f.p.plot(sharey=False)
hb_nc.p.plot(sharey=False)

#%%
#%plot
hb_.append(hb)

#% save
hb = SDto1darray(hb)
hb.to_netcdf('/home/bizzego/tmp/nirs'+subject)

#%%
for hb in hb_:
   hb.p.plot(sharey=False)

#%%