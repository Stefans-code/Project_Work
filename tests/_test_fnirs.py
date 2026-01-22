import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyphysio.specialized.fnirs import Raw2Oxy, Raw2OD, OD2Oxy, ComputeClusters, PCAFilter, \
    plot_probe, get_ss_ls_channels, get_near_channels

from pyphysio import TestData

#%%
nirs = TestData.fnirs(return_signal=True)

hb1 = Raw2Oxy()(nirs)

oc = Raw2OD()(nirs)
hb2 = OD2Oxy()(oc)

#%%
hb1.p.plot()
hb2.p.plot()

#%%
tapping = TestData.tapping(return_signal=True)

ss, ls = get_ss_ls_channels(nirs) ##so to have ss
near_1 = get_near_channels(nirs, ch_target=1)

plot_probe(nirs)

#%%
plt.figure()
nirs.p.plot()

#%%
from pyphysio.specialized.fnirs._dl_sqi import SignalQualityDeepLearning
from pyphysio.segmenters import FixedSegments, fmap

# SQI using Deep Learning
segmenter = FixedSegments(10, 20)
indicators = [SignalQualityDeepLearning()]

sqi = fmap(segmenter, indicators, nirs)

#%%
sqi = sqi['signal_SignalQualityDeepLearning'].sel({'is_good':1, 'component':0})
sqi = sqi.drop_vars(['label'])
sqi.p.plot('.-', sharey=True)

#%%
ratio_min_good = 0.8 #at least 80% of the windows should have a good value
isgood_ratios = np.sum(sqi.values, axis=0)/sqi.values.shape[0]
id_good_channels = np.where(isgood_ratios >= ratio_min_good)[0]

print("good channels: ", id_good_channels)

nirs.attrs['good_channels'] = id_good_channels

nirs = nirs.sel({'channel': id_good_channels})

#%% remove MA
import pyphysio.artefacts as artefacts

MA = artefacts.DetectMA(fuse='component')(nirs)
plt.figure()
MA.p.plot()

nirs['MA'] = MA #add MA info to nirs

nirs_noMA = artefacts.MARA()(nirs)
nirs_noMA = nirs_noMA.drop_vars('MA')

plt.figure()
nirs.p.plot(sharey=False)
nirs_noMA.p.plot(sharey=False)

#remove MA with wavelet
nirs_wav = artefacts.WaveletFilter()(nirs_noMA)

plt.figure()
nirs_noMA.p.plot(sharey=False)
nirs_wav.p.plot(sharey=False)

#%%
hb = Raw2Oxy()(nirs_wav)

plt.figure()
hb.p.plot(sharey=False)

#%% remove physio noise
pcafilt = PCAFilter()
hb_nophysio = pcafilt(hb)

plt.figure()
hb.p.plot(sharey=False)
hb_nophysio.p.plot(sharey=False)

#%%
cluster_compute = ComputeClusters(clusters=[[0,1,2], [4,5,6], [7,8,9]])
clusters = cluster_compute(hb_nophysio)

#%%
import pyphysio.filters as filters

clusters_f = filters.IIRFilter(fp = [0.01, 0.2], btype='bandpass')(clusters)

#%% GLM first level

tapping_values = tapping.values
idx_onset = np.where(np.diff(tapping_values)>0)[0] - 1
t_onset = tapping.p.get_times()[idx_onset]

tapping.p.plot()
plt.vlines(t_onset, 0, 1, 'r')

events = pd.DataFrame({'onset': t_onset})
events['duration'] = 5
events['stim_type'] = 1
        


#%%
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import run_glm

t = clusters_f.p.get_times()

dm = make_first_level_design_matrix(t, events,
                                    hrf_model='glover',
                                    drift_model='polynomial',
                                    drift_order=5)

plt.imshow(dm.values, aspect='auto', interpolation='nearest')

X = dm.values

i_ch = 0
Y = np.expand_dims(clusters_f.p.get_values()[:, i_ch, 0], 1)

assert ~np.isnan(Y).any()

labels, glm_estimates = run_glm(Y, X)
thetas = glm_estimates[labels[0]].theta[:,0]
print(thetas)

#%%
sub = 'S001'
betas = []

for i_ch in range(clusters_f.sizes['channel']):
    
        Y = np.expand_dims(clusters_f.p.get_values()[:, i_ch, 0], 1)

        assert ~np.isnan(Y).any()

        labels, glm_estimates = run_glm(Y, X)
        thetas = glm_estimates[labels[0]].theta[:,0]
        
        betas.append({'subject': sub,
                      'channel': i_ch,
                      'beta': thetas[0]})
            
betas = pd.DataFrame(betas)
print(betas)

