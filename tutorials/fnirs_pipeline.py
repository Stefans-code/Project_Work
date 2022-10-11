import numpy as np
from pyphysio.loaders import load_nirx2
from pyphysio.specialized.fnirs import SignalQualityDeepLearning
from pyphysio.segmenters import FixedSegments, fmap
import pyphysio.artefacts as artefacts
from pyphysio.specialized.fnirs import Raw2Oxy, NegativeCorrelationFilter, SDto1darray
import pyphysio.filters as filters

ratio_min_good = 0.8 #at least 80% of the windows should have a good value
segmenter = FixedSegments(10, 20)

indicators = [SignalQualityDeepLearning()]

#%%
nirsA = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/hyper/pilot/TN002/TN002_RR/tn002fa_001') 
nirsB = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/hyper/pilot/TN002/TN002_RR/tn002fb_001') 

assert np.sum(np.isnan(nirsA.p.main_signal.values)) == 0, "nans in nirsA"
assert np.sum(np.isnan(nirsB.p.main_signal.values)) == 0, "nans in nirsB"

#%%
hb_ = []
for nirs, subject in zip([nirsA, nirsB], ['A', 'B']):
    #% SQI using Deep Learning
    sqi = fmap(segmenter, indicators, nirs)
    sqi = sqi.drop_vars(['component_start', 'component_stop', 'label'])
    sqi = sqi.sel(component = [0])
    
    # select channels
    isgood_vals = sqi['nirs_SignalQualityDeepLearning_isgood'].values[:,:,0]
    isgood_ratios = np.sum(isgood_vals, axis=0)/isgood_vals.shape[0]
    id_good_channels = np.where(isgood_ratios >= ratio_min_good)[0]
    
    print(subject + " good channels: ", id_good_channels)

    #% preprocessing
    nirs = artefacts.MARA()(nirs)
    nirs = artefacts.WaveletFilter()(nirs)
    
    #% convert to hb
    hb = Raw2Oxy(age=21)(nirs)
    
    #% filters
    # hb = filters.FIRFilter([0.5], [0.6])(hb)
    hb = filters.ConvolutionalFilter('rect', 1)(hb)
    hb = NegativeCorrelationFilter()(hb)
    
    #%plot
    hb_.append(hb)

    #% save
    hb = SDto1darray(hb)
    hb.to_netcdf('/home/bizzego/tmp/nirs'+subject)

#%%
for hb in hb_:
    hb.p.plot(sharey=False)
    

