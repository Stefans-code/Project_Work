import numpy as _np
from pyphysio.signal import create_signal
import xarray as xr
import pyphysio.specialized.heart as heart
import pyphysio.segmenters as segm

from pyphysio import TestData

#%%
ecg_data = TestData().ecg()

signal = create_signal(ecg_data, sampling_freq=2048)

ibi = heart.BeatFromECG()(signal)

#%%
ibi_ = ibi.p.process_na('remove')
ibi_.p.plot()

#%%
from pyphysio.interactive import Annotate

# annotator = Annotate(signal, ibi)

bvp_data = TestData().bvp() 

signal = create_signal(bvp_data, sampling_freq=2048)

ibi = heart.BeatFromBP()(signal)

ibi_rco = heart.BeatOptimizer()(ibi)

ibi_corr = heart.RemoveBeatOutliers()(ibi_rco)

ibi = ibi.p.process_na('remove')

#%%
import pyphysio.indicators.frequencydomain as fd

ibi = ibi.p.process_na('remove')

hf = fd.PowerInBand(0.15, 0.4, 'welch')

hf_ibi = hf(ibi.p.resample(4))

#%%
label = _np.zeros(1200)
label[300:600] = 1 #stimulus 1
label[900:1200] = 2 # stimulus 2

label = create_signal(label, sampling_freq = 10)
segmenter = segm.FixedSegments(10, 20, timeline=label)
result = segm.fmap(segmenter, [fd.PowerInBand(0.15, 0.4, 'welch', name='HF'), 
                               fd.PowerInBand(0.04, 0.15, 'welch', name = 'LF')], ibi.p.resample(4))

#%%
result_df = segm.indicators2df(result)