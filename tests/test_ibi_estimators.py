import numpy as _np
from pyphysio.signal import create_signal
import xarray as xr
import pyphysio.specialized.heart as heart

from pyphysio import TestData

#%%
ecg_data = TestData().ecg()

signal = create_signal(ecg_data, sampling_freq=2048)

ibi = heart.BeatFromECG()(signal)

#%%


from pyphysio.interactive import Annotate

annotator = Annotate(signal, ibi)

bvp_data = TestData().bvp() 

signal = create_signal(bvp_data, sampling_freq=2048)

ibi = heart.BeatFromBP()(signal)

ibi_rco = heart.BeatOptimizer()(ibi)

ibi_corr = heart.RemoveBeatOutliers()(ibi_rco)

ibi = ibi.p.process_na('remove')
