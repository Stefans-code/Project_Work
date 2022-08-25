import numpy as np
from pyphysio.signal import create_signal
import xarray as xr

import pyphysio.processing.estimators as est

from pyphysio import TestData

#%%
ecg_data = TestData().ecg()

signal = create_signal(ecg_data, sampling_freq=2048)

ibi = est.BeatFromECG()(signal)

#%%
bvp_data = TestData().bvp()

signal = create_signal(bvp_data, sampling_freq=2048)

ibi = est.BeatFromBP()(signal)
