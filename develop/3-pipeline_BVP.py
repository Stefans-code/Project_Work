import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyphysio as ph

from pyphysio import TestData
bvp_data = TestData.bvp()

fsamp = 2048
bvp = ph.Signal(values = bvp_data, sampling_freq = fsamp)
ibi = ph.BeatFromBP()(bvp)

#%%
np.median(ibi)

#%%
ibi_corrected = ph.Annotate(bvp, ibi).ibi_ok

#%%
id_wrong_ibi = ph.BeatOutliers()(ibi_corrected)
print(id_wrong_ibi)


#%%
ibi_no_outliers = ph.FixIBI(id_wrong_ibi)(ibi_corrected)


#%%
ibi_corrected.plot()
ibi_no_outliers.plot()

