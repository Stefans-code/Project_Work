
import pyphysio as ph
import pyphysio.processing.estimators as est
from pyphysio import TestData
import numpy as np

ecg_data = TestData.ecg()
fsamp=2048
tstart_ecg=0
ecg = ph.EvenlySignal(values = ecg_data[:20480], 
                      sampling_freq = fsamp, 
                      start_time = tstart_ecg)

ibi_ecg = est.BeatFromECG()

ibi = ibi_ecg(ecg)

#%%
print(np.median(ibi))

print(ph.Median()(ibi))

#%%
ibi2 = ph.UnevenlySignal(ibi.get_values(), 
                         fsamp, ibi.get_values()[0],
                         x_values = ibi.get_indices(),
                         x_type='indices')

np.median(ibi2)
