import pyphysio as ph
import numpy as np
import numpy.ma  as ma
from test_utils import info
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,2))

#%
signal = ph.Signal(values, fsamp, start_time)

# #%%

# info(signal)

# #%%
# a= ma.apply_along_axis(ma.mean, 0, signal)
# a= ma.apply_along_axis(np.mean, 0, signal)

# a= np.apply_along_axis(ma.mean, 0, signal)
# a= np.apply_along_axis(np.mean, 0, signal)

#%%
# ma.mean(signal, axis=0)

type(np.mean(signal))
