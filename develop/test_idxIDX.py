import pyphysio as ph
import numpy as np

from test_utils import info_ue

#%%    
signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 4, 3)), 10,
                           # start_time = 100,
                           x_values = np.arange(0, 200, 0.2),
                           x_type='instants')

# signal.ph['info']['good'] = np.array([[0,1,2]])

# signal.plot()
# info(signal)

print(np.median(signal))
result = np.median(signal, axis=0)

#%%
s_ = signal.segment_time(30, 34.5)

info_ue(s_)
# info(signal)

#%%
s_ = signal[300: 345]

info_ue(s_)
# info(signal)

#%%
s_ = signal.segment_idx(300,  345)

info_ue(s_)
# info(signal)

#%%    
signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 4, 3)), 10,
                           # start_time = 100,
                           x_values = np.arange(0, 2000, 2),
                           x_type='indices')

info_ue(signal)

#%%
s_ = signal.segment_time(30, 34.5)

info_ue(s_)
# info(signal)

#%%
s_ = signal[300: 345]

info_ue(s_)
# info(signal)

#%%
s_ = signal.segment_idx(300,  345)

info_ue(s_)
# info(signal)

#%%
s_ = ph.Normalize()(signal)
info_ue(s_)