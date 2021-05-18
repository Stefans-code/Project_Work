import pyphysio as ph
import numpy as np

def info(s_):
    print(type(s_))
    print(s_.shape)
    print(s_.get_indices()[:10])
    print(s_.get_indices()[-10:])
    
    print(s_.get_times()[:10])
    print(s_.get_times()[-10:])
    
    print(s_.get_start_time())
    print(s_.get_end_time())
    
    print(s_.get_duration())

#%%    
signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 10,
                           # start_time = 100,
                           x_values = np.arange(0, 100, 0.1),
                           x_type='instants')

#%%
s_ = signal.segment_time(30, 34.5)
info(s_)

#%%
s_ = signal[300: 345]
info(s_)
# signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 10,
#                             x_values = np.arange(0, 1000),
#                             x_type='indices')

# signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 1000,
#                             x_values = np.logspace(0, 3, 1000),
#                             x_type='instants')

# signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 1000,
#                             x_values = np.unique(np.logspace(0, 8, 2000).astype(int))[:1000],
#                             x_type='indices')