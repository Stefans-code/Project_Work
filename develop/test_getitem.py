import pyphysio as ph
import numpy as np

# signal = ph.EvenlySignal(np.random.uniform(size=(1000, 5, 3)), 10)

# signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 10,
#                            x_values = np.arange(0, 100, 0.1),
#                            x_type='instants')

# signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 10,
#                             x_values = np.arange(0, 1000),
#                             x_type='indices')

# signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 1000,
#                             x_values = np.logspace(0, 3, 1000),
#                             x_type='instants')

signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 5, 3)), 1000,
                            x_values = np.unique(np.logspace(0, 8, 2000).astype(int))[:1000],
                            x_type='indices')

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
s_ = signal[0,0,0]
print(s_)
print(s_.shape)
print(type(s_))

#%%
s_ = signal[100:,:,:]
# print(s_)
info(s_)

#%%
s__ = s_[100:-100,:,:]
# print(s_)
info(s__)

#%%
s_ = signal[100,:,:]
# print(s_)
info(s_)

#%%
s_ = signal[100:101,:,:]
# print(s_)
info(s_)
