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
signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 4, 3)), 10,
                           # start_time = 100,
                           x_values = np.arange(0, 200, 0.2),
                           x_type='instants')

# signal.ph['info']['good'] = np.array([[0,1,2]])

# signal.plot()
# info(signal)

print(np.median(signal))

#%%
s_ = signal.segment_time(30, 34.5)

info(s_)
# info(signal)

#%%
s_ = signal[300: 345]

info(s_)
# info(signal)

#%%
s_ = signal.segment_idx(300,  345)

info(s_)
# info(signal)

#%%    
signal = ph.UnevenlySignal(np.random.uniform(size=(1000, 4, 3)), 10,
                           # start_time = 100,
                           x_values = np.arange(0, 2000, 2),
                           x_type='indices')

info(signal)
#%%
s_ = signal.segment_time(30, 34.5)

info(s_)
# info(signal)

#%%
s_ = signal[300: 345]

info(s_)
# info(signal)

#%%