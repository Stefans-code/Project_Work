import pyphysio as ph
import numpy as np
from test_utils import info
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))
info = {'info': 'info'}
sizes = [1000, (1000, 5), (1000, 5, 3)]

x_indices = [np.arange(1000).astype(int), 
             np.arange(0,2000,2).astype(int), 
             np.unique(np.logspace(0, 5, 2000).astype(int))[:1000]]
fsamps = [10, 0.1, 10.1]
start_times = [-100, 0, 0.00001]

random_values = [np.random.uniform(size = x) for x in sizes]
ones_values = [np.ones(shape = x) for x in sizes]
zeros_values = [np.zeros(shape = x) for x in sizes]

values = [random_values, ones_values, zeros_values]

#%%
for i_x, x in enumerate(x_indices):

i_x = 1
x = np.arange(0,2000,2).astype(int)

    for f in fsamps:
        
        for t in start_times:
            for v in values:
                for i_sh, sh in enumerate(sizes):
                    s = ph.Signal(data=v[i_sh], 
                                  sampling_freq=f, 
                                  start_time=t, info=info, 
                                  x_values=x, x_type='indices')
                
                    start_time = s.get_start_time()
                    indices = s.get_indices()

                    assert np.sum(s.get_values()) == np.sum(s.data[indices])
                    
                    if i_x == 0:
                        assert len(indices) == 1000, len(indices)
                        assert start_time == t, start_time
                        assert s.get_times()[123] == 123/f + t
                    if i_x == 1:
                        assert len(indices) == 1000, len(indices)
                        assert s.get_duration() == 2000/f, s.get_duration()
                    if i_x == 2:
                        assert len(indices) == 1000, len(indices)
                        
                    # general_tests(s,f,t,v[i_sh],i_sh)
                            
signal = ph.Signal(values, fsamp, start_time)
info(signal)

#%%
sig_ = signal.resample(100,'cubic')
info(sig_)

#%%
signal_ = signal[:100]
info(signal_)

#%%
signal_ = signal[:100,0]
info(signal_)

#%%
signal_ = signal[::2,0]
info(signal_)

#%%
signal_ = signal[:100,0,0]
info(signal_)

#%%
signal_ = signal[100:480,0,0]
info(signal_)

#%%
signal_ = signal[100,0,0]
print(type(signal_))

#%%
signal_ = signal[100]
info(signal_)

#%%
signal_ = signal[100, 3]
info(signal_)

#%%
import pyphysio as ph
import numpy as np
from test_utils import info
fsamp = 10
start_time = 0
values = np.random.uniform(size=(1000,5,3))

x_indices = np.arange(0, 2000, 2)

signal = ph.Signal(values, fsamp, start_time,
                   x_values = x_indices,
                   x_type = 'indices')
info(signal)

#%%
sig = signal.fill()
info(sig)


#%%
signal_ = signal[:100]
info(signal_)

#%%
signal_ = signal[:100,0]
info(signal_)

#%%
signal_ = signal[:100,0,0]
info(signal_)

#%%
signal_ = signal[100:480,0,0]
info(signal_)

#%%
signal_ = signal.segment_time(55.3, 80)
info(signal_)

#%%
signal_ = signal.segment_IDX(0, 80)
info(signal_)