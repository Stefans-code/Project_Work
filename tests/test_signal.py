import numpy as np
from pyphysio.signal import create_signal
import xarray as xr

def test_gets(signal):
    assert isinstance(signal, xr.Dataset)
    assert isinstance(signal.p.main_signal, xr.DataArray)
    
    main_sig = signal.p.main_signal
    
    assert (main_sig.p.get_values() == signal.p.get_values()).all()
    assert (main_sig.p.get_times() == signal.p.get_times()).all()
    assert main_sig.p.get_sampling_freq() == signal.p.get_sampling_freq()
    assert main_sig.p.segment_time(0, 0.01) == signal.p.segment_time(0, 0.01)
    

sizes = [1000, (1000), (1000,1), (1000,1,1),
         (1000, 5), (1000, 5, 2), (1, 5, 2)]

sampling_freqs = [0.01, 1, 1./3, 100]

for size in sizes:
    for sampling_freq in sampling_freqs:
        data = np.random.uniform(size = size)
        signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')
        test_gets(signal)
        
   
#%%
print(signal.p.get_sampling_freq())
# data.p.plot()
