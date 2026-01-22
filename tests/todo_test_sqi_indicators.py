import numpy as np
from pyphysio.signal import create_signal
import xarray as xr

import pyphysio.sqi as sqi

indicators = [
    sqi.Kurtosis(threshold=(0, 10)),
    sqi.Entropy(threshold=(0, 10)),
    # sqi.SpectralPowerRatio(threshold=(0, 10)),
    sqi.CVSignal(threshold=(0, 10)),
    sqi.PercentageNAN(threshold=(0, 10))
    ]

def _test_indicators(signal):
    for i in indicators:
        result = i(signal)
        assert result.values.ndim == signal.p.get_values().ndim+1, i
        

def test_indicator():
    sizes = [1000, (1000), (1000,1), (1000,1,1),
             (1000, 5), (1000, 5, 2),
             (1000, 2, 2, 2)]
    
    sampling_freqs = [100]
    
    for size in sizes:
        for sampling_freq in sampling_freqs:
            data = np.random.uniform(size = size)
            signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')
            _test_indicators(signal)
            
test_indicator()
#%%