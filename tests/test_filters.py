import numpy as np
from pyphysio.signal import create_signal
import xarray as xr

import pyphysio.filters as filt

filters = [filt.Normalize(), 
           filt.ConvolutionalFilter('rect', 0.5),
           filt.DeConvolutionalFilter([0.01, 0.005], deconv_method='fft'),
           filt.FIRFilter(20, 30),
           filt.IIRFilter(20, 30),
           filt.KalmanFilter(100, 100),
           filt.NotchFilter(25),
           filt.ImputeNAN(),
           filt.RemoveSpikes()]

def _test_filters(signal):
    assert isinstance(signal, xr.Dataset)
    assert isinstance(signal.p.main_signal, xr.DataArray)
    
    for f in filters:
        print(f)
        result = f(signal)
        # print(f.__name__())
        assert result.p.main_signal.values.ndim == signal.p.get_values().ndim
        

def test_filter():
    sizes = [1000, (1000), (1000,1), (1000,1,1),
             (1000, 5), (1000, 5, 2)]
    
    sampling_freqs = [100]
    
    for size in sizes:
        for sampling_freq in sampling_freqs:
            data = np.random.uniform(size = size)
            signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')
            _test_filters(signal)