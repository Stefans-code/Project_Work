import numpy as _np
#we use the create_signal function.
#TODO: we should test it
from ..signal import create_signal

#%%
def to3d(template, n_ch=None, n_cp=None):
    assert template.ndim <= 3
    if template.ndim == 1:
        assert n_ch is not None
        assert n_cp is not None
        
        template = _np.expand_dims(template, [1,2])
        template = _np.repeat(template, n_ch, axis=1)
        template = _np.repeat(template, n_cp, axis=2)
    elif template.ndim == 2:
        assert n_cp is not None
        template = _np.expand_dims(template, 2)
        template = _np.repeat(template, n_cp, axis=2)
    return(template)

def generate_signal(n_timepoints, n_ch, n_cp, sampling_freq, start_time, sigtype, **kwargs):
    
    out_signal = _np.zeros(shape=(n_timepoints, n_ch, n_cp))
    
    if sigtype == 'd':
        assert 'delta_times' in kwargs
        assert 'delta_values' in kwargs
        delta_values = to3d(kwargs['delta_values'], n_ch, n_cp)
        delta_times = to3d(kwargs['delta_times'], n_ch, n_cp)
        
    if sigtype == 's':
        assert 'amp' in kwargs
        assert 'freq' in kwargs
        
        amp = kwargs['amp']
        freq = kwargs['freq']
        if _np.isscalar(amp):
            amp = _np.array([amp])
        if _np.isscalar(freq):
            freq = _np.array([freq])
        
        amp = to3d(amp, n_ch, n_cp)
        freq = to3d(freq, n_ch, n_cp)
        
    
    for i_ch in range(n_ch):
        for i_cp in range(n_cp):
            if sigtype == '0':
                data = _np.zeros(n_timepoints)
            elif sigtype == '1':
                data = _np.ones(n_timepoints)
            elif sigtype == 'd':
                data = _generate_deltas(n_timepoints, sampling_freq, start_time, 
                                        delta_times[:, i_ch, i_cp].ravel(),
                                        delta_values[:, i_ch, i_cp].ravel())
            elif sigtype == 's':
                data = _generate_sinusoid(n_timepoints, sampling_freq, start_time, 
                                          amp[0, i_ch, i_cp], 
                                          freq[0, i_ch, i_cp])
            out_signal[:, i_ch, i_cp] = data
    
    signal = create_signal(out_signal, 
                           sampling_freq=sampling_freq,
                           start_time=start_time,
                           name = f'synth_{sigtype}')
    return(signal)


def _generate_deltas(n_timepoints, sampling_freq, start_time, delta_times, delta_values):
    assert len(delta_times) == len(delta_values)
    delta_times_nostart = delta_times - start_time
    
    data = _np.zeros(n_timepoints)
    
    idx_delta_times = (delta_times_nostart * sampling_freq).astype(int)
    data[idx_delta_times] = delta_values
    
    return(data)


def _generate_sinusoid(n_timepoints, sampling_freq, start_time, amp, freq):
    times = _np.arange(n_timepoints) / sampling_freq
    values = amp * _np.sin(2*_np.pi*freq* times)
    return(values)


'''
#%%
signal = generate_signal(100, 3,2,10, 0, '1')
print(signal.p.main_signal.values.shape)
signal.p.plot()

#%%
delta_values = _np.zeros((1,3,2))
delta_times = _np.ones((1,3,2))

delta_values[0,0,0] = 4
delta_values[0,1,1] = 6
delta_values[0,2,0] = 2


signal = generate_signal(20, 3, 2, 10, 0, 'd',
                         delta_times = delta_times, 
                         delta_values = delta_values)
print(signal.p.main_signal.values.shape)
signal.p.plot()


#%%
amp = _np.array([[[1,2,3]]])
freq = _np.array([[[2,3,4]]])

signal = generate_signal(2000, 1, 3, 100, 0, 's',
                         amp = amp, 
                         freq = freq)
print(signal.p.main_signal.values.shape)
signal.p.plot()

'''