from ._load_nirs import load_nirx2, load_nirx, load_snirf, load_xrnirs, SDto1darray
from ..signal import create_signal as _create_signal
#TODO: add modules for loading edf, physionet?
import numpy as _np

def info_biopac(datafile):
    import bioread
    data = bioread.read_file(datafile)
    names = [ch.name for ch in data.channels]
    print(names)

def load_biopac(datafile, channel):
    import bioread
    data = bioread.read_file(datafile)
    channels = data.channels
    channel = channels[channel]
    values = channel.data
    fsamp = data.samples_per_second
    
    signal = _create_signal(values, sampling_freq=fsamp, info = {'name':channel.name})
    return(signal)

def load_text(datafile, data_col=0, sampling_freq=None, time_col=None, 
              sep=',', preprocess_function=None):
    assert (sampling_freq is not None) or (time_col is not None), "either sampling frequency or time column shoul be provided"
    data = _np.loadtxt(datafile, delimiter=sep)
    
    values = data[:, data_col] 
    if preprocess_function is not None:
        values = preprocess_function(values)
    if (time_col is not None):
        times = data[:, time_col]
        signal = _create_signal(values, times = times)
    else:
        signal = _create_signal(values, sampling_freq=sampling_freq)
    
    return(signal)
    
    