from ._load_nirs import load_nirx2, load_nirx, load_snirf, load_xrnirs, SDto1darray
from ..signal import create_signal as _create_signal
#TODO: add modules for loading text, edf, physionet?

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