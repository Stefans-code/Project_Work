from .indicators.frequencydomain import *
from .indicators.nonlinear import *
from .indicators.peaks import *
from .indicators.timedomain import *
from .sqi.sqi import *

def preset_resp(prefix='resp', method='welch'):
    e_low = PowerInBand(freq_min=0, freq_max=0.25, method=method, name="energy_low")
    e_high = PowerInBand(freq_min=0.25, freq_max=5, method=method, name="energy_high")
    resp_rate = PeakInBand(freq_min=0.25, freq_max=5, method=method, name="resp_rate")
    
    t = [e_low, e_high, resp_rate]
    
    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t