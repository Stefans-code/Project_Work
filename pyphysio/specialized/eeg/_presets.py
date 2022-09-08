from .indicators.frequencydomain import *
from .indicators.nonlinear import *
from .indicators.peaks import *
from .indicators.timedomain import *
from .sqi.sqi import *

def preset_eeg(prefix="eeg_", method='welch'):
    delta = PowerInBand(freq_min=0, freq_max=3, method=method, name="delta")
    theta = PowerInBand(freq_min=3.5, freq_max=7.5, method=method, name="theta")
    alpha = PowerInBand(freq_min=7.5, freq_max=13, method=method, name="alpha")
    beta = PowerInBand(freq_min=14, freq_max=30, method=method, name="beta")
    total = PowerInBand(freq_min=0, freq_max=100, method=method, name="total")
    
    t = [delta, theta, alpha, beta, total]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t