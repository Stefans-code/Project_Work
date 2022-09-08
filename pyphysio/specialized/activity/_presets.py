from .indicators.frequencydomain import *
from .indicators.nonlinear import *
from .indicators.peaks import *
from .indicators.timedomain import *
from .sqi.sqi import *

def preset_activity(prefix='activity', method='welch'):
    mx = Max(name='maximum')
    mn = Min(name='minimum')
    mean = Mean(name='mean')
    rng = Range(name='range')
    sd = StDev(name='sd')
    auc = AUC(name='auc')
    en_25 = PowerInBand(freq_min=0, freq_max=25, method=method, name="en_25")
    
    t = [mx, mn, mean, rng, sd, auc, en_25]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t
