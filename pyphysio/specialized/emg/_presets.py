from ...indicators.frequencydomain import PowerInBand as _PowerInBand
from ....indicators.timedomain import Max as _Max, Min as _Min, Mean as _Mean, \
    Range as _Range, StDev as _StDev, AUC as _AUC

def preset_emg(prefix='emg_', method = 'welch'):
    mx = _Max(name='maximum')
    mn = _Min(name='minimum')
    mean = _Mean(name='mean')
    rng = _Range(name='range')
    sd = _StDev(name='sd')
    auc = _AUC(name='auc')
    en4_40 = _PowerInBand(freq_min=4, freq_max=40, method=method, name="en_4_40")
    
    t = [mx, mn, mean, rng, sd, auc, en4_40]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t