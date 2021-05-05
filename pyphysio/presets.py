from .indicators.frequencydomain import *
from .indicators.nonlinear import *
from .indicators.peaks import *
from .indicators.timedomain import *
from .sqi.sqi import *

def preset_sqi_ecg(prefix="SQI_", method='ar'):
    K = Kurtosis(name='kurtosis')
    SPR = SpectralPowerRatio(method, name='SPR')
    DE = DerivativeEnergy(name='DE')
    
    t = [K, SPR, DE]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t

def preset_hrv_fd(prefix="IBI_", method='ar'):
    VLF = PowerInBand(interp_freq=4, freq_max=0.04, freq_min=0.00001, method=method, name="VLF_Pow")
    LF = PowerInBand(interp_freq=4, freq_max=0.15, freq_min=0.04, method=method, name="LF_Pow")
    HF = PowerInBand(interp_freq=4, freq_max=0.4, freq_min=0.15, method=method, name="HF_Pow")
    Total = PowerInBand(interp_freq=4, freq_max=2, freq_min=0.00001, method=method, name="Total_Pow")
    
    t = [VLF, LF, HF, Total]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_hrv_td(prefix="IBI_"):
    rmssd = RMSSD(name="RMSSD")
    sdsd = SDSD(name="SDSD")
    RRmean = Mean(name="Mean")
    RRstd = StDev(name="RRstd")
    RRmedian = Median(name="Median")
    pnn10 = PNNx(threshold=10, name="pnn10")
    pnn25 = PNNx(threshold=25, name="pnn25")
    pnn50 = PNNx(threshold=50, name="pnn50")
    mn = Min(name="Min")
    mx = Max(name="Max")
    sd1 = PoincareSD1(name="sd1")
    sd2 = PoincareSD2(name="sd2")
    sd12 = PoincareSD1SD2(name="sd12")
    sdell = PoinEll(name="sdell")
    DFA1 = DFAShortTerm(name="DFA1")
    DFA2 = DFALongTerm(name="DFA2")

    t = [rmssd, sdsd, RRmean, RRstd, RRmedian, pnn10, pnn25, pnn50, mn, mx, sd1, sd2, sd12,
         sdell, DFA1, DFA2]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_phasic(delta, prefix="pha_"):
    mean = Mean()
    std = StDev()
    rng = Range()
    pks_max = PeaksMax(delta=delta)
    pks_min = PeaksMin(delta=delta)
    pks_mean = PeaksMean(delta=delta)
    n_peaks = PeaksNum(delta=delta)
    dur_mean = DurationMean(delta=delta, win_pre=2, win_post=2)
    slopes_mean = SlopeMean(delta=delta, win_pre=2, win_post=2)
    auc = AUC()

    t = [mean, std, rng, pks_max, pks_min, pks_mean, n_peaks, dur_mean, slopes_mean, auc]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.__class__.__name__)

    return t


def preset_tonic(prefix="ton_"):
    mean = Mean()
    std = StDev()
    rng = Range()
    auc = AUC()

    t = [mean, std, rng, auc]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.__class__.__name__)

    return t

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


def preset_emg(prefix='emg_', method = 'welch'):
    mx = Max(name='maximum')
    mn = Min(name='minimum')
    mean = Mean(name='mean')
    rng = Range(name='range')
    sd = StDev(name='sd')
    auc = AUC(name='auc')
    en4_40 = PowerInBand(freq_min=4, freq_max=40, method=method, name="en_4_40")
    
    t = [mx, mn, mean, rng, sd, auc, en4_40]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t


def preset_resp(prefix='resp', method='welch'):
    e_low = PowerInBand(freq_min=0, freq_max=0.25, method=method, name="energy_low")
    e_high = PowerInBand(freq_min=0.25, freq_max=5, method=method, name="energy_high")
    resp_rate = PeakInBand(freq_min=0.25, freq_max=5, method=method, name="resp_rate")
    
    t = [e_low, e_high, resp_rate]
    
    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t

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
