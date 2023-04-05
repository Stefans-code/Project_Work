from ...indicators.frequencydomain import *
from ...indicators.nonlinear import *
from ...indicators.peaks import *
from ...indicators.timedomain import *
from ...sqi import *

def preset_sqi_ecg(prefix="SQI_", method='ar'):
    K = Kurtosis(name='kurtosis')
    SPR = SpectralPowerRatio(method, name='SPR')
    DE = DerivativeEnergy(name='DE')
    
    t = [K, SPR, DE]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t

def preset_hrv_fd(prefix="IBI_", method='welch'):
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
    # pnn10 = PNNx(threshold=10, name="pnn10")
    # pnn25 = PNNx(threshold=25, name="pnn25")
    # pnn50 = PNNx(threshold=50, name="pnn50")
    mn = Min(name="Min")
    mx = Max(name="Max")
    # sd1 = PoincareSD1(name="sd1")
    # sd2 = PoincareSD2(name="sd2")
    # sd12 = PoincareSD1SD2(name="sd12")
    # sdell = PoinEll(name="sdell")
    # DFA1 = DFAShortTerm(name="DFA1")
    # DFA2 = DFALongTerm(name="DFA2")

    t = [rmssd, sdsd, RRmean, RRstd, RRmedian, 
         # pnn10, pnn25, pnn50, 
         mn, mx, 
         # sd1, sd2, sd12,
         # sdell, DFA1, DFA2
         ]

    if prefix is not None:
        for i in t:
            i.set(name=prefix + i.get("name"))

    return t