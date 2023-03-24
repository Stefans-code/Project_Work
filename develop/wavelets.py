import numpy as np
from pyphysio.specialized.fnirs import load_xrnirs
from scipy.signal import detrend as _detrend
# from scipy import ndimage as _ndimage
import matplotlib.pyplot as plt

# import pycwt
# import pycwt.wavelet as wavelet

import scipy.fftpack as fft
from scipy.signal import convolve2d
import pywt

# nirs = load_xrnirs('/home/bizzego/UniTn/data/RP/ITA/Signals/hb_processed/TN035/BASE/A_clusters')

# x = nirs.p.main_signal.values[:,0,0].ravel()

# N = len(x)

# #ignore fsamp
# fsamp = 10
# dt = 1/fsamp
# nNotes = 36

#%%
fsamp = 100
t = np.arange(10000)/fsamp
x = np.sin(2*np.pi*1*t) + np.sin(2*np.pi*5*t)

# plt.plot(t, x)

#%%
def compute_wavelet(x, wtype='cmor_1.5-1.0', scales = None, minScale = 2, nNotes = 12):
    # nOctaves_pywt = int(np.log2(2*N//2))
    # scales_pywt = 2**np.arange(1, nOctaves_pywt, 1.0/nNotes)

    #generate scales for pywt using formulas from pycwt
    #Note: pycwt uses flambda too!
    
    N = len(x)
    
    #use nsamp and nyq_freq instead of t, fsamp
    if scales is None:
        # The scales as of Mallat 1999
        # minScale = 2 # / wavelet.flambda()
        nOctaves = int(np.round(np.log2(N/2) / (1/nNotes)))
        scales = minScale * 2 ** (np.arange(0, nOctaves + 1) * (1/nNotes))
        
    #% wavelet
    # W_pycwt, scales_pycwt, freqs_pycwt, coit_pycwt, _, _ = pycwt.cwt(x, 1, 
    #                                                                  1/nNotes, 
    #                                                                  wavelet=wavelet.MexicanHat())
    # coif_pycwt = 1/coit_pycwt
    
    W, freqs = pywt.cwt(x, scales, wavelet=wtype)
    
    coif_ = 1/(2*np.arange(1, N//2))
    
    # coif_ = fsamp/(2*np.arange(1, N//2))
    min_coif = coif_[-1]
    coif = np.zeros(N) + min_coif
    coif[:len(coif_)] = coif_
    coif[-len(coif_):] = coif_[::-1]

    # W_result_pywt = {'W': W_pywt,
    #                  'scales': scales,
    #                  'freqs': freqs_pywt,
    #                  'coif': coif_pywt}
    
    return(W, freqs, coif)

def smooth(W, scales, nNotes):
    # code adapted from pycwt.mother.Morlet.smooth()

    # The smoothing is performed by using a filter given by the absolute
    # value of the wavelet function at each scale, normalized to have a
    # total weight of unity, according to suggestions by Torrence &
    # Webster (1999) and by Grinsted et al. (2004).
    m, n = W.shape

    n_ = int(2 ** np.ceil(np.log2(len(W[0, :]))))
    # Filter in time.
    k = 2 * np.pi * fft.fftfreq(n_)
    k2 = k ** 2

    # Smoothing by Gaussian window (absolute value of wavelet function)
    # using the convolution theorem: multiplication by Gaussian curve in
    # Fourier domain for each scale, outer product of scale and frequency
    F = np.exp(-0.5 * (scales[:, np.newaxis] ** 2) * k2)  # Outer product
    smooth = fft.ifft(F * fft.fft(W, axis=1, n=n_),
                      axis=1, n=n_, overwrite_x=True)
    T = smooth[:, :n]  # Remove possibly padded region due to FFT

    if np.isreal(W).all():
        T = T.real

    # Filter in scale. For the Morlet wavelet it's simply a boxcar with
    # 0.6 width.
    wsize = nNotes*2
    
    #create boxcar win
    win = np.zeros(int(np.round(wsize)))
    win[0] = win[-1] = 0.5
    win[1:-1] = 1
    win /= win.sum()
    
    T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

    return T

#%%
scales_target = pywt.frequency2scale('cmor_1.5-1.0', np.arange(0.01, 20, 0.01)/fsamp)

WT, freqs, coif = compute_wavelet(x,
                                  wtype='cmor_1.5-1.0',
                                  scales=scales_target)

#%%
scales = pywt.frequency2scale('mexh', freqs)

N=WT.shape[1]

scaleMatrix = np.ones([1, N]) * scales[:, None]

WT_norm = WT**2 / scaleMatrix

# plt.imshow(abs(WT_norm), aspect='auto')

spect = np.mean(abs(WT_norm), axis=1)

plt.plot(freqs*fsamp, spect)

#%%
plt.imshow(abs(WT_norm), aspect='auto')

#%%
#select coi
Wcoi_pywt = W_pywt['W'].copy()
for i in range(W_pywt['W'].shape[1]):
    idx_na = np.where(W_pywt['freqs'] < W_pywt['coif'][i])[0]
    Wcoi_pywt[idx_na, i] = np.nan


Wcoi_pycwt = W_pycwt['W'].copy()
for i in range(W_pycwt['W'].shape[1]):
    idx_na = np.where(W_pycwt['freqs'] < W_pycwt['coif'][i])[0]
    Wcoi_pycwt[idx_na, i] = np.nan


fig, axes = plt.subplots(2,1)
axes[0].imshow(abs(Wcoi_pywt), aspect='auto')
axes[1].imshow(abs(Wcoi_pycwt), aspect='auto')

#%% xwt
nirs1 = load_xrnirs('/home/bizzego/UniTn/data/RP/ITA/Signals/hb_processed/TN035/BASE/A_clusters')
nirs2 = load_xrnirs('/home/bizzego/UniTn/data/RP/ITA/Signals/hb_processed/TN035/BASE/B_clusters')

x1 = nirs1.p.main_signal.values[:,0,0].ravel()
x2 = nirs2.p.main_signal.values[:,0,0].ravel()

max_len = np.min([len(x1), len(x2)])
x1 = x1[:max_len]
x2 = x2[:max_len]

# detrend and normalize
x1 = _detrend(x1,type='linear')
x2 = _detrend(x2,type='linear')
stddev1 = x1.std()
x1 = x1 / stddev1
stddev2 = x2.std()
x2 = x2 / stddev2

#%%
# cwt
W1 = compute_wavelet(x1)
W2 = compute_wavelet(x2)

#%%
idx_type = 1 #0 = pywt, 1 = pycwt
coef1 = W1[idx_type]['W']
coef2 = W2[idx_type]['W']

freqs = W1[idx_type]['freqs']
scales = W1[idx_type]['scales']

# Calculates the cross transform of xs1 and xs2.
coef12 = coef1 * coef2.conj()

# coherence
scaleMatrix = np.ones([1, N]) * scales[:, None]

#%%
# S1_pywt = _ndimage.gaussian_filter( np.abs(coef1)**2 / scaleMatrix, sigma=9)
# S2_pywt = _ndimage.gaussian_filter( np.abs(coef2)**2 / scaleMatrix, sigma=0.01)
# S12_pywt = _ndimage.gaussian_filter(       coef12    / scaleMatrix, sigma=0.01)

S1_pywt = smooth( np.abs(coef1)**2 / scaleMatrix, scales)
S2_pywt = smooth( np.abs(coef2)**2 / scaleMatrix, scales)
S12_pywt = smooth(       coef12    / scaleMatrix, scales)


WC_pywt = abs(S12_pywt)**2 / (S1_pywt * S2_pywt)

plt.figure()
# plt.imshow(S1_pywt[-128:,:], aspect='auto')
plt.imshow(abs(S1_pywt[-128:,:]), aspect='auto')

#%%


S1_pycwt =  smooth(       np.abs(coef1)**2 / scaleMatrix, scales)
S2_pycwt =  smooth(       np.abs(coef2)**2 / scaleMatrix, scales)
S12_pycwt = smooth(              coef12    / scaleMatrix, scales)

WC_pycwt = np.abs(S12_pycwt) ** 2 / (S1_pycwt * S2_pycwt)

plt.figure()
plt.imshow(abs(S1_pycwt[-128:,:]), aspect='auto')
# plt.imshow(WC_pycwt[-128:,:], aspect='auto')

pycwt.wct(x1, x2, 1/fsamp, wavelet=wavelet.MexicanHat())
#check lowlevel
#%%
plt.figure()
plt.imshow(WCT_wt, aspect='auto')

plt.figure()
plt.imshow(WCT_cwt, aspect='auto')

#%%
# cone of influence in frequencies (conservative version)
frequencies = freqs / dt

coif_ = 1/(2*times[1:len(times)//2])
min_coif = coif_[-1]

coif = _np.zeros(len(times)) + min_coif

coif[:len(coif_)] = coif_
coif[-len(coif_):] = coif_[::-1]