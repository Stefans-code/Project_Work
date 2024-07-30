#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:57:48 2024

@author: bizzego
"""
import numpy as _np
import statsmodels.api as _sm
from scipy.stats import median_abs_deviation as _median_abs_deviation
from scipy.signal import correlate as _correlate
from sklearn.metrics import normalized_mutual_info_score as _normalized_mutual_info_score
from .utils import Wavelet as _Wavelet

def _get_lagged(data1, data2, idx_lag):
    #idx_lag is the difference between 
    #the start of data1 and the start of data 2
    if idx_lag == 0:
        data1_out = data1
        data2_out = data2
    
    #idx_lag >0 data1 is delayed
    if idx_lag >0:
        data2_out = data2[idx_lag:]
        data1_out = data1[:-idx_lag]
    
    #idx_lag <0 data2 is anticipated
    if idx_lag <0:
        idx_lag = -idx_lag
        data1_out = data1[idx_lag:]
        data2_out = data2[:-idx_lag]
    return(data1_out, data2_out)

def _IRLS(y, X, max_iter=50):
    done = False
    iterations = 0
    beta_old = _np.ones(X.shape[1])
    #initialize weights to ones
    weights = _np.ones(len(y))
    while(not done):
        #a- solve beta by WLS
        #fit weighted LS
        model_WLS = _sm.WLS(y, X, weights=weights)
        results_WLS = model_WLS.fit()
        #get new beta
        beta_new = results_WLS.params
        
        #b- recalculate weights
        residuals_WLS = results_WLS.resid
        weights = _sm.robust.norms.TukeyBiweight(c=4.685).weights(residuals_WLS)
        change = abs(_np.min((beta_new - beta_old)/beta_old))
        
        #c- repeat steps 5a-b until changes in beta are small (<1%)
        if (change <0.01) or (iterations >= max_iter):
            done = True
        
        beta_old = beta_new
        
        iterations +=1
    return(beta_new)


def compute_between_channel_pairs(function, signal_1, signal_2=None, channels=None, idx_offset=None, **kwargs):
    
    if signal_2 is None:
        signal_2 = signal_1
        idx_offset = 1 if idx_offset is None else idx_offset #if no signal_2, then do not compute the correlation for same channels
    else:
        
        #check dims
        shape_1 = list(signal_1.sizes.values())
        shape_2 = list(signal_1.sizes.values())
        
        #TODO signal_1 and signal_2 might have a different number of channels
        for i,j in zip(shape_1, shape_2):
            assert i == j, "Sizes are not the same"
        idx_offset = 0 if idx_offset is None else idx_offset
    
    if channels is None:
        channels = _np.arange(shape_1[1])
        
    n_components = shape_1[2]
    corr_mat = _np.ones(shape=(len(channels), len(channels), n_components))
    
    for i_comp in range(n_components):
        for i_ch in _np.arange(len(channels)):
            ch_1 = channels[i_ch]
            s_1 = signal_1.isel({'channel': [ch_1], 'component': [i_comp]})
            
            for j_ch in _np.arange(i_ch+idx_offset, len(channels)):
                ch_2 = channels[j_ch]
                s_2 = signal_2.isel({'channel': [ch_2], 'component': [i_comp]})
                
                R = function(s_1, s_2, **kwargs)
                
                corr_mat[i_ch, j_ch, i_comp] = R
                corr_mat[j_ch, i_ch, i_comp] = R

    
    return(corr_mat)

def robust_correlation(s1, s2):
    '''
    Santosa et al 2017 "Characterization and correction of the false-discovery rates in resting state connectivity using functional near-infrared spectroscopy"
    
    NOTE: signal_1 and signal_2 are assumed pre-whitened
    '''
    
    #get values
    s1 = s1.p.get_values().ravel()
    s2 = s2.p.get_values().ravel()
    
    r = [_np.sqrt(x**2 + y**2) for (x, y) in zip(s1, s2)]
    sigma = 1.4826*_median_abs_deviation(r)
    r_norm = r/sigma
    
    weights = _sm.robust.norms.TukeyBiweight(c=4.685).weights(r_norm)
    
    s1_s = s1*weights
    s2_s = s2*weights
    
    X1 = _np.expand_dims(s2_s, 1)
    X1 = _np.concatenate([_np.ones(shape=(len(s2_s), 1)), X1],
                         axis=1) #add constant term
    
    beta12 = _IRLS(s1_s, X1)
    
    X2 = _np.expand_dims(s1_s, 1)
    X2 = _np.concatenate([_np.ones(shape=(len(s1_s), 1)), X2], 
                         axis=1) #add constant term
    
    beta21 = _IRLS(s2_s, X2)
    
    R = _np.sqrt(beta12[1]*beta21[1])
    return(R)

def dtw_distance(s1, s2,
                 method='Euclidean',step='asymmetric', 
                 wtype='sakoechiba', openend=True, openbegin=True, 
                 wsize=5):
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    
    #get values
    s1 = s1.p.get_values().ravel()
    s2 = s2.p.get_values().ravel()
    
    rpy2.robjects.numpy2ri.activate()
    R = rpy2.robjects.r
    DTW = importr('dtw')
    dtwstep = getattr(DTW, step)
    
    alignment = R.dtw(s1, s2, dist_method=method, 
                      step_pattern=dtwstep, 
                      window_type=wtype,
                      keep_internals=False, distance_only=True, 
                      open_end=openend, open_begin=openbegin, 
                      **{'window.size':wsize})
    
    dist = alignment.rx('distance')[0][0]
    
    return(dist)
    

def lagged_cross_corr(s1, s2,
                      maxlag = 10, absolute=False):
    #get values
    s1 = s1.p.get_values().ravel()
    s2 = s2.p.get_values().ravel()
    
    dist_lags = []
    for curr_lag in _np.arange(-maxlag, maxlag+1):
        s1_lag, s2_lag = _get_lagged(s1, s2, curr_lag)
        
        s1_lag = (s1_lag - _np.mean(s1_lag))/_np.std(s1_lag)
        s2_lag = (s2_lag - _np.mean(s2_lag))/_np.std(s2_lag)
        
        c = _correlate(s1_lag, s2_lag, mode='valid')
        c = c/len(s1_lag) #normalize as different lags have different lengths
        
        dist_lags.append(c)
    
    dist_lags = _np.array(dist_lags)
    if absolute:
        dist_lags = abs(dist_lags)
    
    dist = dist_lags[_np.argmax(dist_lags)][0]
    return(dist)

def mutual_info(s1, s2, 
                nbins=100):
    #get values
    s1 = s1.p.get_values().ravel()
    s2 = s2.p.get_values().ravel()
    
    #compute bins to discretize the signals
    bins1 = _np.linspace(_np.min(s1), _np.max(s1), nbins)
    bins2 = _np.linspace(_np.min(s2), _np.max(s2), nbins)
    
    #discretize
    s1_digit = _np.digitize(s1, bins1)
    s2_digit = _np.digitize(s2, bins2)
    
    #compute mi
    mi = _normalized_mutual_info_score(s1_digit, s2_digit)
    return(mi)

def wavelet_cohoerence(s1, s2, target_freqs=None, **kwargs):
    '''
    Parameters
    ----------
    s1 : signal 1
        pyphysio xarray
    s2 : TYPE
        pyphysio xarray
    target_freqs : array-like, optional
        The target frequencies (need to be in a decreasing order). 
        If None the scales will be automatically computed and, thus, the frequencies.
        The default is None.

    Returns
    -------
    
    
    '''
    import scipy.fftpack as _fft
    from scipy.signal import convolve2d as _convolve2d
    
    def smooth(W, scales, nNotes):
        # code adapted from pycwt.mother.Morlet.smooth()

        m, n = W.shape

        n_ = int(2 ** _np.ceil(_np.log2(len(W[0, :]))))
        # Filter in time.
        k = 2 * _np.pi * _fft.fftfreq(n_)
        k2 = k ** 2

        # Smoothing by Gaussian window (absolute value of wavelet function)
        F = _np.exp(-0.5 * (scales[:, _np.newaxis] ** 2) * k2)  # Outer product
        smooth = _fft.ifft(F * _fft.fft(W, axis=1, n=n_),
                           axis=1, n=n_, overwrite_x=True)
        T = smooth[:, :n]  # Remove possibly padded region due to FFT

        if _np.isreal(W).all():
            T = T.real

        # Filter in scale
        wsize = nNotes*2
        
        #create boxcar win
        win = _np.zeros(int(_np.round(wsize)))
        win[0] = win[-1] = 0.5
        win[1:-1] = 1
        win /= win.sum()
        
        T = _convolve2d(T, win[:, _np.newaxis], 'same')  # Scales are "vertical"

        return T
    
    Wavelet = _Wavelet(freqs=target_freqs, detrend=True, normalize=False, **kwargs)
    
    W1 = Wavelet(s1)
    W2 = Wavelet(s2)
    
    coef1 = W1.p.main_signal.values[:,:,0,0]
    coef2 = W2.p.main_signal.values[:,:,0,0]
    coef12 = coef1 * coef2.conj()
    
    scales = Wavelet._params['scales']
    nNotes = Wavelet._params['nNotes']
    scaleMatrix = _np.ones([1, coef1.shape[1]]) * scales[:, None]
    
    coef1 = _np.abs(coef1)**2 / scaleMatrix
    coef2 = _np.abs(coef2)**2 / scaleMatrix
    coef12 = coef12    / scaleMatrix
    #TODO see if we can se gaussian smooth: _ndimage.gaussian_filter(coef1, 8)
    S1 = smooth( coef1, scales, nNotes)
    S2 = smooth( coef2, scales, nNotes)
    S12 = smooth(coef12, scales, nNotes)
    
    WC = abs(S12)**2 / (S1*S2)
    
    WC = Wavelet._compute_coi(WC)
    WC_out = _np.nanmean(WC)
    return(WC_out)
    
    
    