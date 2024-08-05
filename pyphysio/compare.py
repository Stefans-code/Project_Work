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

import statsmodels.tsa.api as _smt

#%%%
# IAAFT surrogates
#from : https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/surrogates.py

def _ft(x):
    """Return simple Fourier transform surrogates.

    Returns phase randomized (FT) surrogates that preserve the power
    spectrum (or equivalently the linear correlations), but completely
    destroy the probability distribution.

    Parameters
    ----------
    x : array
        Real input array containg the time series.

    Returns
    -------
    y : array
        Surrogates with the same power spectrum as x.
    """
    y = _np.fft.rfft(x, axis = 0)

    phi = 2 * _np.pi * _np.random.random(y.shape[0])

    phi[0] = 0.0
    if x.shape[0] % 2 == 0:
        phi[-1] = 0.0

    y = y * _np.exp(1j * phi)
    
    return _np.fft.irfft(y, n=x.shape[0], axis=0)

def surrogate_AAFT(x):
    """Return amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted (AAFT) surrogates with
    crudely the same power spectrum and distribution as the original
    data (Theiler et al. 1992).  AAFT surrogates are used in testing
    the null hypothesis that the input series is correlated Gaussian
    noise transformed by a monotonic time-independent measuring
    function.

    Parameters
    ----------
    x : array
        1-D input array containg the time series.

    Returns
    -------
    y : array
        Surrogate series with (crudely) the same power spectrum and
        distribution.
    """
    # Generate uncorrelated Gaussian random numbers.
    y = _np.random.normal(size=x.shape[0])

    # Introduce correlations in the random numbers by rank ordering.
    y = _np.sort(y)[_np.argsort(_np.argsort(x, axis=0), axis=0)]
    
    y = _ft(y)

    return _np.sort(x, axis=0)[_np.argsort(_np.argsort(y, axis = 0), axis=0)]

def surrogate_IAAFT(x, maxiter=1000, atol=1e-8, rtol=1e-10):
    """Return iterative amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted (IAAFT) surrogates with
    the same power spectrum (to a very high accuracy) and distribution
    as the original data using an iterative scheme (Schreiber & Schmitz
    1996).

    Parameters
    ----------
    x : array
        1-D real input array of length N containing the time series.
    maxiter : int, optional (default = 1000)
        Maximum iterations to be performed while checking for
        convergence.  The scheme may converge before this number as
        well (see Notes).
    atol : float, optional (default = 1e-8)
        Absolute tolerance for checking convergence (see Notes).
    rtol : float, optional (default = 1e-10)
        Relative tolerance for checking convergence (see Notes).

    Returns
    -------
    y : array
        Surrogate series with (almost) the same power spectrum and
        distribution.
    i : int
        Number of iterations that have been performed.
    e : float
        Root-mean-square deviation (RMSD) between the absolute squares
        of the Fourier amplitudes of the surrogate series and that of
        the original series.

    Notes
    -----
    To check if the power spectrum has converged, we see if the absolute
    difference between the current (cerr) and previous (perr) RMSDs is
    within the limits set by the tolerance levels, i.e., if abs(cerr -
    perr) <= atol + rtol*perr.  This follows the convention used in
    the NumPy function numpy.allclose().

    Additionally, atol and rtol can be both set to zero in which
    case the iterations end only when the RMSD stops changing or when
    maxiter is reached.
    """
    # Calculate "true" Fourier amplitudes and sort the series.
    ampl = _np.abs(_np.fft.rfft(x, axis = 0))
    sort = _np.sort(x, axis = 0)

    # Previous and current error.
    perr, cerr = (-1, 1)

    # Start with a random permutation.
    t = _np.fft.rfft(_np.random.permutation(x))

    for i in range(maxiter):
        # Match power spectrum.
        s = _np.real(_np.fft.irfft(ampl * t / _np.abs(t), n=x.shape[0]))

        # Match distribution by rank ordering.
        y = sort[_np.argsort(_np.argsort(s, axis=0))]

        t = _np.fft.rfft(y)
        cerr = _np.sqrt(_np.mean((ampl ** 2 - _np.abs(t) ** 2) ** 2))

        # Check convergence.
        if abs(cerr - perr) <= atol + rtol * abs(perr):
            break
        else:
            perr = cerr

    # Normalize error w.r.t. mean of the "true" power spectrum.
    return y, i, cerr / _np.mean(ampl ** 2)

#%%%
# AR process
def _fit_AR(x, maxlag=10):
    mdl = _smt.AR(x).fit(maxlag=maxlag, ic='aic', trend='nc')
    return(mdl.params)

def _sim_AR(alpha, n, ndisc):
    ar = _np.r_[1, -alpha]
    ma = _np.r_[1,0]
    x = _smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin = ndisc) 
    return(x)

def surrogate_ARMA(x, maxlag = 30, ndisc = 1000, estimate=True):
    if estimate:
        est_lag = _smt.AR(x).select_order(maxlag=maxlag, ic='aic', trend='c')
        print(est_lag)
    else:
        est_lag = maxlag
    
    mean_x = _np.mean(x)
    std_x = _np.std(x)
    x = (x- mean_x)/std_x
    
    alpha = _fit_AR(x, est_lag)
    n = len(x)
    x_surr = _sim_AR(alpha, n, ndisc=ndisc)
    x_surr = (x_surr - _np.mean(x_surr))/_np.std(x_surr)
    x_surr = x_surr*std_x + mean_x
    return(x_surr)

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

def compare_channels(function, signal_1, signal_2=None, channels=None, 
                     diag_only=False, gen_surrogates=False, **kwargs):
    """
    Computes a matrix of a signal comparison metric between the 
    channel pairs of two signals.

    Parameters
    ----------
    function : callable
        Function to compute the comparison metric between two signals.
        Should accept two signals and optional keyword arguments.
    signal_1 : xarray.DataArray
        First input signal with dimensions 'channel' and 'component'.
    signal_2 : xarray.DataArray, optional
        Second input signal. If not provided, `signal_1` is used twice.
    channels : array-like, optional
        Channels to compare. If not provided, all channels are used.
    diag_only : boolean, optional
        Whether to only compute the metric between the same channels.
        Default: False
    **kwargs : dict
        Additional keyword arguments passed to the `function`.

    Returns
    -------
    numpy.ndarray
        Correlation matrix with shape (len(channels), len(channels), n_components),
        where n_components is the number of components in the signals.
    """
    
    if signal_2 is None:
        #TODO: implement surrogates
        if gen_surrogates:
            pass
        signal_2 = signal_1
        idx_offset = 1 #if no signal_2, do not compute the metric for same channels
    else:
        
        #check dims
        shape_1 = list(signal_1.sizes.values())
        shape_2 = list(signal_1.sizes.values())
        
        #TODO signal_1 and signal_2 might have a different number of channels
        for i,j in zip(shape_1, shape_2):
            assert i == j, "Sizes are not the same"
        idx_offset = 0 #if signal_2 is given, compute the metric for same channels
    
    if channels is None:
        channels = _np.arange(signal_1.sizes['channel'])
    
    n_components = signal_1.sizes['component']
    corr_mat = _np.nan*_np.ones(shape=(len(channels), len(channels), n_components))
    
    for i_comp in range(n_components):
        
        for i_ch in _np.arange(len(channels)):
            ch_1 = channels[i_ch]
            s_1 = signal_1.isel({'channel': [ch_1], 'component': [i_comp]})
            
            if diag_only:
                inner_max = i_ch+1
                idx_offset = 0
            else:
                inner_max = len(channels)
            
            for j_ch in _np.arange(i_ch+idx_offset, inner_max):
                print(i_ch, j_ch)
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

def wavelet_coherence(W1, W2, wavelet_object, **kwargs):
    '''
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
    
    coef1 = W1.p.main_signal.values[:,:,0,0]
    coef2 = W2.p.main_signal.values[:,:,0,0]
    coef12 = coef1 * coef2.conj()
    
    scales = wavelet_object._params['scales']
    nNotes = wavelet_object._params['nNotes']
    scaleMatrix = _np.ones([1, coef1.shape[1]]) * scales[:, None]
    
    coef1 = _np.abs(coef1)**2 / scaleMatrix
    coef2 = _np.abs(coef2)**2 / scaleMatrix
    coef12 = coef12    / scaleMatrix
    
    S1 = smooth( coef1, scales, nNotes)
    S2 = smooth( coef2, scales, nNotes)
    S12 = smooth(coef12, scales, nNotes)
    
    WC = abs(S12)**2 / (S1*S2)
    
    WC = wavelet_object._compute_coi(WC)
    WC_out = _np.nanmean(WC)
    return(WC_out)
    
    
    