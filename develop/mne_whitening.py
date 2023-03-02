import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

def _yule_walker(X, order=1):
    """Compute Yule-Walker (adapted from statsmodels).
    Operates in-place.
    """
    from scipy import linalg
    assert X.ndim == 2
    denom = X.shape[-1] - np.arange(order + 1)
    r = np.zeros(order + 1, np.float64)
    for di, d in enumerate(X):
        d -= d.mean()
        r[0] += np.dot(d, d)
        for k in range(1, order + 1):
            r[k] += np.dot(d[0:-k], d[k:])
    r /= denom * len(X)
    rho = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])
    sigmasq = r[0] - (r[1:] * rho).sum()
    return rho, np.sqrt(sigmasq)

def fit_iir_model_raw(data, order=2):
    print(data.shape)
    coeffs, _ = _yule_walker(data, order=order)
    return np.array([1.]), np.concatenate(([1.], -coeffs))

#%
data_path = sample.data_path()
raw_fname = f'{data_path}/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname)
order = 10  # define model order

data = raw[[0],:][0] 
print(data.shape)


#%%
data = np.loadtxt('/home/bizzego/tmp/nirs.txt')
data = data.reshape((1,-1)) / 1000000000000

#%%
# Estimate AR models on raw data
b, a = fit_iir_model_raw(data, order=order)

#%%
d = data.ravel()
d = d.ravel()  # make flat vector
innovation = signal.convolve(d, a, 'valid')
d_ = signal.lfilter(b, a, innovation)  # regenerate the signal
d_ = np.r_[d_[0] * np.ones(order), d_]  # dummy samples to keep signal length

#%%
plt.close('all')
plt.figure()
plt.plot(data.ravel(), label='signal')
plt.plot(d_, label='regenerated signal')
plt.legend()

plt.figure()
plt.psd(d, Fs=raw.info['sfreq'], NFFT=2048)
plt.psd(innovation, Fs=raw.info['sfreq'], NFFT=2048)
plt.psd(d_, Fs=raw.info['sfreq'], NFFT=2048, linestyle='--')
plt.legend(('Signal', 'Innovation', 'Regenerated signal'))
plt.show()

#%%


'''
def fit_iir_model_raw(raw, order=2, picks=None, tmin=None, tmax=None,
                      verbose=None):
    r"""Fit an AR model to raw data and creates the corresponding IIR filter.
    The computed filter is fitted to data from all of the picked channels,
    with frequency response given by the standard IIR formula:
    .. math::
        H(e^{jw}) = \frac{1}{a[0] + a[1]e^{-jw} + ... + a[n]e^{-jnw}}
    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    order : int
        Order of the FIR filter.
    %(picks_good_data)s
    tmin : float
        The beginning of time interval in seconds.
    tmax : float
        The end of time interval in seconds.
    %(verbose)s
    Returns
    -------
    b : ndarray
        Numerator filter coefficients.
    a : ndarray
        Denominator filter coefficients.
    """
    start, stop = None, None
    if tmin is not None:
        start = raw.time_as_index(tmin)[0]
    if tmax is not None:
        stop = raw.time_as_index(tmax)[0] + 1
    picks = _picks_to_idx(raw.info, picks)
    data = raw[picks, start:stop][0]


    plt.plot(data.ravel())
    # rescale data to similar levels
    picks_list = _picks_by_type(pick_info(raw.info, picks))
    scalings = _handle_default('scalings_cov_rank', None)
    print(scalings)
    _apply_scaling_array(data, picks_list=picks_list, scalings=scalings)
    
    plt.plot(data.ravel())
    # do the fitting
    coeffs, _ = _yule_walker(data, order=order)
    return np.array([1.]), np.concatenate(([1.], -coeffs))
'''