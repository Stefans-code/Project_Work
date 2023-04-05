# from https://github.com/mne-tools/mne-python/blob/maint/1.1/mne/time_frequency/ar.py#L33-L78
# and  https://mne.tools/1.1/auto_examples/time_frequency/temporal_whitening.html

import numpy as np
from scipy import linalg
from scipy import signal
from pyphysio.specialized.fnirs import load_xrnirs
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

def yule_walker(X, order=1):
    """Compute Yule-Walker (adapted from statsmodels).
    Operates in-place.
    """

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
    
    return np.array([1.]), np.concatenate(([1.], -rho))


#%%
nirs= load_xrnirs('/home/bizzego/UniTn/data/RP/SG/Signals/hb_processed/RP174/RP/B_clusters')
X = nirs.p.main_signal.values[:,[0],0].T


#%%

X = (X - np.mean(X))/np.std(X)
X = X - X[0, 0]


# res = AutoReg(X, lags = 10).fit()
rho, sigma = sm.regression.yule_walker(X.ravel(), order=30, method="mle")
a_ = np.concatenate(([1.], -rho))

#%%
b, a = yule_walker(X, order=30)

#%%
innovation = signal.convolve(X.ravel(), a, 'valid')
d = signal.lfilter(b, a, innovation)  # regenerate the signal

#%%
plt.plot(X.T)
plt.plot(d[:50])