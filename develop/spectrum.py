# https://stackoverflow.com/questions/52723383/difference-between-scipy-periodogram-and-self-implemented-power-spectral-density
# https://stackoverflow.com/questions/52690632/analyzing-seasonality-of-google-trend-time-series-using-fft/52691914#52691914
# https://stackoverflow.com/questions/22338415/scipy-periodogram-terminology-confusion
# https://stackoverflow.com/questions/57828899/prefactors-computing-psd-of-a-signal-with-numpy-fft-vs-scipy-signal-welch

# https://notebook.community/pxcandeias/py-notebooks/DSP_FFT_psd#Fourier-transform

#%%

import numpy as np
import scipy.signal as sp
from numpy.fft import fft, fftfreq

import matplotlib.pyplot as plt

def abs2(x):
    return x.real**2 + x.imag**2

#%% signal settings
N = 10000

frequencies = [3, 20, 30]
amplitudes = [8, 1.5, 4]

fig, axes = plt.subplots(3,1, sharex=True)

#%
for fs in [100, 500, 1000]:
    for N in [10000]:# [fs, fs*2, fs*10]:
        #% create signal
        
        time = np.arange(N)/fs
        x = np.zeros(N)
        for (freq, amp) in zip(frequencies, amplitudes):
            component = amp*np.sin(2*np.pi*freq*time)
            x += component
            
        
        
        #% compute density
        f_p, Pxx_den_p = sp.periodogram(x, fs)
        Pxx_den_p = Pxx_den_p / len(Pxx_den_p)
        # f_w, Pxx_den_w = sp.welch(x, fs, nperseg=2048)
        
        idx_band = np.where((f_p >= 1) & (f_p <= 4))[0]
        P_den = np.sum(Pxx_den_p[idx_band]*fs)
        
        axes[0].plot(f_p, Pxx_den_p)
        # axes[0].plot(f_w, Pxx_den_w)
        
        axes[0].set_xlabel('frequency [Hz]')
        
        axes[0].set_ylabel('PSD [V**2/Hz]')
        
        
        #% compute spectrum
        f_p, Pxx_spec_p = sp.periodogram(x, fs, scaling='spectrum')
        # f_w, Pxx_spec_w = sp.welch(x, fs, scaling='spectrum', nperseg=2048)
        
        idx_band = np.where((f_p >= 1) & (f_p <= 4))[0]
        
        P_spec = 2*np.sum(Pxx_spec_p[idx_band])
        print(P_den, P_spec)
        
        axes[1].plot(f_p, 2*Pxx_spec_p)
        # axes[1].plot(f_w, 2*Pxx_spec_w)
        
        axes[1].set_xlabel('frequency [Hz]')
        
        axes[1].set_ylabel('Linear spectrum [V RMS]')
    
    
        #compute fft
        fftx = fft(x)
        
        # Pxx_ = np.abs(fftx)**2 / N * (1/fs) # --> density
        Pxx_ = abs(fftx)**2 / N ** 2 # --> spectrum
        f = fftfreq( len(x), d = 1./fs ) 
        
        axes[2].plot(f, Pxx_)
        
        axes[2].set_xlabel('frequency [Hz]')
        
        axes[2].set_ylabel('Density')
        
    
plt.xlim(0, 50)


#%%
import pyphysio as ph
from pyphysio.signal import create_signal
from pyphysio.utils import PSD
from pyphysio.indicators import fd

#%% signal settings
N = 10000

frequencies = [3, 20, 30]
amplitudes = [8, 1.5, 4]

#%
for fs in [100, 500, 1000]:
    for N in [10000]:# [fs, fs*2, fs*10]:
        #% create signal
        
        time = np.arange(N)/fs
        x = np.zeros(N)
        for (freq, amp) in zip(frequencies, amplitudes):
            component = amp*np.sin(2*np.pi*freq*time)
            x += component
            
        #% compute density
        f_p, Pxx_den_p = sp.periodogram(x, fs, nfft=2048)
        Pxx_den_p = Pxx_den_p / len(Pxx_den_p)
        
        x = create_signal(x, sampling_freq=fs)
        
        psd = PSD('period', window='boxcar', scaling='density')(x)
        
        f_P = psd['freq'].values
        P_P = psd.p.main_signal.values.ravel()
        
        # f_w, Pxx_den_w = sp.welch(x, fs, nperseg=2048)
        
        idx_band = np.where((f_p >= 1) & (f_p <= 4))[0]
        P_den = np.sum(Pxx_den_p[idx_band]*fs)
        
        print(P_den, fd.PowerInBand(1, 4, 'period', window='boxcar', nfft=2048)(x).p.main_signal.values[0])
        