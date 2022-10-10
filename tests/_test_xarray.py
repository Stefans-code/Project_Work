from pyphysio.signal import create_signal
import numpy as np
from pyphysio.processing import Algorithm
import matplotlib.pyplot as plt

import pyphysio.processing.filters as filt
import pyphysio.indicators.timedomain as td
import pyphysio.indicators.frequencydomain as fd
import pyphysio.processing.tools as tools
import pyphysio.sqi.sqi as sqi

def test_general_functions():
    n_ch = 1
    n_cp = 1
    data = np.random.uniform(size = (10000, n_ch, n_cp)) + np.random.uniform(0, 10, size = (1,n_ch,n_cp))
    sampling_freq = 1000
    signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')
        
    #%%
    signal_mean = td.Mean()(signal)
    
    signal_out = filt.Normalize(norm_method = 'min')(signal)
    
    signal_out = filt.KalmanFilter(200, 100)(signal)
    
    #%%
    signal_std = td.StDev()(signal)
    signal_sum = td.Sum()(signal)
    signal_auc = td.AUC()(signal)
    signal_detauc = td.DetrendedAUC()(signal)
    
    #%%
    signal_out = filt.Normalize(norm_method='maxmin')(signal)
    # plt.plot(signal.p.main_signal.values[:,0,0])
    # plt.plot(signal_out.p.main_signal.values[:,0,0])
    
    signal_out = filt.IIRFilter(200, 350)(signal)
    # plt.plot(signal.valuess[:,0,0])
    # plt.plot(signal_out.values[:,0,0])
    
    signal_out = filt.NotchFilter(200)(signal)
    # plt.plot(signal.p.get_values()[:,0,0])
    # plt.plot(signal_out.p.get_values()[:,0,0])
    
    
    signal_out = filt.FIRFilter(100, 200)(signal)
    # plt.plot(signal.values[:,0,0])
    # plt.plot(signal_out.values[:,0,0])
    
    #%%
    # fig, axes = plt.subplots(4,4, sharex=True)
    
    for i_row, R in enumerate([0.1, 10, 100, 1000]):
        for i_col, ratio in enumerate([1.1, 10, 100, 1000]):
            signal_out = filt.KalmanFilter(R, ratio)(signal)
            # axes[i_row, i_col].plot(signal.p.main_signal.values[:,0,0])
            # axes[i_row, i_col].plot(signal_out.p.main_signal.values[:,0,0])
    
    #%%
    signal_out = filt.RemoveSpikes()(signal)
    # plt.plot(signal.p.get_values()[:,0,0])
    # plt.plot(signal_out.p.get_values()[:,0,0])
    
    
    signal_out = filt.ConvolutionalFilter('rect', 5)(signal)
    # plt.plot(signal.values[:,0,0])
    # plt.plot(signal_out.values[:,0,0])
    
    
    signal_out = filt.DeConvolutionalFilter([0.2, 0.5, 0.3])(signal)
    # plt.plot(signal.values[:,0,0])
    # plt.plot(signal_out.values[:,0,0])
    
    
    #%% test some tools
    sig_out = tools.PeakDetection(0.02)(signal)
    sig_out = tools.Maxima(method='windowing', win_len=0.25, win_step=0.05)(signal)
    psd = tools.PSD('welch')(signal)
    
    #%%
    res = sqi.SpectralPowerRatio(threshold=[0.5, 0.8], method = 'welch')(signal)
