import numpy as _np
import numpy as np
from pyphysio.signal import create_signal
import xarray as xr
import pyphysio.specialized.eda as eda_tools
import pyphysio.filters as flt
import pyphysio.utils as utils
from pyphysio import TestData
import matplotlib.pyplot as plt
import pyphysio as ph

#%%
eda_data = TestData().eda()

signal = create_signal(eda_data, sampling_freq=2048)

#%%
# signal = ph.load('/home/bizzego/UniTn/didattica/neurotechnology/datasets/DEAP/signals/eda/s21')
signal = flt.ConvolutionalFilter('rect', 2)(signal)

# signal /=1000

#%%
# signal = ph.load('/home/bizzego/Downloads/recognitiveneuroscienceandneurotechnology/P6_Base2')

# signal = ph.create_signal(signal.p.get_values(), sampling_freq=10)
#%%
# signal = flt.ConvolutionalFilter('rect', 1, normalize=True)(signal)
# signal.p.plot()

#%%
fsamp = 8
signal = signal.p.resample(fsamp)
# driver.p.plot()

#%%
driver = eda_tools.DriverEstim(t1=0.75, t2=2, optim=True, amplitude=0.01)(signal)
driver = flt.ConvolutionalFilter('rect', 1, normalize=True)(driver)

#%%
# driver.p.segment_time(5, 7).p.plot('.')
phasic = eda_tools.PhasicEstim(0.01, win_pre=5, win_post=5, polyfit=True)(driver)#.p.segment_time(5, 7))

# phasic.p.plot('.')

#%%
tonic = eda_tools.PhasicEstim(0.01, win_pre=2, win_post=2, polyfit=True, return_phasic=False)(driver)

#%%
driver.p.plot()
tonic.p.plot()
phasic.p.plot()
plt.show()


#%%
import pyphysio.indicators.peaks as pk

pks_max = pk.PeaksMax(delta=0.005)(phasic)

pks_num = pk.PeaksNum(delta=0.005)(phasic)

dur_mean = pk.DurationMean(0.005, 2, 2)(phasic)

slope = pk.SlopeMin(0.005, 2, 2)(phasic)

#%%
phasic_indicators = eda_tools.preset_phasic(delta=0.005)

indicators = []
for ind in phasic_indicators:
    indicators.append(ind(phasic))
#%%
# #%%
# #%
# # fig, axes = plt.subplots(3,1,sharex=True)
# # plt.sca(axes[0])
# # signal.p.plot()

# # plt.sca(axes[1])
# # driver.p.plot()
# # tonic.p.plot()

# # plt.sca(axes[2])
# # phasic.p.plot()

# #%%
# # #%%
# # results = []
# # for fsamp in [4,8,16]:
# #     for t1 in [0.007, 0.07, 0.7]:
# #         for t2 in [2,5,10]:
# #             driver_estim = eda_tools.DriverEstim(t1=t1, t2=t2, smooth=False)
# #             bateman = driver_estim._gen_bateman(fsamp)
# #             results.append({'fsamp': fsamp,
# #                             't1': t1,
# #                             't2': t2,
# #                             'max': np.max(bateman),
# #                             'sum': np.sum(bateman)/len(bateman),
# #                             'mean': np.mean(bateman),
# #                             'sum_norm': np.sum(bateman)*len(bateman)/fsamp})
# # import pandas as pd

# # results = pd.DataFrame(results)

# #%%
# # fig, axes = plt.subplots(4,3, sharex=True, sharey='col')
# # for i1, t1 in enumerate([0.007, 0.07, 0.7]):
# #     for i2, t2 in enumerate([2,5,10]):
# #         for fsamp in [4,8,16]:
# #             signal_ = signal.p.resample(fsamp)
# #             driver_estim = eda_tools.DriverEstim(t1=t1, t2=t2, rescale_driver=True)
# #             # bateman = driver_estim._gen_bateman(fsamp)
# #             driver = driver_estim(signal_)
# #             driver_values = driver.p.get_values().ravel()
# #             # driver_values = driver_values*np.max(bateman)*fsamp
# #             driver_values -= driver_values[0]
            
# #             t = np.arange(len(driver_values))/fsamp
# #             axes[i1, i2].plot(t, driver_values)
# #             axes[i1, i2].grid()
# #         axes[i1, i2].set_title(f'{t1} - {t2}')
# # plt.sca(axes[3,0])
# # signal_.p.plot()

# # axes[0,0].set_xlim(87, 93)

# #%%            
        
# # for it, fsamp in enumerate([4, 8, 16]):
# #     driver_estim = eda_tools.DriverEstim(t1=0.25, t2=5, smooth=False)
# #     bateman = driver_estim._gen_bateman(fsamp)
# #     driver = driver_estim(signal)
# #     driver_values = driver.p.get_values().ravel()
# #     driver_values = driver_values*np.sum(bateman)/len(bateman)
# #     axes[it].plot(driver_values)
    
# #%%
# # print(np.max(bateman))
# # amplitude = 0.01
# # l = 2*len(bateman)

# # bateman_target = _np.zeros(l)

# # # driver_target[l//2] = 1



# # bateman_target[int(len(bateman)/2):int(len(bateman)/2)+len(bateman)] = amplitude * bateman / _np.max(bateman)

# # # bateman_target[int(len(bateman)/2):int(len(bateman)/2)+len(bateman)] = bateman / _np.sum(bateman)

# # fft_signal = _np.fft.fft(bateman_target, n=l)
# # fft_irf = _np.fft.fft(bateman, n=l)
# # out = abs(_np.fft.ifft(fft_signal / fft_irf))
# # out[0] = out[1]
# # out[-1] = out[-2]

# # factor = np.max(out)
# # # print(factor)


# # driver_values = driver.p.get_values()
# # driver_values = driver_values/factor
# # driver = create_signal(driver_values, sampling_freq = fsamp)

# #%
# # #%%
# # fig, axes = plt.subplots(3,1, sharex=True)
# # plt.sca(axes[0])
# # signal.p.plot()
# # # bateman_target.p.plot()

# # plt.sca(axes[1])
# # driver.p.plot()

# # plt.sca(axes[2])
# # driver_norm.p.plot()

# #%
# # fsamp = signal.p.get_sampling_freq()
# # driver = driver_norm



# #%%
# i_pre = []
# i_post = []

# for idx_max in maxp:
#     idx_pre = idx_max-1
#     s_pre = dd[idx_pre]
#     while (s_pre>-0.5):
#         idx_pre -=1
#         s_pre = dd[idx_pre]
    
#     i_pre.append(idx_pre)
    
#     idx_post = idx_max+1
#     s_post = dd[idx_post]
#     while (s_post<-0.5):
#         idx_post +=1
#         s_post = dd[idx_post]
        
#     i_post.append(idx_post)

# axes[1].vlines(i_pre, 0.1, 0.25, 'g')
# axes[1].vlines(i_post, 0.1, 0.25, 'r')

# axes[2].vlines(i_pre, -1, 0, 'g')
# axes[2].vlines(i_post, -1, 0, 'r')        
# #%%        
        


#  # Linear interpolation to substitute the peaks
#  driver_no_peak = _np.copy(signal)
#  for I in range(len(idx_pre)):
#      i_st = idx_pre[I]
#      i_sp = idx_post[I]

#      if not _np.isnan(i_st) and not _np.isnan(i_sp):
#          idx_base = _np.arange(i_sp - i_st)
#          coeff = (signal[i_sp] - signal[i_st]) / len(idx_base)
#          driver_base = idx_base * coeff + signal[i_st]
#          driver_no_peak[i_st:i_sp] = driver_base

#  # generate the grid for the interpolation
#  idx_grid = _np.arange(0, len(driver_no_peak) - 1, grid_size * fsamp)
#  idx_grid = _np.r_[idx_grid, len(driver_no_peak) - 1]

#  driver_grid = _Signal(driver_no_peak[idx_grid], sampling_freq = fsamp, 
#                        start_time= signal.get_start_time(), info=signal.get_info(),
#                        x_values=idx_grid, x_type='indices')
#  tonic = driver_grid.fill(kind='cubic')

#  phasic = signal - tonic
# #%%

# t1 = 0.75
# t2 = 2.0

# signal_values = signal.p.get_values().ravel()
# fsamp = signal.p.get_sampling_freq()
# irf = _gen_bateman(fsamp, [t1, t2])



# driver = flt.DeConvolutionalFilter(irf, normalize=True, deconv_method='fft')(signal)

# #%%
# driver = flt.ConvolutionalFilter(irftype='gauss', win_len=_np.max([0.2, 1 / fsamp]) * 8, normalize=True)(driver)

# #%%
# # idx_max_bat = _np.argmax(bateman)bateman_target

# # # Prepare the input signal to avoid starting/ending peaks in the driver
# # bateman_first_half = bateman[0:idx_max_bat + 1]
# # bateman_first_half = signal_values[0] * (bateman_first_half - _np.min(bateman_first_half)) / (
# #     _np.max(bateman_first_half) - _np.min(bateman_first_half))

# # bateman_second_half = bateman[idx_max_bat:]
# # bateman_second_half = signal_values[-1] * (bateman_second_half - _np.min(bateman_second_half)) / (
# #     _np.max(bateman_second_half) - _np.min(bateman_second_half))

# # signal_in = _np.r_[bateman_first_half, signal_values, bateman_second_half]

# #%% deconvolution
# s = signal_values# - signal_values[0]

# irf = irf / (_np.sum(irf) * len(irf)/fsamp)

# l = len(s)

# fft_signal = _np.fft.fft(s, n=l)
# fft_irf = _np.fft.fft(irf, n=l)
# out = _np.fft.ifft(fft_signal / fft_irf)

# out[0] = out[1]
# out[-1] = out[-2]

# t = _np.arange(len(s))/fsamp
# plt.plot(t, s)
# plt.plot(t, out)

# #%%
# t_irf = _np.arange(len(irf))/fsamp
# plt.plot(t_irf, irf)


# #%%
# plt.plot(out)

# #%%
# # elif deconvolution_method == 'sps':
    
# from scipy.signal import deconvolve as _deconvolve

# s -=s.mean()
# irf -= irf.mean()
# out_dec, _ = _deconvolve(s, irf)


# #%%    
#     #fix size
#     #TODO half before, half after?
#     out = _np.ones(len(signal))*out_dec[-1]
#     out[:len(out_dec)] = out_dec
# else:
# #     print('Deconvolution method not implemented. Returning original signal.')
# #     out = s
    
# driver = driver[idx_max_bat + 1: idx_max_bat + len(signal)]

# # gaussian smoothing
# driver = _ConvolutionalFilter(irftype='gauss', win_len=_np.max([0.2, 1 / fsamp]) * 8, normalize=True)(driver)

# driver = _Signal(driver, sampling_freq=fsamp, start_time=signal.get_start_time(), info=signal.get_info())
