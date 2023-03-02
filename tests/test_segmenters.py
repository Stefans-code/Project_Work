from pyphysio.signal import create_signal
import numpy as np
from pyphysio.indicators import td, fd
import pyphysio.segmenters as segm

n_ch = 3
n_cp = 2
data = np.random.uniform(size = (10000, n_ch, n_cp)) + np.random.uniform(0, 10, size = (1,n_ch,n_cp))
sampling_freq = 1000
signal = create_signal(data, sampling_freq=sampling_freq, name = 'random')


#%%
x = np.zeros(10000)
x[2500:3000] = 1 
x[5000:8000] = 2

stim = create_signal(x, sampling_freq=1000, name = 'stimulus')

#%%
segmenter = segm.LabelSegments(timeline=stim, drop_mixed=False, drop_cut=False)
result = segm.fmap(segmenter, [td.Mean(), td.StDev()], signal)

result = segm.fmap(segmenter, [fd.PowerInBand(10, 200, 'welch')], signal)

#%%
segmenter = segm.FixedSegments(0.5, 2, timeline=stim, drop_mixed=False, drop_cut=False)
result = segm.fmap(segmenter, [td.Mean(), td.StDev()], signal)

#%%
segmenter = segm.RandomFixedSegments(10, 2, timeline=stim, drop_mixed=False, drop_cut=False)
result = segm.fmap(segmenter, [td.Mean(), td.StDev()], signal)