from pyphysio.signal import create_signal
import numpy as np
import pyphysio.indicators.timedomain as td
import pyphysio.indicators.frequencydomain as fd
import pyphysio.segmenters as segm

n_ch = 3
n_cp = 2
data = np.random.uniform(size = (10000, n_ch, n_cp)) + np.random.uniform(0, 10, size = (1,n_ch,n_cp))
sampling_freq = 1000
signal = create_signal(data, sampling_freq=sampling_freq)


#%%
x = np.zeros(shape=(10000,1,1))
x[2500:3000, 0,0] = 1 
x[5000:8000, 0,0] = 2

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