import numpy as np
import numpy as _np
import pyphysio as ph
import matplotlib.pyplot as plt
# plt.ioff()

from test_utils import generate_evenly, info

np.random.seed(10)
fsamp = 10
signal = generate_evenly((1000, 4, 3), fsamp)

algorithms = [ph.Mean(name='ind')]#,

label = np.zeros(1000)
label[250:500] = 1
label[600:800] = 2

label = ph.Signal(label, fsamp)

segmenter = ph.FixedSegments(10, 20, timeline=label, reference = signal, drop_mixed=False)

#%%
fmap_results = ph.fmap(segmenter, algorithms, signal)
df = ph.indicators2df(fmap_results)

#%%
segmenter = ph.LabelSegments(timeline=label)
fmap_results = ph.fmap(segmenter, algorithms, signal)
df = ph.indicators2df(fmap_results)

#%%
segmenter = ph.RandomFixedSegments(10, 5, reference=signal, timeline=label)
fmap_results = ph.fmap(segmenter, algorithms, signal)
df = ph.indicators2df(fmap_results)

#%%
segmenter = ph.CustomSegments([0, 30, 45], [10, 50, 47], timeline=label)
fmap_results = ph.fmap(segmenter, algorithms, signal)
df = ph.indicators2df(fmap_results)

#%%
signal.plot()
label.plot()
fmap_results['ind'].plot()

