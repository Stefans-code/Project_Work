import numpy as np
import numpy as _np
import pyphysio as ph
import matplotlib.pyplot as plt
# plt.ioff()
# from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells

from test_utils import generate_evenly, info

np.random.seed(10)
fsamp = 10

#%
signal = generate_evenly((1000,5,3), fsamp)

#%%
# # #%
K = ph.Kurtosis([0,1], name='K')(signal)
E = ph.Entropy([0,1], name='E')(signal)
DE = ph.DerivativeEnergy([0,1], name='DE')(signal)

#%%
sqi = [ph.Kurtosis([0,1], name='K'),
       ph.Entropy([0,1], name='E'),
       ph.DerivativeEnergy([0,1], name='DE')]

segmenter = ph.FixedSegments(5, drop_cut=False, drop_mixed=False)
compute_global = True
ratio = 0.9

signal_sqi = ph.ComputeQuality(sqi,
                               segmenter=segmenter,
                               compute_global = compute_global,
                               ratio=ratio)(signal)

#%%
info = signal.get_info()

print(info['sqi']['K'].shape)
print(info['good'].shape)
print(type(info['sqi']['K']))
print(type(info['good']))

#%% test segmentation of sqi
s = signal[:, 0]
info_s = s.get_info()
print(info_s['sqi']['K'].shape)
print(info_s['good'].shape)
print(type(info_s['sqi']['K']))
print(type(info_s['good']))

#%%
info = signal.get_info()

print(info['sqi']['K'].shape)
print(info['good'].shape)
print(type(info['sqi']['K']))
print(type(info['good']))
