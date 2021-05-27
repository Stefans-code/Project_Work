import numpy as np
import numpy as _np
import pyphysio as ph
import matplotlib.pyplot as plt
# plt.ioff()
# from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells

from test_utils import generate_evenly, info_evenly, check

np.random.seed(10)
fsamp = 10

#%
signal = generate_evenly((1000,2), fsamp)

# # #%
# K = ph.Kurtosis([0,1], name='K')(signal)
# E = ph.Entropy([0,1], name='E')(signal)

# print(K[0].shape)
# print(E[0].shape)
# m = ph.Mean()(signal)

##%%
# indicators = [ph.Kurtosis([0,1], name='K'),]

# results = ph.fmap(ph.FixedSegments(5, drop_cut=False), indicators, signal)

# print(results['K'][0].shape)

# #%%
#%%
sqi = [ph.Kurtosis([0,1], name='K'),
       ph.Entropy([0,1], name='E'),
       ph.DerivativeEnergy([0,1], name='DE')]

segmenter = ph.FixedSegments(5, drop_cut=False, drop_mixed=False)
compute_global = True
ratio = 0.9

signal = ph.ComputeQuality(sqi,
                           segmenter=segmenter,
                           compute_global = compute_global,
                           ratio=ratio)(signal)
info = signal.get_info()
print(info['sqi']['K'].shape)
print(info['good'].shape)
type(info['sqi']['K'])
type(info['good'])

s = signal[:, 0]
info_s = s.get_info()
print(info_s['sqi']['K'].shape)
print(info_s['good'].shape)
type(info_s['sqi']['K'])
type(info_s['good'])

info = signal.get_info()
print(info['sqi']['K'].shape)


#%%
# print(qc)
#%
#%
print(signal.get_info()['sqi']['K'].shape)
print(signal.get_info()['good'].shape)
# # print(signal.get_info()['good'])
signal.get_info()['sqi']['K'].plot('.')