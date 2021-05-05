#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:14:01 2021

@author: bizzego
"""

import numpy as np
import pyphysio as ph

#%%
for i in range(2):
    
    signal = ph.EvenlySignal(np.random.normal(size=(1000,4,2)), 10)
    signal.plot(ncols=1)



#%%

signal_u = ph.UnevenlySignal(np.random.normal(size=1000), 10, x_values = np.arange(1000), x_type = 'indices')

signal_u.plot()


#%%
def normalize(x):
    return( (x - np.mean(x))/np.std(x))

#%%
import numpy as np
import pyphysio as ph

signal = ph.EvenlySignal(np.random.normal(size=(1000,4,2)), 10)
sig_ = ph.Normalize(norm_method='standard')(signal)


#%%
import pyphysio as ph
import numpy as np

label = np.zeros(1000)
label[250:300] = 1
label[600:800] = 2

label = ph.EvenlySignal(label, 10)

seg = ph.BaseSegmentation.SegmentsWithLabelSignal()
