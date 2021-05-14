import pyphysio as ph
import numpy as np

signal = ph.EvenlySignal(np.random.uniform(size=(1000, 5, 3)), 10)
# signal.plot()
#%%
signal[100:].shape #slice(100, None, None)
signal[100:, 4:, 1].shape #(slice(100, None, None), 0)

#%%

# class NNumpy(np.ndarray):
    
#     def __setitem__(self, item):
#         print(type(item))
#         return super().__setitem__(item)
    
#     def __getitem__(self, item):
#         print(type(item))
#         return super().__getitem__(item)
    
# s = NNumpy(10)
