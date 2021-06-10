import pyphysio as ph
import numpy as np

#signal = ph.EvenlySignal(np.random.uniform(size=1000), 10)

signal = ph.Signal(np.random.uniform(size=(1000,3,5)), 10)

#%%
from pynirs.loaders import load_nirx

DATADIR = '/home/bizzego/UniTn/data/fnirs_sexism/original/2021-02-05_002'

signal = load_nirx(DATADIR, False)

#%%
type(signal)

type(signal+3)

type(signal*3)

type(signal/np.nanmean(signal))
