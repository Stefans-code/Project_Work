import numpy as _np
import pyphysio as ph
from pyphysio.loaders.load_nirx import load_nirx
import matplotlib.pyplot as plt
import pyphysio.fnirs.artifacts as artifacts
from pyphysio.fnirs.convert import Raw2Oxy
#%
nirs = load_nirx('/home/bizzego/UniTn/data/fnirs_sexism/original/F02_2', False)

nirs_noMA = artifacts.MARA()(nirs, scheduler='single-threaded')

nirs_wav = artifacts.WaveletFilter()(nirs)

#%%
hb = Raw2Oxy()(nirs_wav)
