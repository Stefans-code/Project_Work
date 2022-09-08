from pyphysio.loaders import load_nirx
import pyphysio.artefacts as artefacts
from pyphysio.specialized.fnirs import Raw2Oxy
#%
nirs = load_nirx('/home/bizzego/UniTn/data/fnirs_sexism/original/F02_2', False)

# nirs_noMA = artefacts.MARA()(nirs, scheduler='single-threaded')

nirs_wav = artefacts.WaveletFilter()(nirs)

#%%
hb = Raw2Oxy()(nirs_wav)

import pyphysio.filters as filters

hb_ = filters.IIRFilter(0.2, 0.01)(hb)
