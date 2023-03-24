from pyphysio.loaders import load_nirx, load_nirx2
import numpy as _np
#%
# nirs = load_nirx('/home/bizzego/UniTn/data/fnirs_sexism/original/F02_2')
# nirs = nirs.p.process_na('impute')

nirs2 = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/2022-09-13_001')
nirs3 = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/2022-09-13_001_var')

#%%
import pyphysio.artefacts as artefacts

nirs_noMA = artefacts.MARA()(nirs2, scheduler='single-threaded')
nirs_wav = artefacts.WaveletFilter()(nirs_noMA)

#%%
from pyphysio.specialized.fnirs import Raw2Oxy, NegativeCorrelationFilter, SDto1darray

hb = Raw2Oxy()(nirs_wav)
hb = NegativeCorrelationFilter()(hb)

import pyphysio.filters as filters

hb = filters.FIRFilter([0.01, 0.2], [0.001, 0.4])(hb)

#%%
hb = SDto1darray(hb)
hb.to_netcdf('/home/bizzego/tmp/nirs')


#%%
from pynirs.plot_probe import plot_probe
from pyphysio.specialized.fnirs import Raw2Oxy

plot_probe(nirs2.p.main_signal.attrs)
plot_probe(nirs3.p.main_signal.attrs)

hb2 = Raw2Oxy()(nirs2)
hb3 = Raw2Oxy()(nirs3)

hb2.p.plot()
hb3.p.plot()


#%%
import matplotlib.pyplot as plt
SD2 = nirs2.p.main_signal.attrs
SD3 = nirs3.p.main_signal.attrs

f, axes = plt.subplots(1,2)
axes[0].axis('auto')
axes[0].plot(SD2['SrcPos'][:,2], SD2['SrcPos'][:,0], 'o')
axes[1].axis('auto')
axes[1].plot(SD3['SrcPos'][:,2], SD3['SrcPos'][:,0], 'o')
