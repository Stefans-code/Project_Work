from pyphysio.loaders import load_nirx2
import numpy as _np
import pyphysio.artefacts as artefacts
from pyphysio.specialized.fnirs import Raw2Oxy, NegativeCorrelationFilter, \
    plot_probe, get_ss_ls_channels, get_near_channels, PCAFilter, RegressShortSeparation, \
        ComputeClusters
from pyphysio.loaders import SDto1darray
import pyphysio.filters as filters
from pyphysio.specialized.fnirs import Raw2Oxy

nirs = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/2022-09-13_001')

get_ss_ls_channels(nirs, max_dist=3.1) ##so to have ss
get_near_channels(nirs, ch_target=1)
# plot_probe(nirs)
# plt.close('all')

pcafilt = PCAFilter()
res = pcafilt(nirs)
#%%

regshort = RegressShortSeparation(max_dist=3.1) ##so to have ss
res = regshort(nirs)

cluster_compute = ComputeClusters(clusters=[[0,1,2,3], [4,5,6,7], [8,9,10,11]],
                                  mode='pca')
clusters = cluster_compute(nirs)

MA = artefacts.DetectMA()(nirs)
nirs['MA'] = MA
nirs_noMA = artefacts.MARA()(nirs, scheduler='single-threaded')
nirs_noMA = nirs_noMA.drop_vars('MA')
nirs_wav = artefacts.WaveletFilter()(nirs_noMA)

hb = Raw2Oxy()(nirs_wav)
