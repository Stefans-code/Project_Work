from pyphysio.loaders import load_nirx2

nirs_A = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/hyper/pilot/TN002/TN002_NC/tn002fa_001') 
nirs_B = load_nirx2('/home/bizzego/UniTn/data/fnirs_technical_validation/hyper/pilot/TN002/TN002_NC/tn002fb_001') 

#%%
from pynirs.plot_probe import plot_probe

plot_probe(nirs_A.p.main_signal.attrs)
plot_probe(nirs_B.p.main_signal.attrs)

#%%
from pyphysio.specialized.fnirs import Raw2Oxy

hb_A = Raw2Oxy()(nirs_A)
hb_B = Raw2Oxy()(nirs_B)

hb_A.p.plot(sharey=False)
hb_B.p.plot(sharey=False)