import bioread
import numpy as np
import pyphysio as ph
import os

datadir = '/home/bizzego/Downloads'
outdir = '/home/bizzego/tmp/cecilia'

target_signal = 'ECG'

subject = 'Subj1_empathy.acq'

#%%
subname = subject.split('.')[0]
data = bioread.read_file(os.path.join(datadir, subject))

amp_digital_channels = 5

channel_labels = [data.channels[i].name for i in range(len(data.channels))]

i_target = None
i_digital = []
for i_ch, label_ch in enumerate(channel_labels):
    if target_signal in label_ch:
        i_target = i_ch
    if 'Digital' in label_ch:
        i_digital.append(i_ch)

if i_target is not None:
    signal = data.channels[i_target]
    fsamp = signal.samples_per_second
    values = signal.data

signal = ph.EvenlySignal(values, fsamp)    
signal.to_pickle(os.path.join(outdir, f'{subname}_ECG.pkl'))

if len(i_digital)>0:
    #for each channel/bit
    #add the bits considering the exponential associated to each bit position
    trg = np.zeros(len(values))
    exp_digital = range(len(i_digital))
    for (i_ch, exp) in zip(i_digital, exp_digital): 
        trg = trg+ (data.channels[i_ch].data/amp_digital_channels)*(2**exp)
    
    trg = ph.EvenlySignal(trg, fsamp)
    trg.to_pickle(os.path.join(outdir, f'{subname}_TRG.pkl'))