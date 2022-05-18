import bioread
import numpy as np

datafile = '/home/bizzego/Downloads/Subj1_empathy.acq'

data = bioread.read_file(datafile)

amp_digital_channels = 5
#%%
target_signal = 'ECG'

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
    
# if len(i_digital)>0:

#for each channel/bit
#add the bits considering the exponential associated to each bit position
trg = np.zeros(len(values))
exp_digital = range(len(i_digital))
for (i_ch, exp) in zip(i_digital, exp_digital): 
    trg = trg+ (data.channels[i_ch].data/amp_digital_channels)*(2**exp)

# create pkl and save
# trg = ph.EvenlySignal(values = trg, sampling_freq = fsamp, signal_type = 'trg', start_time = 0)
# trg.to_pickle(f'{outdir}/trg/{sub}.pkl')