from ._load_nirs import load_nirx2, load_nirx, load_snirf, load_xrnirs, SDto1darray
from ..signal import create_signal as _create_signal
#TODO: add modules for loading edf, physionet?
import numpy as _np
import pprint
import math



def info_xdf(datafile):
    import pyxdf
    
    # Load the XDF file
    data, header = pyxdf.load_xdf(datafile, verbose=True)
    
    # Extract stream names
    names = [stream["info"]["name"][0] for stream in data]
    print(names)




def load_xdf(datafile, fresamp=None, extract_SD=False):
    import pyxdf
    data, header = pyxdf.load_xdf(datafile, verbose=True)
    

    names = []
    for i_d, d in enumerate(data):
        info = d['info']
        names.append(info['name'][0])
        # print(i_d, 
        #       info['name'][0], 
        #       info['type'], 
        #       info['channel_count'])
    names = _np.array(names)


    #% nirs raw from nirs
    idx_nirs = _np.where(names == 'Aurora')[0]
    assert len(idx_nirs) > 0, "no Aurora stream"

    idx_nirs = idx_nirs[0]
    t_nirs = data[idx_nirs]['time_stamps']
    data_fnirs = data[idx_nirs]
    data_nirs = _np.array(data_fnirs['time_series'])
    fsamp = data_fnirs['info']['effective_srate']
        
     
    t0 = t_nirs[0]
    t_nirs = t_nirs - t0

    channels_info = data_fnirs['info']['desc'][0]['channels'][0]['channel']
    idx_raw = _np.where([ch['type'][0] == 'nirs_raw' for ch in channels_info])[0]
        
    n_channels = len(idx_raw)//2

    signal_values_w1 = _np.array(data_fnirs['time_series'][:, idx_raw[:n_channels]])
    signal_values_w2 = _np.array(data_fnirs['time_series'][:, idx_raw[n_channels:]])
        
    signal_values = _np.stack([signal_values_w1, signal_values_w2], axis=2)

    # pprint.pprint(signal_values_w2)

    nirs = _create_signal(signal_values, times=t_nirs)


    if extract_SD:

        #% Extract info from montage
        montage_info = data_fnirs['info']['desc'][0]['montage'][0] 
        # print(montage_info)
        sources_info = montage_info['optodes'][0]['sources'][0]['source'] 
        # pprint.pprint(sources_info)
        detectors_info = montage_info['optodes'][0]['detectors'][0]['detector'] # 
        # pprint(detectors_info)
        channels_info = data_fnirs['info']['desc'][0]['channels'][0]['channel'] 
        pprint.pprint(data_fnirs['info']['desc'][0].keys())  
        # pprint(len(channels_info))  # 181 
        # pprint.pprint(channels_info[0:10])


        #% create SD
        SD = {}

        # generating SD['Lambda']
        wavelengths = []
        nirs_info = data_fnirs['info']['desc'][0]['channels'][0]['channel']

        for entry in nirs_info:
            if 'wavelength' in entry and entry['wavelength']: # Check if 'wavelength' exists and is non-empty
                wavelengths.append(float(entry['wavelength'][0])) # Add the first element of 'wavelength' to the list and convert to float
                
        unique_wavelengths = list(set(wavelengths)) # get unique elements, remove duplicates       


        if len(unique_wavelengths) == 2:
            SD['Lambda'] = unique_wavelengths
            print(SD['Lambda'])  # Output the wavelengths
        else:
            print("Unexpected number of unique wavelengths:", len(unique_wavelengths))
            
            
        # generating SD['Channels']
        info_channels = []
        n_ch = data_fnirs['time_series'].shape[1] // 4
        # print(n_ch)





        srcPos=[]
        srcPos2d=[]
        detPos=[]
        detPos2d=[]
            
        sources_info = montage_info['optodes'][0]['sources'][0]['source']
        print("len sources_info: ", len(sources_info))
        for source in sources_info:
            label = int(source['label'][0])  # Convert label to integer
            # print(label)
            
            location = source['location'][0]
            x = float(location['x'][0])
            y = float(location['y'][0])
            z = float(location['z'][0])
            u = float(location['u'][0])
            v = float(location['v'][0])
            srcPos.append([f"{x / 10:.6f}", f"{y / 10:.6f}", f"{z / 10:.6f}"])
            srcPos2d.append([f"{u / 10:.6f}", f"{v / 10:.6f}"])
         
        # print(srcPos)
          
                
        detectors_info = montage_info['optodes'][0]['detectors'][0]['detector']
        for detector in detectors_info:
            label = int(detector['label'][0])  # Convert label to integer
            location = detector['location'][0]
            x = float(location['x'][0])
            y = float(location['y'][0])
            z = float(location['z'][0])
            u = float(location['u'][0])
            v = float(location['v'][0])
            detPos.append([f"{x / 10:.6f}", f"{y / 10:.6f}", f"{z / 10:.6f}"])
            detPos2d.append([f"{u / 10:.6f}", f"{v / 10:.6f}"])
        
            
        srcPos = _np.array(srcPos, dtype=float)
        srcPos2d = _np.array(srcPos2d, dtype=float)
        detPos = _np.array(detPos, dtype=float)
        detPos2d = _np.array(detPos2d, dtype=float)
        
        
        SD['SrcPos'] = []
        SD['SrcPos2D'] = []
        SD['DetPos'] = []
        SD['DetPos2D'] = []
        
        SD['SrcPos'].append(srcPos)
        SD['SrcPos2D'].append(srcPos2d)
        SD['DetPos'].append(detPos)
        SD['DetPos2D'].append(detPos2d)
        
        # pprint.pprint(SD['SrcPos'])
        # pprint.pprint(SD['SrcPos2D'])
        # pprint.pprint(SD['DetPos'])
        # pprint.pprint(SD['DetPos2D'])
            
        ch_dict = []  # To store channel information  
        for i_ch in range(n_ch):  # Iterate over all valid channels
            channel = channels_info[i_ch+1]
            
            # Extract source and detector indices from 'custom_name' (e.g., 'S4-D5')
            custom_name = channel['custom_name'][0]
            src_label, det_label = custom_name.split('-')
            idx_src = int(src_label[1:]) - 1  # Convert 'S#' to index (0-based)
            idx_det = int(det_label[1:]) - 1  # Convert 'D#' to index (0-based)
            
            # Get source and detector positions
            src_3d = list(map(float, srcPos[idx_src]))
            det_3d = list(map(float, detPos[idx_det]))
            src_2d = list(map(float, srcPos2d[idx_src]))
            det_2d = list(map(float, detPos2d[idx_det]))
            
            # Compute 3D distance
            distance = math.sqrt(sum((s - d)**2 for s, d in zip(src_3d, det_3d)))
            # Alla fine la "distance" presente nel canale è equivalente a quella calcolata a partire da S e D
            
            # Compute 2D distance
            distance2D = math.sqrt(sum((s - d)**2 for s, d in zip(src_2d, det_2d)))
            
            # Append channel information
            ch_dict.append([i_ch + 1, idx_src, idx_det, distance, distance2D])
        
        # pprint.pprint(ch_dict)

        ch_dict = [[int(row[0]), int(row[1]), int(row[2]), round(row[3], 8), round(row[4], 8)]
            for row in ch_dict
        ]
        
        pprint.pprint(ch_dict)
        
        
        
        
        SD['Channels'] = ch_dict
        SD['SpatialUnit'] = 'cm'
        SD['smpling_freq'] = fsamp
        SD['start_time'] = t_nirs[0]
        SD['sampling_freq'] = 'unevenly'
        nirs.p.main_signal.attrs = SD
    
    #resample to a fixed FSAMP
    nirs = nirs.p.resample(10)
        
    nirs = SDto1darray(nirs)
    print("SD:", SD['Lambda'])

    return(nirs)

def info_biopac(datafile):
    import bioread
    data = bioread.read_file(datafile)
    names = [ch.name for ch in data.channels]
    print(names)

def load_biopac(datafile, channel, trigger = False):
    import bioread
    data = bioread.read_file(datafile)
    if (trigger):
        channels = data.channels

        trigger = _np.zeros(data.channels[0].data.shape[0])
        for i, ch in enumerate(channel):
            digital_channel = channels[ch].data
            trigger = trigger + (2**i)*digital_channel
        
        values = trigger/5
        
    else:
        channels = data.channels
        channel = channels[channel]
        values = channel.data
    fsamp = data.samples_per_second
    
    signal = _create_signal(values, sampling_freq=fsamp)
    return(signal)

def load_text(datafile, data_col=0, sampling_freq=None, time_col=None, 
              sep=',', preprocess_function=None):
    assert (sampling_freq is not None) or (time_col is not None), "either sampling frequency or time column shoul be provided"
    data = _np.loadtxt(datafile, delimiter=sep)
    
    values = data[:, data_col] 
    if preprocess_function is not None:
        values = preprocess_function(values)
    if (time_col is not None):
        times = data[:, time_col]
        signal = _create_signal(values, times = times)
    else:
        signal = _create_signal(values, sampling_freq=sampling_freq)
    
    return(signal)
    

def info_lsl(datafile):
    import pyxdf
    data, header = pyxdf.load_xdf(datafile, verbose=True)
    names = _np.array([d['info']['name'][0] for d in data])
    print(names)
    
def load_lsl(datafile, idx_stream, fresamp=None):
    import pyxdf
    lsl_data, _ = pyxdf.load_xdf(datafile, verbose=True)
    stream_data = lsl_data[idx_stream]
    t = stream_data['time_stamps']
    signal_values = stream_data['time_series']
    try:
        signal = _create_signal(signal_values, times=t)
        
        if fresamp is not None:
            signal = signal.p.resample(fresamp)
        return(signal)
    except Exception as e:
        print(e)
        return(t, signal_values)
    