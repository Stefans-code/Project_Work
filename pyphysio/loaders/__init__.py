from ._load_nirs import load_nirx2, load_nirx, load_snirf, load_xrnirs, SDto1darray
from ..signal import create_signal as _create_signal
#TODO: add modules for loading edf, physionet?
import numpy as _np
import pyxdf

def get_xdf_stream_names(datafile):
    # Load the XDF file
    data, header = pyxdf.load_xdf(datafile, verbose=True)
    
    # Extract stream names
    names = [stream["info"]["name"][0] for stream in data]
    return names

def info_xdf(datafile):
    print(get_xdf_stream_names(datafile))


def load_xdf(datafile, stream_name, stream_type='signal', select_multiple=None,
             start_time = None, times_from_lsl=True):
    
    data, header = pyxdf.load_xdf(datafile, verbose=True)

    recoding_dict = None #will be not None if used for recoding of events; in that case it should be added to the attrs
    signals_info = None #will be not None if multiple signals are present
    
    names = _np.array(get_xdf_stream_names(datafile))


    idx_stream = _np.where(names == stream_name)[0]
    assert len(idx_stream) > 0, f"No streams found with name: {stream_name}"
    
    if select_multiple is None:
        assert len(idx_stream) == 1, f"More than one streams found with name: {stream_name}. Please set the parameter 'select_multiple' to load the appropriate stream"
        idx_stream = idx_stream[0]
    else:
        idx_stream = idx_stream[select_multiple]
        
    data_ = data[idx_stream]
    t = data_['time_stamps']
    
    if start_time is not None:
        t = t - t[0] + start_time
    fsamp = data_['info']['effective_srate']
    
    
    if stream_type == 'signal':
        signal_values = _np.array(data_['time_series'])
        if signal_values.shape[1] > 1:
            channel = data_['info']['desc'][0]['channels'][0]['channel']
            signals_info = [dict(c) for c in channel]
        
    elif stream_type == 'events':
        signal_values =  _np.array(data_['time_series']).ravel()
        if isinstance(signal_values[0], str):
            unique_values = _np.unique(signal_values)
            recoding_dict = dict([(v, i) for i,v in enumerate(unique_values)])
            signal_values = _np.array([recoding_dict[v] for v in signal_values])
        
    elif stream_type == 'nirs':
        channels_info = data_['info']['desc'][0]['channels'][0]['channel']
        
        idx_raw = _np.where([ch['type'][0] == 'nirs_raw' for ch in channels_info])[0]
            
        n_channels = len(idx_raw)//2
    
        signal_values_w1 = _np.array(data_['time_series'][:, idx_raw[:n_channels]])
        signal_values_w2 = _np.array(data_['time_series'][:, idx_raw[n_channels:]])
            
        signal_values = _np.stack([signal_values_w1, signal_values_w2], axis=2)
        
        montage_info = data_['info']['desc'][0]['montage'][0] # pprint(montage_info)
        sources_info = montage_info['optodes'][0]['sources'][0]['source'] 
        detectors_info = montage_info['optodes'][0]['detectors'][0]['detector'] # 
        
        SD = {}

        # generating SD['Lambda']
        wavelengths = []
        nirs_info = data_['info']['desc'][0]['channels'][0]['channel']

        for entry in nirs_info:
            if 'wavelength' in entry and entry['wavelength']: # Check if 'wavelength' exists and is non-empty
                wavelengths.append(float(entry['wavelength'][0]))  # Add the first element of 'wavelength' to the list and convert to float
                
        unique_wavelengths = list(set(wavelengths)) # get unique elements, remove duplicates       

        assert len(unique_wavelengths) == 2, f"Unexpected number of wavelengths: {len(unique_wavelengths)}"
        SD['Lambda'] = unique_wavelengths

        distance_conv = 1
        
        info_channels = []
        for i_ch in range(0, n_channels):
            channel = data_['info']['desc'][0]['channels'][0]['channel'][i_ch+1] #getting the i_ch+1 channel (i_ch=0 is 'frame')
            
            # creating i_ch
            label = channel['label'][0].split(':')[0]
            source_id = int(label.split('-')[0])  # Extract source id
            detector_id = int(label.split('-')[1])  # Extract detector id
            idx_src = source_id - 1  # Convert to 0-based index, qui per poi matchare indici per canali
            idx_det = detector_id - 1  # Convert to 0-based index

            distance = float(channels_info[i_ch+1]['distance'][0])#_np.linalg.norm(srcPos - detPos)
            if distance > 10:
                distance_conv = 10
            
            distance = distance/distance_conv
              
            ch_dict = [i_ch, idx_src, idx_det, distance]#, distance2D]

            info_channels.append(ch_dict)
        
        info_channels = _np.array(info_channels)

        srcPos = []
        sources_info = montage_info['optodes'][0]['sources'][0]['source']
        for source in sources_info:
            location = source['location'][0]
            x = float(location['x'][0])/distance_conv
            y = float(location['y'][0])/distance_conv
            z = float(location['z'][0])/distance_conv
            srcPos.append([x, y, z])
        
        SD['SrcPos'] = _np.array(srcPos)
        
        detPos = []
        detectors_info = montage_info['optodes'][0]['detectors'][0]['detector']
        for detector in detectors_info:
            location = detector['location'][0]
            x = float(location['x'][0])/distance_conv
            y = float(location['y'][0])/distance_conv
            z = float(location['z'][0])/distance_conv
            detPos.append([x, y, z])

        SD['DetPos'] = _np.array(detPos)
        # Format the data: cast first 3 columns to integers and round last 2 columns to 8 decimal places
        info_channels[:, :3] = info_channels[:, :3].astype(int)  # First 3 columns to integers
        info_channels[:, 3:] = _np.round(info_channels[:, 3:], 8)  # Last 2 columns to 8 decimals

        SD['Channels'] = info_channels
        SD['SpatialUnit'] = 'cm'
        
        SD['sampling_freq'] = fsamp
        SD['start_time'] = t[0]
        
        #% try to extact subj info
        try:
            subject = data_['info']['desc'][0]['demographics'][0]['subject'][0]
        except:
            subject = None
        
        #% try to extact age info
        try:
            age = data_['info']['desc'][0]['demographics'][0]['age'][0]
        except:
            age = None
                
    if times_from_lsl: #rely on timestamps from LSL
        signal = _create_signal(signal_values, times=t)    
    else:
        #if we dont trust the LSL times, we probably wont trust the effective_srate
        fsamp = float(data_['info']['nominal_srate'][0])
        assert fsamp > 0
        signal = _create_signal(signal_values, start_time=t[0], sampling_freq=fsamp)
        
    if stream_type == 'nirs':
        signal.p.main_signal.attrs = SD
        if subject is not None:
            signal.p.main_signal.attrs.update({'subject': subject})
        if age is not None:
            signal.p.main_signal.attrs.update({'age': age})
    
    elif signals_info is not None:
        signal.p.main_signal.attrs.update({'signals_info': signals_info})
    elif recoding_dict is not None:
        signal.p.main_signal.attrs.update({'event_codes': recoding_dict})
    
    if stream_type != 'events':
        signal = signal.p.resample(fsamp)
    return(signal)

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
    
