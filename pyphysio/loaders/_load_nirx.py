import os
import numpy as _np
import xarray as _xr
import h5py as _h5py
from ..signal import create_signal
import scipy.optimize as _opt
import scipy.io as _spio
from itertools import product
import math

_xr.set_options(keep_attrs=True)

#TODO get_stim(folder) to retrieve stimulus
'''
#detector dir:
if 'Conditions' in filelist:
    filelist_cond = os.listdir(f'{DATADIR}/Conditions')
    idx_evt = np.where([x.endswith('.evt') for x in filelist_cond])[0][0]
    FILE_EVT = f'Conditions/{filelist_cond[idx_evt]}'
else:
    idx_evt = np.where([x.endswith('.evt') for x in filelist])[0][0]
    FILE_EVT = filelist[idx_evt]

idx, codes = load_events(f'{DATADIR}/{FILE_EVT}', has_stim)

N = data.shape[0]
stim = np.zeros(N)
if len(idx)>0:
    stim[idx] = codes
    
stim = ph.EvenlySignal(stim, sampling_freq=fsamp, start_time = 0)
nirs = nirs.assign_coords(stim=('time', stim))
'''    

def load_nirx2(DATADIR):
    filelist = os.listdir(DATADIR)
    
    # LOAD HDR
    idx_hdr = _np.where([x.endswith('hdr') for x in filelist])[0][0]
    FILE_HDR = filelist[idx_hdr]
    with open(f'{DATADIR}/{FILE_HDR}', 'r') as f:
        content = f.readlines()
    
    content_dict = {}
    for i in content:
        if '=' in i:
            k = i.split('=')[0]
            v = _remove_regexp(i.split('=')[1])
            if v !='#':
                content_dict[k] = [v]
        elif '[' not in i:
            if k == 'Channel Mask':
                i = _remove_regexp(i)
                if i != '#':
                    row = [int(x) for x in i.split('     ')]
                    content_dict[k].append(row)
            if k == 'Channel indices':
                row = _remove_regexp(i).split(', ')
                row = [ [int(x.split('-')[0]), int(x.split('-')[1])] for x in row]
                    
                content_dict[k] = row
    
    content_dict['Channel Mask'] = _np.array(content_dict['Channel Mask'][1:])
    # content_dict['Channel indices'] = dict([(i, k) for i,k in enumerate(content_dict['Channel indices'])])

    SD = {}
    
    
    SD['SDmask'] = content_dict['Channel Mask']
    SD['SDkey'] = _np.array(content_dict['Channel indices'])
    
    idx_snirf = _np.where([x.endswith('snirf') for x in filelist])[0]
    
    if len(idx_snirf) == 1:
        SD['SpatialUnit'] = 'cm' #TODO; check
        FILE_SNIRF = filelist[idx_snirf[0]]
    
        # LOAD snirf
        f = _h5py.File(f'{DATADIR}/{FILE_SNIRF}', 'r')
    
        # formatVersion = f['formatVersion'][()][0]
        nirs_data = f['nirs']['data1']['dataTimeSeries'][()]
        
        n_channels = int(nirs_data.shape[1] / 2)
        nirs_data_out = []
        for i in range(n_channels):
            ch_data = nirs_data[:, [i, i+n_channels]]
            # ch_data = ch_data[:, _np.newaxis, :]
            nirs_data_out.append(ch_data)
        
        nirs_data = _np.stack(nirs_data_out, 1)
        time = f['nirs']['data1']['time'][()]
        fsamp = 1/(time[1] - time[0])
        
        nirs_probe_metadata = {}
        for k in f['nirs']['probe'].keys():
            nirs_probe_metadata[k] = f['nirs']['probe'][k][()]
            
        SD['Lambda'] = nirs_probe_metadata['wavelengths']
        SD['SrcPos'] = nirs_probe_metadata['sourcePos3D']/10
        SD['SrcPos2D'] = nirs_probe_metadata['sourcePos2D']/10
        
        SD['DetPos'] = nirs_probe_metadata['detectorPos3D']/10
        SD['DetPos2D'] = nirs_probe_metadata['detectorPos2D']/10
        
        SD['ChnPos'] = _np.array(compute_channelsPos(SD['SDkey'],
                                                     SD['SrcPos'], 
                                                     SD['DetPos']))
        
        SD['ChnPos2D'] = _np.array(compute_channelsPos(SD['SDkey'],
                                                       SD['SrcPos2D'], 
                                                       SD['DetPos2D']))
    else:
        SD['SpatialUnit'] = 'cm' #TODO; check
        n_channels = 2
        Lambda = [760., 850.]
        SD['Lambda'] = Lambda #
        SD['SrcPos'] = None
        SD['SrcPos2D'] = None
        
        SD['DetPos'] = None
        SD['DetPos2D'] = None
        
        SD['ChnPos'] = None
        
        SD['ChnPos2D'] = None

        fsamp = float(content_dict['Sampling rate'][0])

        nirs_data = []
        for i_wl in range(len(Lambda)):
            idx_data = _np.where([x.endswith(f'wl{i_wl+1}') for x in filelist])[0][0]
            FILE_DATA  = filelist[idx_data]
            data_ = _np.loadtxt(f'{DATADIR}/{FILE_DATA}')
            nirs_data.append(data_)
        
        nirs_data = _np.stack(nirs_data, axis=2)
        
        idx_pi = _np.where([x.endswith('probeInfo.mat') for x in filelist])[0][0]
        FILE_PI = filelist[idx_pi]
        
        srcPos, detPos = load_probeInfo(f'{DATADIR}/{FILE_PI}')
        
        SD['SrcPos'] = srcPos
        SD['DetPos'] = detPos
        
        ChnPos = _np.array(compute_channelsPos(SD['SDkey'], SD['SrcPos'], SD['DetPos']))
        SD['ChnPos'] = ChnPos
    
    # if full: #no urgent
    #     ids =[]
    #     for k in f['nirs']['data1'].keys():
    #         if k.startswith('measurementList'):
    #             ids.append(k.split('measurementList')[1])
    
    #     nirs_signal_metadata = {}
    #     for m in ids:
    #         d = {}
    #         for k in f['nirs']['data1'][f'measurementList{m}'].keys():
    #             d[k] = f['nirs']['data1'][f'measurementList{m}'][k][()]
    #         nirs_signal_metadata[m] = d
    
    #     nirs_acquisition_metadata = {}
    
    #     for k in f['nirs']['metaDataTags'].keys():
    #         nirs_acquisition_metadata[k] = f['nirs']['metaDataTags'][k][()][0]
        
    nirs = create_signal(nirs_data, sampling_freq=fsamp, start_time=0, name = 'nirs', info=SD)

    return(nirs)

#TODO: find source and cite!
#%%
#=====================
# supporting functions
#=====================

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], _spio.matlab.mio5_params.mat_struct):
                dict[key] = _todict(dict[key])
        return dict   

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, _spio.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict
    
    data = _spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def compute_channelsPos(SDkey, srcPos, detPos):
    chPos = []
    for i_ch, sd in enumerate(SDkey):
        source_xyz = srcPos[sd[0]]
        detector_xyz = detPos[sd[1]]
        chPos.append((source_xyz + detector_xyz)/2)
    # SD['ChPos'] = _np.array(chPos)
    return(chPos)

def _remove_regexp(string):
    string = string.replace('\n', '')
    string = string.replace('"', '')
    string = string.replace("'", '')
    return(string)
   
def _parseSD(SDMask, n_wl):
    M, N = SDMask.shape
    
    # third column is all ones by default... I have no idea what it is, just copying behavior from original script
    idexes_to_stack = [_np.array((int(i+1), int(j+1), 1)) for i,j in product(range(M), range(N)) if SDMask[i,j] == 1 ]
    idexes = _np.vstack(idexes_to_stack)
    nChannels = idexes.shape[0]
    output_to_stack = [_np.hstack( (idexes, (i+1)*_np.ones(nChannels).reshape(nChannels,1) ) ) for i in range(n_wl)]
    output = _np.vstack(output_to_stack)  
    return(output.astype(int))

def _find_origin(pi):
    def fun(x):
        k = _np.ones(len(pi['optode_coords']))
        for i in range(len(k)):
            k[i] = _np.linalg.norm(pi['optode_coords'][i,:] - x)
        return(_np.std(k))
        
    origin = _opt.fmin(func = fun, x0=[0, 0, 0], disp=False)
    return(origin)

def _cluster_search_mat(pi):

    index = pi['channel_indices']
    k = index[0,0] #k is id of the src/det
    source = 1
    cluster = 1
    
    found = dict()
    found[cluster] = [k]
    found_src = []
    found_src.append(k)
    found_det = []
    
    while (index.shape[0] > 0):
        
        if source: # work on source
            chn = _np.where(index[:,0] == k)[0]# %channels with source = k
            if len(chn)>0: #there are channels with source = k
                for i in range(len(chn)):
                    if not (found[cluster] == index[chn[i],1]).any():
                        found[cluster].append(index[chn[i],1])
                        found_det.append(index[chn[i],1])

                index = _np.delete(index, chn, axis=0) #remove channels from index list

            if index.shape[0] == 0:
                break

            found_src.remove(found_src[0]) #remove current source index
            if len(found_src) == 0: #change to detector indexes
                source = 0
                if len(found_det) == 0:
                    k = index[0,1] # if both are empty, re-initialize
                    found_det.append(k)
                    cluster = cluster + 1 #go to next cluster
                    found[cluster] = [k]
                else:
                    k = found_det[0]
            else:
                k = found_src[0]
        else:
            chn = _np.where(index[:,1] == k)[0]# %channels with detector = k
            if len(chn)>0:
                for i in range(len(chn)):
                    if not (found[cluster] == index[chn[i],0]).any():
                        found[cluster].append(index[chn[i],0])
                        found_src.append(index[chn[i],0])

                index = _np.delete(index, chn, axis=0) #remove channels from index list

            if index.shape[0] == 0:
                break

            found_det.remove(found_det[0]) #remove current source index
            if len(found_det) == 0: #change to detector indexes
                source = 1
                if len(found_src) == 0:
                    k = index[0,0] #if both are empty, re-initialize
                    found_src.append(k)
                    cluster = cluster + 1 #go to next cluster
                    found[cluster] = [k]
                else:
                    k = found_src[0]

            else:
                k = found_det[0]
    return(found)

def _rotmat(point, direction, theta):
    a = point[0]
    b = point[1]
    c = point[2]
    
    t = direction/_np.linalg.norm(direction)
    u = t[0]
    v = t[1]
    w = t[2]

    si = _np.sin(theta)
    co = _np.cos(theta)

    mat = _np.zeros((4,4))

    # rotational part    
    mat[0:3, 0:3] = [[(u*u + (v*v + w*w) * co), (u*v*(1-co) - w*si),     (u*w*(1-co) + v*si)],
                      [(u*v*(1-co) + w*si),      (v*v + (u*u + w*w)*co),  (v*w*(1-co) - u*si)],
                      [(u*w*(1-co) - v*si),      (v*w*(1-co) + u*si),     (w*w + (u*u + v*v)*co)]]

    # translational part
    mat[0,3] = (a*(v*v+w*w)-u*(b*v+c*w)) * (1-co) + (b*w-c*v)*si
    mat[1,3] = (b*(u*u+w*w)-v*(a*u+c*w)) * (1-co) + (c*u-a*w)*si
    mat[2,3] = (c*(u*u+v*v)-w*(a*u+b*v)) * (1-co) + (a*v-b*u)*si
    mat[3,3] = 1
    
    return(mat)

def _rotate_clusters(probeInfo):

    src = probeInfo['probes']['coords_s3']
    det = probeInfo['probes']['coords_d3']
    channels = probeInfo['probes']['index_c']
    
    pi = dict()
    pi['nsources'] = len(src)
    pi['ndetectors'] = len(det)
    
    pi['optode_coords'] = _np.concatenate([src, det], axis=0)
    
    pi['channel_indices'] = _np.zeros((len(channels),2)).astype(int)
    pi['channel_distances'] = _np.zeros(len(channels))
    
    for i in range(len(channels)):
        src_i = int(channels[i,0])
        det_i = int(channels[i,1])
        pi['channel_indices'][i,:] = [src_i, len(src)+det_i]
        pi['channel_distances'][i] = _np.linalg.norm(src[src_i-1,:] - det[det_i-1,:])
    
    origin = _find_origin(pi)
    for i in range(len(pi['optode_coords'])):
        pi['optode_coords'][i,:] = pi['optode_coords'][i,:] - origin
    
    clusters = _cluster_search_mat(pi)
    
    newcoords = _np.zeros_like(pi['optode_coords'])
    
    for i in range(len(clusters)):
        
        idx = _np.array(clusters[i+1]) -1 # because srcid = 1 is idx=0
        center = _np.mean(pi['optode_coords'][idx,:], axis=0)
        
        #center in spherical coordinates
        center_phi = math.atan2(center[1], center[0])
        
        #phi tangent vector
        tangent = [-_np.sin(center_phi), _np.cos(center_phi), 0]
        
        #angle between r and z unit vectors
        angle = math.acos( center[2]/_np.linalg.norm(center) )
        
        mat = _rotmat(center, tangent, -angle)
        
        coords =  _np.dot(mat, _np.hstack([pi['optode_coords'][idx,:], _np.ones((len(idx),1))]).T).T
        
        
        newcoords[idx,:] = coords[:,0:3]
    return(newcoords)

def load_probeInfo(FILE):
    probeInfo = loadmat(FILE)['probeInfo']
    
    src = probeInfo['probes']['coords_s3']
    det = probeInfo['probes']['coords_d3']
    newcoords = _np.concatenate([src, det], axis=0)
    
    nsrc = src.shape[0]
    ndet = det.shape[0]
    
    srcPos = _np.zeros((nsrc, 3))
    srcPos[:,2] = newcoords[0:nsrc,2] #src coords
    srcPos[:,0:2] = -newcoords[0:nsrc,0:2]# %additional 180º rotation
    
    detPos = _np.zeros((ndet, 3)) #!!! WTF
    detPos[:,2] = newcoords[nsrc:,2] #det coords
    detPos[:,0:2] = -newcoords[nsrc:,0:2] #additional 180º rotation
    
    return(srcPos, detPos)
    
def load_events(FILE, has_stim=True):
    if not has_stim:
        return(_np.array([0]), _np.array([1]))
    
    events = _np.loadtxt(FILE)
    if events.shape[0] == 0:
        return(_np.array([]), _np.array([]))
    
    # added: work with only one event:
    if events.ndim == 2:
        idx = events[:,0].astype(int)
        codes = _np.sum(events[:,1:].astype(int) * 2**_np.arange(8), axis = 1)
        return(idx, codes)
    elif events.ndim == 1:
        idx = events[0].astype(int)
        codes = _np.sum(events[1:].astype(int) * 2**_np.arange(8))
        return(_np.array([idx]), _np.array([codes]))
    else:
        print('Error processing event file')
        
#%%
def load_nirx(DATADIR):
    """Import NIRS data generated with NIRx devices.
    
    Parameters
    ----------
    DATADIR : str
        Path to the directory containing the files
    
    has_stim : boolean, optional
        Whether the try to load the information about the stimuli
              

    Returns
    -------
    nirs : pynirs.NIRS
            Object cointaining the nirs data and metadata
    """
    
    filelist = os.listdir(DATADIR)
    idx_hdr = _np.where([x.endswith('hdr') for x in filelist])[0][0]
    FILE_HDR = filelist[idx_hdr]
    
    HDR_FILE = f'{DATADIR}/{FILE_HDR}'
    
    with open(HDR_FILE, 'r') as f:
        content = f.readlines()
    
    content_dict = {}
    for i in content:
        if '=' in i:
            k = i.split('=')[0]
            v = _remove_regexp(i.split('=')[1])
            if v !='#':
                content_dict[k] = [v]
            else:
                content_dict[k] = []
        elif '[' not in i:
            if k == 'S-D-Mask':
                i = _remove_regexp(i)
                if i != '#' and i != '':
                    row = [int(x) for x in i.split('\t')]
                    content_dict[k].append(row)
    
    Lambda = [float(content_dict['Wavelengths'][0].split('\t')[0]),
              float(content_dict['Wavelengths'][0].split('\t')[1])]
    SDMask = _np.array(content_dict['S-D-Mask'])

    ml = _parseSD(SDMask, len(Lambda))
    n_channels = int(ml.shape[0] / len(Lambda))
    
    SDKey_support = {}
    for k in content_dict['S-D-Key'][0].split(','):
        if ':' in k:
            sd = k.split(':')[0]
            idx = k.split(':')[1]
            SDKey_support[sd] = int(idx)
            
    SDKey = []
    goodIDX = []
    for i in range(n_channels):
        s = ml[i,0]-1
        d = ml[i,1]-1
        SDKey.append([s,d])
        goodIDX.append(SDKey_support[f'{s+1}-{d+1}']-1)

    fsamp = float(content_dict['SamplingRate'][0])
    
    nsrc = int(content_dict['Sources'][0])
    ndet = int(content_dict['Detectors'][0])
    
    SD = {
        'SpatialUnit': 'cm', #TODO: check
        'Lambda': Lambda,
        'SDmask' : SDMask,
        'SDkey': _np.array(SDKey)
    }
    

    idx_pi = _np.where([x.endswith('probeInfo.mat') for x in filelist])[0][0]
    FILE_PI = filelist[idx_pi]
    
    srcPos, detPos = load_probeInfo(f'{DATADIR}/{FILE_PI}')
    SD['SrcPos'] = srcPos
    SD['DetPos'] = detPos
    
    ChnPos = _np.array(compute_channelsPos(SD['SDkey'], SD['SrcPos'], SD['DetPos']))
    SD['ChnPos'] = ChnPos
    
    data = []
    for i_wl in range(len(Lambda)):
        idx_data = _np.where([x.endswith(f'wl{i_wl+1}') for x in filelist])[0][0]
        FILE_DATA  = filelist[idx_data]
        data_ = _np.loadtxt(f'{DATADIR}/{FILE_DATA}')
        data.append(data_[:, goodIDX])
    
    data = _np.stack(data, axis=2)
    
    # #detector dir:
    # if 'Conditions' in filelist:
    #     filelist_cond = os.listdir(f'{DATADIR}/Conditions')
    #     idx_evt = _np.where([x.endswith('.evt') for x in filelist_cond])[0][0]
    #     FILE_EVT = f'Conditions/{filelist_cond[idx_evt]}'
    # else:
    #     idx_evt = _np.where([x.endswith('.evt') for x in filelist])[0][0]
    #     FILE_EVT = filelist[idx_evt]
    
    # N = data.shape[0]
    # stim = _np.zeros(N)
    
    # if has_stim:
    #     idx, codes = load_events(f'{DATADIR}/{FILE_EVT}', has_stim)
    #     if len(idx)>0:
    #         stim[idx] = codes
    nirs = create_signal(data, sampling_freq=fsamp, start_time=0, name = 'nirs', info=SD)
    
    # nirs = nirs.assign_coords(stim=('time', stim))
    return(nirs)