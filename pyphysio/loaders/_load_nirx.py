import os
import numpy as _np
import xarray as _xr
from ..signal import create_signal
import scipy.optimize as _opt
import scipy.io as _spio
import math

_xr.set_options(keep_attrs=True)

#TODO: find source and cite!
#%%
#=====================
# supporting functions
#=====================
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

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = _spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def compute_channelsPos(SD):
    measList = SD['MeasList']
    srcPos = SD['SrcPos']
    detPos = SD['DetPos']
    chPos = []
    for ch in measList:
        source_xyz = srcPos[ch[0]-1]
        detector_xyz = detPos[ch[1]-1]
        chPos.append((source_xyz + detector_xyz)/2)
    SD['ChPos'] = _np.array(chPos)

def _remove_regexp(string):
    string = string.replace('\n', '')
    string = string.replace('"', '')
    string = string.replace("'", '')
    return(string)

def _parse_line(content, key, equal='='):
    idx_row = _np.where([x.startswith(key) for x in content])[0][0]
    row = content[idx_row]
    row = _remove_regexp(row)
    split = row.split(equal)
    return(split)

from itertools import product

def _parse_multiline(content, key):
    idx_row_st = _np.where([x.startswith(key) for x in content])[0][0]
    idx_row_sp = _np.where([x.startswith('#') for x in content])[0]

    idx_row_sp_true = idx_row_sp[_np.where(idx_row_sp>idx_row_st)[0][0]]                       
    
    rows = content[idx_row_st+1:idx_row_sp_true]
    rows_ = []
    for R in rows:
        rows_.append(_np.array(R.split('\t')).astype(float))
    
    return(_np.array(rows_))
    
def _parseSD(SDMask, SDKey, n_wl):
    M, N = SDMask.shape
    # third column is all ones by default... I have no idea what it is, just copying behavior from original script
    idexes = _np.vstack(_np.array((int(i+1), int(j+1), 1)) for i,j in product(range(M), range(N)) if SDMask[i,j] == 1 )
    nChannels = idexes.shape[0]
    output = _np.vstack(_np.hstack( (idexes, (i+1)*_np.ones(nChannels).reshape(nChannels,1) ) ) for i in range(n_wl) )  
    return(output.astype(int))

def _find_goodIDK(SDMask, SDKey):
    M, N = SDMask.shape
    goodIDX = [ SDKey["{0}-{1}".format(i+1, j+1)] - 1 for i,j in product(range(M), range(N)) if SDMask[i,j] == 1 ]
    return(goodIDX)

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
        src_i = channels[i,0]
        det_i = channels[i,1]
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

#%%
def load_hdr(HDR_FILE):
    with open(HDR_FILE, 'r') as f:
        content = f.readlines()
        
    Lambda = _np.array(_parse_line(content, 'Wavelengths')[1].split('\t')).astype(float)
    SDMask = _parse_multiline(content, 'S-D-Mask')
    SDKey = dict([(x.split(':')[0], int(x.split(':')[1])) for x in _parse_line(content, 'S-D-Key')[1].split(',')[:-1]])
    # SDKey = [x.split(':')[0] for x in _parse_line(content, 'S-D-Key')[1].split(',')[:-1]]
    ml = _parseSD(SDMask, SDKey, len(Lambda))
    
    fsamp = float(_parse_line(content, 'Sampling')[1])
    
    nSrcs = _np.max(ml[:,0])
    nDets = _np.max(ml[:,1])
    
    SD = {
        'Lambda': Lambda,
        'nSrcs': nSrcs,
        'nDets': nDets,
        'MeasList': ml,
        'SpatialUnit': 'cm', # probeInfo coordinates are in cm
        'SDmask' : SDMask,
        'SDkey': SDKey,
        'fsamp': fsamp
    }
    return(SD)

def load_probeInfo(FILE):
    probeInfo = loadmat(FILE)['probeInfo']
    newcoords = _rotate_clusters(probeInfo)
    return(newcoords)

def load_data(FILE, goodIDX):
    data = _np.loadtxt(FILE)[:,goodIDX]
    return(data)
    
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
def load_nirx(DATADIR, has_stim=True):
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
    
    SD = load_hdr(f'{DATADIR}/{FILE_HDR}')

    idx_pi = _np.where([x.endswith('probeInfo.mat') for x in filelist])[0][0]
    FILE_PI = filelist[idx_pi]
    
    newcoords = load_probeInfo(f'{DATADIR}/{FILE_PI}')
    nsrc = SD['nSrcs']
    ndet = SD['nDets']
    
    srcPos = _np.zeros((nsrc, 3))
    srcPos[:,2] = newcoords[0:nsrc,2] #src coords
    srcPos[:,0:2] = -newcoords[0:nsrc,0:2]# %additional 180º rotation
    SD['SrcPos'] = srcPos
    
    detPos = _np.zeros((ndet, 3))
    detPos[:,2] = newcoords[nsrc:,2] #det coords
    detPos[:,0:2] = -newcoords[nsrc:,0:2] #additional 180º rotation
    SD['DetPos'] = detPos
    compute_channelsPos(SD)    

    fsamp = SD['fsamp']
    wavelengths = SD['Lambda']
    goodIDX = _find_goodIDK(SD['SDmask'], SD['SDkey'])
    
    data = []
    for i_wl in range(len(wavelengths)):
        idx_data = _np.where([x.endswith(f'wl{i_wl+1}') for x in filelist])[0][0]
        FILE_DATA  = filelist[idx_data]
        data.append(load_data(f'{DATADIR}/{FILE_DATA}', goodIDX))
    
    data = _np.stack(data, axis=2)
    
    #detector dir:
    if 'Conditions' in filelist:
        filelist_cond = os.listdir(f'{DATADIR}/Conditions')
        idx_evt = _np.where([x.endswith('.evt') for x in filelist_cond])[0][0]
        FILE_EVT = f'Conditions/{filelist_cond[idx_evt]}'
    else:
        idx_evt = _np.where([x.endswith('.evt') for x in filelist])[0][0]
        FILE_EVT = filelist[idx_evt]
    
    N = data.shape[0]
    stim = _np.zeros(N)
    
    if has_stim:
        idx, codes = load_events(f'{DATADIR}/{FILE_EVT}', has_stim)
        if len(idx)>0:
            stim[idx] = codes
    
    info = {}
    for k,v in SD.items():
        info[k] = v
    
    #remove undesired attributes (that prevent saving to netcdf)
    del info['fsamp']
    del info['SDkey']
    
    for key in ['MeasList', 'SDmask', 'SrcPos', 'DetPos', 'ChPos']:
        info[f'{key}_DIM'] = info[key].shape[0]
        info[key] = info[key].ravel()
        
    nirs = create_signal(data, sampling_freq=fsamp, start_time=0, name = 'nirs', info=info)
    
    nirs = nirs.assign_coords(stim=('time', stim))
    return(nirs)