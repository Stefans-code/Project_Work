#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:50:15 2022

@author: bizzego
"""

import numpy as _np
from scipy.stats import median_abs_deviation as _mad
import scipy.linalg as _sal
from ...processing import Algorithm as _Algorithm
from csaps import csaps as _csaps
from ...processing.filters import IIRFilter as _IIRFilter
import matplotlib.pyplot as plt
import pywt

class MARA(_Algorithm):
    #TODO: split into a detect MA and correct artifacts
    '''
    F Scholkmann et al 2010 Physiol. Meas. 31 649
    '''
    def __init__(self, win_len = 5, threshold = None, fuse=False, **kwargs):
        _Algorithm.__init__(self, win_len=win_len, threshold=threshold, **kwargs)
        
        #IDEA for the MA detection, we can do that by channel or globally
        #and adapt the behaviour of the algorithm on the different dimensions:
        if fuse:
            self.dimensions = {'time' : 0, 'channels':1, 'components':1}
        else:
            self.dimensions = {'time' : 0}
        #TODO: compute threshold automatically
        #TODO: MA estimated from fused channels?
    
    
    def algorithm(self, signal):
        params = self._params
        win_len = params['win_len']
        threshold = params['threshold']
        
        fsamp = signal.p.get_sampling_freq()
        data_ch = signal.values.ravel()

        #compute threshold
        if threshold is None:
            signal_f = _IIRFilter(fp=0.05, fs=0.01)(signal)
            data_ch_filt = signal_f.values.ravel()
            threshold = _np.std(data_ch_filt)
            
        # 1 moving standard deviation MSD (win size/step?) 
        idx_len = int(win_len*fsamp)
        half = idx_len //2
        MSD = []
        for i in range(len(data_ch - idx_len)):
            MSD.append(_np.std(data_ch[i: i+idx_len]))

        # 2 detection moving artifacts (MA) start and end
        #IDEA: use peakdetection to identify the MA
        MSD = _np.array(MSD)
        MSD = MSD - _np.median(MSD)
        MSD = (MSD >= threshold).astype(int)
        
        idxlen_smooth = int(2*fsamp) #see tMask
        MSD = _np.convolve(MSD, _np.ones(idxlen_smooth)/idxlen_smooth, 'same') #TODO: needed?
        MSD = (MSD > 0).astype(int)

        MSD_ = _np.diff(MSD)
        idx_st = _np.where(MSD_ > 0)[0]
        idx_sp = _np.where(MSD_ < 0)[0]

        # 3 create list of segments w/ MA x_bad and w/o MA x_good
        #TODO manage special cases with MA at beginning or end
        x_good = []
        x_bad = []
        idx_start = 0
        for id_MA, (idx_st_MA, idx_sp_MA) in enumerate(zip(idx_st,idx_sp)):
            # if (idx_sp_MA - idx_st_MA) < 2:
            #     plt.plot(signal.values.ravel())
            x_good.append(data_ch[idx_start: idx_st_MA + half])
            x_bad.append(data_ch[idx_st_MA + half : idx_sp_MA + half])
            idx_start = idx_sp_MA + half
        x_good.append(data_ch[idx_start:])
        
        # 4 spline interpolation (X_MA_s) of each segment in X_MA
        #+5 subtraction of X_MA_s from each X_MA
        x_corr = []
        for x in x_bad:    
            if len(x) <=2:
                x_corr.append(x)
            else:
                idxs = _np.arange(len(x))
                x_ = _csaps(idxs, x, idxs, smooth=0.01)
                x_corr.append(x - x_)

        # 6 reconstruction
        n_samples = int(fsamp / 3)
        #TODO: see paper for the number of samples to consider
        x_correct_segments = []
        for i in range(len(x_corr)):
            x_correct_segments.append(x_good[i])
            x_correct_segments.append(x_corr[i])
        x_correct_segments.append(x_good[-1])

        x_reconstructed = []
        x_prev_mean = _np.mean(x_correct_segments[0][:n_samples])
        for x_segment in x_correct_segments:
            if len(x_segment)>0:
                x_segment_demean = x_segment  - _np.mean(x_segment) + x_prev_mean
                x_reconstructed.append(x_segment_demean)
                x_prev_mean = _np.mean(x_segment_demean[-n_samples:])
                
        x = _np.concatenate(x_reconstructed, axis=0)
       
        return(x)
    
class WaveletFilter(_Algorithm):
    """
    See Molavi 2012

    """    
    def __init__(self, iqr=1.5, **kwargs):
        _Algorithm.__init__(self, iqr=iqr, **kwargs)
        self.dimensions = {'time' : 0}
        
    
    def _normalization_noise(self, y):
        qmf = _np.array([-0.0915, -0.1585, 0.5915, -0.3415])
        
        n = len(y)
        c = _np.convolve(_np.tile(y, 2), qmf, 'same')[:n]
        MAD = _mad(c, scale='normal')
        
        if MAD !=0:
            y_norm = (1/1.4826)*y/MAD
            coeff = 1/(1.4826*MAD)
        else:
            y_norm = y
            coeff = 1
        return(y_norm, coeff)
    
    def algorithm(self, signal): #TODO: correct sintax for **kwargs
        
        signal_values = signal.values.ravel()
        
        iqr = self._params['iqr']
        
        n = len(signal_values)
        N = int(_np.ceil(_np.log2(n)))
        
        L = 4
        D = N - L
        
        n_padded = 2**N
        signal_padded = _np.zeros(shape = n_padded)
        signal_padded[:n] = signal_values
        mean_padded = _np.mean(signal_padded)
        signal_padded = signal_padded-mean_padded
        
        signal_norm, norm_coeff = self._normalization_noise(signal_padded)
        
        #+++++++++++++++++++++++++
        #compute discrete wavelet transform on signal and shifted version
        #for all block lengths
        wp = _np.zeros(shape=(n_padded,D+1))
        wp[:,0] = signal_norm
        
        for d in range(0, D):
            n_blocks = int(2**d) # number of blocks in the level
            l_blocks = int(n/n_blocks) # length of the blocks in the level
            for b in range(0, n_blocks):
                # first time take signal, from the second the approximation
                s = wp[b*l_blocks:b*l_blocks+l_blocks,0] 
                #create a shift version of the block
                s_shift = _np.concatenate([[s[-1]], s[0:-1]])
                
                #discrete wavelet transform
                cA,cD = pywt.dwt(s,'db2', mode='periodization')
                cA_shift, cD_shift = pywt.dwt(s_shift,'db2', mode='periodization')
                
                #save coefficients
                wp[b*l_blocks : b*l_blocks+len(cA), 0] = cA
                wp[b*l_blocks+len(cA):b*l_blocks+len(cA)+len(cA_shift),0] = cA_shift
                
                wp[b*l_blocks:b*l_blocks+len(cD),d+1] = cD
                wp[b*l_blocks+len(cD):b*l_blocks+len(cD) + len(cD_shift),d+1] = cD_shift
        
        #+++++++++++++++++++++++++
        #filter oulier coefficients
        n_tmp = n
        
        for d in range(1, D):
            n_tmp = n_tmp//2
            n_blocks = int(2**d)
            l_blocks = int(n/n_blocks)
            for b in range(0, n_blocks):
                sr = wp[b*l_blocks:b*l_blocks+l_blocks,d]
                
                #obtain outliers based on IQR
                sr_ = sr[:n_tmp] # compute statistics only on original data
                quants = _np.quantile(sr_,[.25, .50, .75]) # compute quantiles
                IQR = quants[2]-quants[0] # compute interquartile range
                prob1 = quants[2]+IQR*iqr#
                prob2 = quants[0]-IQR*iqr#
                outliers_1 = _np.where(sr>prob1)[0]
                outliers_2 = _np.where(sr<prob2)[0]
                outliers = _np.concatenate([outliers_1, outliers_2])
                
                #set outliers to zero
                sr[outliers] = 0 
                
                #save results
                wp[b*l_blocks:b*l_blocks+l_blocks,d] = sr
        
        #++++++++++++++++++++++++
        #discrete inverse transform to obtain the reconstructed signal
        approx = wp[:,0] #approximation coefficients in the first column
        
        for d in range(D-2,-1,-1):
            n_blocks = int(2**d)
            l_blocks = int(n/n_blocks)
            l_blocks_2 = l_blocks//2
            
            for b in range(0,n_blocks):
                cD = wp[b*l_blocks : b*l_blocks+l_blocks_2, d+1]
                cD_shift = wp[b*l_blocks+l_blocks_2 : b*l_blocks+2*l_blocks_2, d+1]
                cA = approx[b*l_blocks : b*l_blocks+l_blocks_2]
                cA_shift = approx[b*l_blocks+l_blocks_2 : b*l_blocks+l_blocks_2*2]
                
                s1 = pywt.idwt(cA,cD,'db2', mode='periodization')
                s_shift = pywt.idwt(cA_shift,cD_shift,'db2', mode='periodization')
                s2 = _np.concatenate([ s_shift[1:], [s_shift[0]]])
                
                approx[b*l_blocks:b*l_blocks+len(s1)] = (s1+s2)/2
                
        reconstructed = approx/norm_coeff + mean_padded
        return(reconstructed[:n])
    
class PCAFilter(_Algorithm):
    """
    See Molavi 2012

    """    
    def __init__(self, nSV=0.8, **kwargs):
        _Algorithm.__init__(self, nSV=nSV, **kwargs)
        self.dimensions = 'none'
    
    # def __call__(self, signal, manage_original):
    #     return _Algorithm.__call__(self, signal,
    #                                by='none', 
    #                                manage_original=manage_original)
    
    def algorithm(self, signal): #TODO: correct syntax for **kwargs

        nSV = self._params['nSV']
        n_channels = signal.p.get_nchannels()
        y = signal.p.get_values()
        # idx_good_channels = signal.get_good_channels()
        # y = y_[:, idx_good_channels]
        
        
        y = _np.concatenate([y[:,:,0], y[:,:,1]], axis=1)
        c = _np.dot(y.T, y)
        V, St, _ = _sal.svd(c)
        svs = St / _np.sum(St)
        
        ev = _np.zeros(len(svs))
        if nSV>1:
            ev[:nSV] = 1
        else:
            svsc = svs
            for idx in _np.arange(1, len(svs)):
                svsc[idx] = svsc[idx-1] + svs[idx]
            ev[svsc<=nSV] = 1
        #%
        ev = _np.diag(ev)
        
        y = y - _np.linalg.multi_dot([y, V, ev, V.T])
        
        y = _np.stack([y[:, :n_channels], y[:, n_channels:]], axis=2)
        return(y)
    
class NegativeCorrelationFilter(_Algorithm):
    '''
    Functional near infrared spectroscopy (NIRS) signal improvement based on negative correlation between oxygenated and deoxygenated hemoglobin dynamics
    '''
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.dimensions = {'time':0, 'component':0}
        
    # def __call__(self, signal, manage_original):
    #     return _Algorithm.__call__(self, signal,
    #                                by='channel', 
    #                                manage_original=manage_original)
    
    def algorithm(self, signal):
        oxy = signal.values[:,0,0]
        oxy_true = _np.zeros_like(oxy)
        
        deoxy = signal.values[:,0,1]
        deoxy_true = _np.zeros_like(oxy)
        
        
        alpha = _np.std(oxy)/_np.std(deoxy)
    
        oxy_true = 0.5 * (oxy - alpha*deoxy)
        deoxy_true = -oxy_true/alpha
        
        signal_out = _np.zeros_like(signal.values)
        signal_out[:,0,0] = oxy_true
        signal_out[:,0,1] = deoxy_true
        
        return(signal_out)


"""

def __finalize_special__(res_sig):
    # print('----->', self.name, 'finalize')
    original_coords = list(res_sig.coords)
    res_sig = res_sig.reset_coords()
    
    dimensions = list(res_sig.dims)
    for c in original_coords:
        if c not in dimensions:
            res_sig = res_sig.drop(c)
    res_sig = res_sig.to_array()
    res_sig = res_sig.squeeze(dim='variable').drop('variable')
    # print('<-----', self.name, 'finalize')
    return res_sig

class FunctionalSeparationFilter(_Algorithm):
    '''
    Yamada, T., Umeyama, S., & Matsuda, K. (2012). 
    Separation of fNIRS signals into functional and systemic components 
    based on differences in hemodynamic modalities. 
    PloS one, 7(11), e50271.
    
    From:
        https://unit.aist.go.jp/hiiri/nrehrg/download/dl002_download.html
    '''
    
    def __init__(self, kf=-0.6, nbins=8, **kwargs):
        _Algorithm.__init__(self, kf=kf, nbins=nbins, **kwargs)
        self.dimensions = 'special'
    
    def __finalize__(self, res_sig, arr_window):
        return __finalize_special__(res_sig)
    
    def __get_template__(self, signal):
        out = _np.zeros(shape=(signal.sizes['time'],
                               signal.sizes['channel'],
                               4))
        
        out = _xr.DataArray(out, dims=('time', 'channel', 'component'),
                            coords = {'time': signal.coords['time'].values,
                                      'channel': signal.coords['channel'],
                                      'component': _np.arange(4)})
        return {'channel': 1}, out
    
    def algorithm(self, signal):
        def _mi(x1,x2, bins=8):
            c_xy = _np.histogram2d(x1, x2, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            return mi
        
        kf = self._params['kf']
        nbins = self._params['nbins']
        
        signal_values = signal.p.main_signal.values
        signal_functional_out = _np.zeros_like(signal_values)
        signal_systemic_out = _np.zeros_like(signal_values)
        
        ks_grid = _np.arange(0,5,0.01)
        ks_ = []
        
        n_channels = signal.sizes['channel']
        for i_ch in range(n_channels):
            cmin = _np.inf
            ks_min = ks_grid[0]
            signal_ch = signal_values[:,i_ch,:]
            
            done=False
            counter_up=0
            i_grid=0
            c_ = []
            while not done:
                ks = ks_grid[i_grid]
                p = _np.dot(signal_ch, _np.linalg.inv(_np.array([[1,ks],[1,kf]])))
                c = _mi(p[:,0],p[:,1], nbins)
                c_.append(c)
                if c < cmin:
                    cmin = c
                    ks_min = ks
                else:
                    counter_up +=1
                i_grid +=1
                
                #I can stop after I found the first minimum
                if counter_up == 10:
                    done=True
            
            p = _np.dot(signal_ch, _np.linalg.inv(_np.array([[1,ks_min],[1,kf]])))
            ks_.append(ks_min)
            
            signal_systemic_out[:,i_ch, 0] = p[:,0]
            signal_systemic_out[:,i_ch, 1] = ks*p[:,0]
            
            signal_functional_out[:, i_ch, 0] = p[:,1]
            signal_functional_out[:, i_ch, 1] = kf*p[:,1]
        
        signal_out = _np.concatenate([signal_functional_out, signal_systemic_out], axis=2)
        # signal_out = signal.clone_properties(signal_out)
        # signal_out.update_info('ks', ks_)
        out = signal.copy(deep=True)
        out = out.pad(component=(1,1), mode='edge')
        out = out.assign_coords(component=_np.arange(4))
        out.values = signal_out
        
        self._params['ks'] = ks_
        
        return out
"""