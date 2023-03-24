import numpy as _np
from ._base_algorithm import _Algorithm
from copy import deepcopy as copy
from .filters import IIRFilter as _IIRFilter, Normalize as _Normalize
import pywt
from scipy.stats import median_abs_deviation as _mad, iqr as _iqr

#TODO: There could be three types of classes:
# - DetectNAME (to detect artefacts), 
# - CorrectNAME (to correct detected artefacts), and
# - NAME (algorithm that do both)
# see Di Lorenzo et al: https://www.sciencedirect.com/science/article/pii/S1053811919305531?via%3Dihub

class DetectMA(_Algorithm):
    '''
    F Scholkmann et al 2010 Physiol. Meas. 31 649
    '''
    def __init__(self, win_len=1, win_mask=1, method='iqr',
                 iqr=1.5,
                 th_std = None, th_std_coeff=None, 
                 th_amp = None, fuse=False, **kwargs):
        
        assert method in ['iqr', 'mad', 'fixed']
        if method == 'fixed':
            assert th_std is not None
            assert th_amp is not None
        if method == 'mad':
            assert th_std_coeff is not None
            th_amp = _np.Inf #deactivate detection based on AMP
            #TODO: ideas on how to compute threshold for AMP?
            
        _Algorithm.__init__(self, win_len=win_len, win_mask=win_mask,
                            method=method, iqr=iqr,
                            th_std=th_std, th_std_coeff=th_std_coeff, 
                            th_amp=th_amp, **kwargs)
        
        #IDEA for the MA detection, we can do that by channel or globally
        #(using fused channels)
        #and adapt the behaviour of the algorithm on the different dimensions:
        if fuse:
            self.dimensions = {'time' : 0, 'channels':1, 'components':1}
        else:
            self.dimensions = {'time' : 0}
   
    
    def algorithm(self, signal):
        params = self._params
        win_len = params['win_len']
        win_mask = params['win_mask']
        method = params['method']
        th_std = params['th_std']
        th_std_coeff= params['th_std_coeff']
        th_amp = params['th_amp']
        iqr = params['iqr']
        
        fsamp = signal.p.get_sampling_freq()
                
        signal_norm = _Normalize()(signal)
        signal_values = signal_norm.values.ravel()
        
        
        # compute moving standard deviation (MSD) and range (AMP)
        idx_len = int(win_len*fsamp)
        half = idx_len // 2
        MSD = []
        AMP = []
        for i in range(len(signal_values) - idx_len):
            signal_values_win = signal_values[i: i+idx_len]
            MSD.append(_np.std(signal_values_win))
            AMP.append(_np.max(signal_values_win) - _np.min(signal_values_win))
        
        MSD = _np.array(MSD)
        AMP = _np.array(AMP)
        
        #compute thresholds
        if method == 'iqr':
            #get thresholds using iqr
            quants = _np.quantile(MSD, [.25, .50, .75])
            IQR = quants[2]-quants[0]
            th_MSD = quants[2]+IQR*iqr
            
            quants = _np.quantile(AMP, [.25, .50, .75])
            IQR = quants[2]-quants[0]
            th_AMP = quants[2]+IQR*iqr
        
        elif method == 'mad':    
            signal_f =  _Normalize()(_IIRFilter(fp = [0.01, 0.5], btype='bandpass')(signal))
            signal_values_filt = signal_f.values.ravel()
            th_MSD = th_std_coeff*_np.median(abs(signal_values_filt - _np.median(signal_values_filt)))
            th_AMP = th_amp
            
        else:
            th_MSD = th_std
            th_AMP = th_amp
        
        # fig, axes = plt.subplots(2,1,sharex=True)
        # axes[0].plot(data_ch)
        # axes[1].plot(np.arange(len(MA))+half, MA)
        # axes[1].plot(np.arange(len(MSD))+half, MSD)
        # axes[1].plot(np.arange(len(MSD))+half, AMP)
        # axes[1].hlines(threshold, 0, len(MSD))
        # axes[1].hlines(1.5, 0, len(MSD))
        
        # 2 detection motion artifacts (MA)
        #IDEA: use peakdetection to identify the MA (on the MSD and AMP)
        
        # MSD = MSD - _np.median(MSD) #TODO: needed for method = 'mad'?
        MSD = (MSD >= th_MSD)
        AMP = (AMP > th_AMP)
        
        MA = (AMP | MSD).astype(int)
        
        idxlen_smooth = int(win_mask*fsamp)
        MA = _np.convolve(MA, _np.ones(idxlen_smooth)/idxlen_smooth, 'same')
        
        signal_out = _np.zeros(len(signal_values))
        idx_MA = _np.where(MA>0)[0] + half
        signal_out[idx_MA] = 1
        return(signal_out)
        
        
class MARA(_Algorithm):
    '''
    F Scholkmann et al 2010 Physiol. Meas. 31 649
    '''
    def __init__(self, MA, **kwargs):
        _Algorithm.__init__(self, MA=MA, **kwargs)
        self.dimensions = {'time' : 0}
    
    def algorithm(self, signal):
        from csaps import csaps as _csaps

        signal_values = signal.values.ravel()
        fsamp = signal.p.get_sampling_freq()
        
        params = self._params
        MA = params['MA']
        
        ch = signal.channel.values[0]
        cp = signal.component.values[0]
        MA_signal = MA.isel(channel=[ch], component=[cp])
        MA_signal_values = MA_signal.p.main_signal.values.ravel()
        MA_signal_diff = _np.diff(MA_signal_values)
        idx_st = _np.where(MA_signal_diff > 0)[0]
        idx_sp = _np.where(MA_signal_diff < 0)[0]
        
        #manage special cases with MA at beginning or end
        #TODO: check
        if len(idx_sp)>0:
            if ((len(idx_st)==0) or (idx_sp[0] < idx_st[0])): #starting with a MA
                idx_st = _np.insert(idx_st, 0, [0])
        
        if len(idx_st)>0:
            if ((len(idx_sp)==0) or (idx_sp[-1] < idx_st[-1])): #ending with a MA
                idx_sp = _np.insert(idx_sp, len(idx_sp), len(MA_signal))
        
        
        # 3 create list of segments w/ MA x_bad and w/o MA x_good
        x_good = []
        x_bad = []
        idx_start = 0
        for id_MA, (idx_st_MA, idx_sp_MA) in enumerate(zip(idx_st,idx_sp)):
            # if (idx_sp_MA - idx_st_MA) < 2:
            #     plt.plot(signal.values.ravel())
            x_good.append(signal_values[idx_start: idx_st_MA])
            x_bad.append(signal_values[idx_st_MA : idx_sp_MA])
            idx_start = idx_sp_MA
        x_good.append(signal_values[idx_start:])
        
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
        #% normalize using computed mean abs dev

        #values taken from MATLAB and homer2/3 installer
        qmf = _np.array([-0.0915, -0.1585, 0.5915, -0.3415])
        
        n = len(y)
        # circular convolution (final length = length(y))
        c = _np.convolve(_np.tile(y, 2), qmf)[n:2*n]

        # downsample by 2
        c_downsampled = c[1::2]

        #compute mean abs dev
        meanAbsDev = _np.mean(abs(c_downsampled - _np.mean(c_downsampled)))

        if meanAbsDev !=0:
            y_norm = (1/1.4826)*y/meanAbsDev
            coeff = 1/(1.4826*meanAbsDev)
        else:
            y_norm = y
            coeff = 1
        return(y_norm, coeff)
    
    def algorithm(self, signal): 
        signal_values = signal.values.ravel()
        
        #%
        mean_original = _np.mean(signal_values)
        signal_values = signal_values - mean_original
        #%
        
        nsamples = len(signal_values)
        
        #params
        iqr = self._params['iqr']
        L = 4;  # Lowest wavelet scale used in the analysis
        N = _np.ceil(_np.log2(nsamples))
        D = int(N-L)
        wavename = 'db2'
        
        #% create padded signal
        nsamples_out = int(2**N)
        signal_padded = _np.zeros(nsamples_out) #% data length should be power of 2  
        signal_padded[:nsamples] = signal_values #% zeros pad data to have length of power of 2   

        # removing mean value
        mean_padded = _np.mean(signal_padded);
        signal_padded = signal_padded-mean_padded;

        #% normalize
        signal_padded, norm_coeff = self._normalization_noise(signal_padded)
        
        #% compute wavelets coefficients
        wp = _np.zeros((nsamples_out,D+1));

        wp[:,0] = signal_padded

        for d in range(D):
            n_blocks = 2**d; # number of blocks in the level
            l_blocks = int(nsamples_out/n_blocks); # length of the blocks in the level
            
            for b in range(2**d):
                
                # first time take signal, from the second the approximation
                s = wp[b*l_blocks:b*l_blocks+l_blocks,0]
                # create a shift version of the block
                s_shift = _np.array([s[-1]] + list(s[:-1]))
                
                
                # discrete wavelet transform
                [cA,cD] = pywt.dwt(s,wavename, mode='periodization')
                # discrete wavelet transform of the shifted version
                [cA_shift,cD_shift] = pywt.dwt(s_shift,wavename, mode='periodization')
                
                #store values in wp
                wp[b*l_blocks : b*l_blocks+l_blocks//2,0] = cA
                wp[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks, 0] = cA_shift
                
                wp[b*l_blocks:b*l_blocks+l_blocks//2,d+1] = cD
                wp[b*l_blocks+l_blocks//2:b*l_blocks+l_blocks,d+1] = cD_shift

        #% filter outliers of wavelets coefficients
        nsamples_tmp = nsamples
        for d in _np.arange(1, D): #AS BEFORE, but skipping d=0
            nsamples_tmp = nsamples_tmp//2
            n_blocks = 2**d
            l_blocks = int(nsamples_out/n_blocks)
            
            for b in range(2**d):
                sr = wp[b*l_blocks:b*l_blocks+l_blocks,d]
                # compute statistics only on original data
                sr_temp = sr[:nsamples_tmp]
                
                # compute quantiles
                quants = _np.quantile(sr_temp, [.25, .50, .75],
                                     method='hazen')
                
                # compute interquartile range
                IQR = quants[2]-quants[0]
                prob1 = quants[2]+IQR*iqr
                prob2 = quants[0]-IQR*iqr
                
                #get outliers
                outliers_1 = _np.where(sr>prob1)[0]
                outliers_2 = _np.where(sr<prob2)[0]
                outliers = _np.concatenate([outliers_1, outliers_2])
                
                # set outliers to 0
                sr[outliers] = 0 
                wp[b*l_blocks:b*l_blocks+l_blocks,d] = sr
        
        #% reconstruct signal
        approx = wp[:,0]#)'; % approximation coefficients in the first column

        for d in range(D-1, -1, -1):
            n_blocks = 2**d;
            l_blocks = int(nsamples_out/n_blocks)
            for b  in range(2**d):
                #get coefficients
                cD = wp[b*l_blocks : b*l_blocks+l_blocks//2, d+1]
                cD_shift = wp[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks,d+1]
                cA = approx[b*l_blocks : b*l_blocks+l_blocks//2]
                cA_shift = approx[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks]
                
                
                # discrete inverse wavelet transform
                s1 = pywt.idwt(cA,cD, wavename,
                               mode='periodization')
                # discrete inverse wavelet transform of the shifted version
                s_shift = pywt.idwt(cA_shift,cD_shift, wavename,
                               mode='periodization')
                
                # reshifting the shifted version 
                s2 = _np.array(list(s_shift[1:]) + [s_shift[0]])
                
                # reconstruct the approximation of the next level
                approx[b*l_blocks:b*l_blocks+l_blocks] = (s1+s2)/2

        #restore original scale
        approx = approx/norm_coeff+mean_padded
        
        reconstructed = approx[:nsamples] + mean_original
        
        return(reconstructed)