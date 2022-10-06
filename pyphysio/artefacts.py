import numpy as _np
from ._base_algorithm import _Algorithm

from .filters import IIRFilter as _IIRFilter
import pywt
from scipy.stats import median_abs_deviation as _mad

#TODO: There could be three types of classes:
# - DetectNAME (to detect artefacts), 
# - CorrectNAME (to correct detected artefacts), and
# - NAME (algorithm that do both)
# see Di Lorenzo et al: https://www.sciencedirect.com/science/article/pii/S1053811919305531?via%3Dihub


class MARA(_Algorithm):
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
        from csaps import csaps as _csaps
        
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