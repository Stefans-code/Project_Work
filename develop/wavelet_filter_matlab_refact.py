import numpy as np
import matplotlib.pyplot as plt
import pywt

dod = np.loadtxt('/home/bizzego/tmp/nirs.txt', delimiter='\t')
nsamples = dod.shape[0]

iqr = 1.5;
L = 4;  # Lowest wavelet scale used in the analysis
N = np.ceil(np.log2(nsamples))
D = int(N-L);

wavename = 'db2'


qmfilter = np.array([-0.0915, -0.1585, 0.5915, -0.3415])

#%% create padded signal
nsamples_out = int(2**N)
signal_padded = np.zeros(nsamples_out) #% data length should be power of 2  
signal_padded[:nsamples] = dod #% zeros pad data to have length of power of 2   

# removing mean value
mean_padded = np.mean(signal_padded);
signal_padded = signal_padded-mean_padded;


c = np.convolve(np.tile(signal_padded,2),qmfilter)[nsamples_out:2*nsamples_out] 
# downsample by 2
c_downsampled = c[1::2]

#compute mean abs dev
meanAbsDev = np.mean(abs(c_downsampled - np.mean(c_downsampled)))
    
# normalize
if (meanAbsDev != 0):
    signal_padded =  (1/1.4826)*signal_padded/meanAbsDev;
    norm_coeff = 1/(1.4826*meanAbsDev);
else:
    signal_padded = signal_padded;
    norm_coeff = 1;
	
#%% compute wavelets coefficients
wp = np.zeros((nsamples_out,D+1));

wp[:,0] = signal_padded

for d in range(D):
    n_blocks = 2**d; # number of blocks in the level
    l_blocks = int(nsamples_out/n_blocks); # length of the blocks in the level
    
    for b in range(2**d):
        
        # first time take signal, from the second the approximation
        s = wp[b*l_blocks:b*l_blocks+l_blocks,0]
        # create a shift version of the block
        s_shift = np.array([s[-1]] + list(s[:-1]))
        
        
        # discrete wavelet transform
        [cA,cD] = pywt.dwt(s,wavename, mode='periodization')
        # discrete wavelet transform of the shifted version
        [cA_shift,cD_shift] = pywt.dwt(s_shift,wavename, mode='periodization')
        
        #store values in wp
        wp[b*l_blocks : b*l_blocks+l_blocks//2,0] = cA
        wp[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks, 0] = cA_shift
        
        wp[b*l_blocks:b*l_blocks+l_blocks//2,d+1] = cD
        wp[b*l_blocks+l_blocks//2:b*l_blocks+l_blocks,d+1] = cD_shift

#%% filter outliers of wavelets coefficients
nsamples_tmp = nsamples
for d in np.arange(1, D): #AS BEFORE, but skipping d=0
    nsamples_tmp = nsamples_tmp//2
    n_blocks = 2**d
    l_blocks = int(nsamples_out/n_blocks)
    
    for b in range(2**d):
        sr = wp[b*l_blocks:b*l_blocks+l_blocks,d]
        # compute statistics only on original data
        sr_temp = sr[:nsamples_tmp]
        
        # compute quantiles
        quants = np.quantile(sr_temp, [.25, .50, .75],
                             method='hazen')
        
        # compute interquartile range
        IQR = quants[2]-quants[0]
        prob1 = quants[2]+IQR*iqr
        prob2 = quants[0]-IQR*iqr
        
        #get outliers
        outliers_1 = np.where(sr>prob1)[0]
        outliers_2 = np.where(sr<prob2)[0]
        outliers = np.concatenate([outliers_1, outliers_2])
        
        # set outliers to 0
        sr[outliers] = 0 
        wp[b*l_blocks:b*l_blocks+l_blocks,d] = sr

#%% reconstruct signal
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
        s2 = np.array(list(s_shift[1:]) + [s_shift[0]])
        
        # reconstruct the approximation of the next level
        approx[b*l_blocks:b*l_blocks+l_blocks] = (s1+s2)/2

#restore original scale
approx = approx/norm_coeff+mean_padded

dodWavelet = approx[:nsamples]

#%% ALL FINE UNTIL HERE
# except approx issues on low significance digits (e-15)