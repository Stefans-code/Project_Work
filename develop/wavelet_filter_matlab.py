import numpy as np
import matplotlib.pyplot as plt

dod = np.loadtxt('/home/bizzego/tmp/nirs.txt', delimiter='\t').astype(np.double)

iqr = 1.5;

SignalLength = dod.shape[0]
N = np.ceil(np.log2(SignalLength))
DataLength = int(2**N)
DataPadded = np.zeros(DataLength).astype(np.double) #% data length should be power of 2  

db2 = np.array([0.3415, 0.5915, 0.1585, -0.0915]).astype(np.double)
qmfilter = np.array([-0.0915, -0.1585, 0.5915, -0.3415]).astype(np.double)
L = 4;  # Lowest wavelet scale used in the analysis

idx_ch = 0
DataPadded[:SignalLength] = dod #% zeros pad data to have length of power of 2   

DCVal = np.mean(DataPadded);
DataPadded = DataPadded-DCVal;#    % removing mean value

y = DataPadded
qmf = qmfilter    

n = y.shape[0]
c = np.convolve(np.tile(y,2),qmf)[n:2*n]; #% circular convolution (final length = length(y))

y_downsampled = c[1::2]# % downsample by 2

meanAbsDev = np.mean(abs(y_downsampled - np.mean(y_downsampled))); #TODO: MEAN OR MEDIAN? CHECK HOMER3
    
if (meanAbsDev != 0):
    y_norm =  (1/1.4826)*y/meanAbsDev;
    coeff = 1/(1.4826*meanAbsDev);
else:
    y_norm = y;
    coeff = 1;
	
NormCoeff = coeff

x = y_norm
wavename = 'db2'

D = int(N-L);
n = len(x);

wp = np.zeros((n,D+1));

wp[:,0] = x

import pywt
for d in range(D):
    n_blocks = 2**d; # % number of blocks in the level
    l_blocks = int(n/n_blocks); #% length of the blocks in the level
    
    for b in range(2**d):
        
        s = wp[b*l_blocks:b*l_blocks+l_blocks,0]#; % first time take signal, from the second the approximation
        s_shift = np.array([s[-1]] + list(s[:-1]))# % create a shift version of the block
        
        
        [cA,cD] = pywt.dwt(s,wavename, mode='periodization');#  % discrete wavelet transform
        [cA_shift,cD_shift] = pywt.dwt(s_shift,wavename, mode='periodization'); # % discrete wavelet transform of the shifted version
        
        wp[b*l_blocks : b*l_blocks+l_blocks//2,0] = cA;
        wp[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks, 0] = cA_shift;
        
        wp[b*l_blocks:b*l_blocks+l_blocks//2,d+1] = cD;
        wp[b*l_blocks+l_blocks//2:b*l_blocks+l_blocks,d+1] = cD_shift;


count = 0
SignalLength_tmp = SignalLength
for d in np.arange(1, D): #AS BEFORE, but skipping d=0
    SignalLength_tmp = SignalLength_tmp//2
    n_blocks = 2**d #; % number of blocks in the level
    l_blocks = int(n/n_blocks)#; % length of the blocks in the level
        
    # for b=0:(2^j-1)       
    for b in range(2**d):
        sr = wp[b*l_blocks:b*l_blocks+l_blocks,d];
        sr_temp = sr[:SignalLength_tmp]#; % compute statistics only on original data
        quants = np.quantile(sr_temp, [.25, .50, .75],
                             method='hazen');#  % compute quantiles
        IQR = quants[2]-quants[0]#;  % compute interquartile range
        prob1 = quants[2]+IQR*iqr;
        prob2 = quants[0]-IQR*iqr; 
        outliers_1 = np.where(sr>prob1)[0]
        outliers_2 = np.where(sr<prob2)[0]
        outliers = np.concatenate([outliers_1, outliers_2]);
        
        sr[outliers] = 0 #;  % set outliers to 0
        wp[b*l_blocks:b*l_blocks+l_blocks,d] = sr;        
        count+=1

approx = wp[:,0]#)'; % approximation coefficients in the first column

for d in range(D-1, -1, -1):
    n_blocks = 2**d;
    l_blocks = int(n/n_blocks)
    for b  in range(2**d):
        cD = wp[b*l_blocks : b*l_blocks+l_blocks//2, d+1]
        cD_shift = wp[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks,d+1]
        cA = approx[b*l_blocks : b*l_blocks+l_blocks//2]
        cA_shift = approx[b*l_blocks+l_blocks//2 : b*l_blocks+l_blocks]
        
        s1 = pywt.idwt(cA,cD, wavename,
                       mode='periodization')#; % discrete inverse wavelet transform
        s_shift = pywt.idwt(cA_shift,cD_shift, wavename,
                       mode='periodization')#; % discrete inverse wavelet transform of the shifted version
        s2 = np.array(list(s_shift[1:]) + [s_shift[0]])#; % reshifting the shifted version 
        
        approx[b*l_blocks:b*l_blocks+l_blocks] = (s1+s2)/2#; % reconstruct the approximation of the next level

approx = approx/NormCoeff+DCVal;           

dodWavelet = approx[:len(dod)]

#%% ALL FINE UNTIL HERE
# except approx issues on low significance digits (e-15)