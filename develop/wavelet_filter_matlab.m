format long
dod = importdata('/home/bizzego/tmp/nirs.txt');

iqr = 1.5;
SignalLength = size(dod,1); % #time points of original signal
N = ceil(log2(SignalLength)); % #of levels for the wavelet decomposition
DataPadded = zeros (2^N,1); % data length should be power of 2  

load('/home/bizzego/Downloads/db2.mat')
qmfilter = qmf(db2,4);
L = 4;  % Lowest 
% wavelet scale used in the analysis

DataPadded(1:SignalLength) = dod;  % zeros pad data to have length of power of 2   
DataPadded(SignalLength+1:end) = 0;  

DCVal = mean(DataPadded);
DataPadded = DataPadded-DCVal;    % removing mean value

DataLength = size(DataPadded,1);  

y = DataPadded';
c = cconv(y,qmfilter,length(y)); % circular convolution (final length = length(y))

c_downsampled = dyaddown(c); % downsample by 2

medianAbsDev = mad(c_downsampled);

if medianAbsDev ~= 0
	yn =  (1/1.4826).*y./medianAbsDev;
    NormCoeff = 1/(1.4826*medianAbsDev);
else
	yn = y;
    NormCoeff = 1;
end

wavename='db2';
D = N-L;
n = length(yn);
wp = zeros(n,D+1);

wp(:,1) = yn';

dwtmode('per');  % set the wavelet mode to periodization


for d=0:(D-1)
    n_blocks = 2^d; % number of blocks in the level
    l_blocks = n/n_blocks; % length of the blocks in the level
    for b=0:(2^d-1) 
        s = wp(b*l_blocks+1:b*l_blocks+l_blocks,1)'; % first time take signal, from the second the approximation
        s_shift = [s(end) s(1:end-1)]; % create a shift version of the block
        
        [cA,cD] = dwt(s,wavename);  % discrete wavelet transform
        [cA_shift,cD_shift] = dwt(s_shift,wavename); % discrete wavelet transform of the shifted version

        wp(b*l_blocks+1:b*l_blocks+l_blocks/2,1) = cA;
        wp(b*l_blocks+l_blocks/2+1:b*l_blocks+l_blocks,1) = cA_shift;
        
        wp(b*l_blocks+1:b*l_blocks+l_blocks/2,d+2) = cD;
        wp(b*l_blocks+l_blocks/2+1:b*l_blocks+l_blocks,d+2) = cD_shift;
    end
end

n=size(wp,1);       % Length of data vector with zero padding
N=log2(size(wp,1)); % Finest scale (original signal)
SignalLength_tmp = SignalLength;
count=0;
for j=1:N-L-1
    SignalLength_tmp = fix(SignalLength_tmp/2);
    n_blocks = 2^j; % number of blocks in the level
    l_blocks = n/n_blocks; % length of the blocks in the level
    for b=0:(2^j-1)   
        sr = wp(b*l_blocks+1:b*l_blocks+l_blocks,j+1);
        sr_temp = sr(1:SignalLength_tmp); % compute statistics only on original data
        quants = quantile(sr_temp,[.25 .50 .75]);  % compute quantiles
        IQR = quants(3)-quants(1);  % compute interquartile range
        prob1 = quants(3)+IQR*iqr;
        prob2 = quants(1)-IQR*iqr; 
        outliers_1 = find(sr>prob1);
        outliers_2 = find(sr<prob2);
        outliers = [outliers_1' outliers_2'];
        
        sr(outliers) = 0;  % set outliers to 0
        wp(b*l_blocks+1:b*l_blocks+l_blocks,j+1) = sr;
        count = count+1;
    end
end

%
[n,D] = size(wp);
D = D-1;

dwtmode('per');

approx = wp(:,1)'; % approximation coefficients in the first column
for d = D-1:-1:0
    n_blocks = 2^d;
    l_blocks = n/n_blocks;
    for b = 0:(2^d-1)
        cD = wp(b*l_blocks+1  :  b*l_blocks+l_blocks/2,d+2)';
        cD_shift = wp(b*l_blocks+l_blocks/2+1:b*l_blocks+l_blocks,d+2)';
        cA = approx(b*l_blocks+1:b*l_blocks+l_blocks/2);
        cA_shift = approx(b*l_blocks+l_blocks/2+1:b*l_blocks+l_blocks);
        
        s1 = idwt(cA,cD,wavename); % discrete inverse wavelet transform
        s_shift = idwt(cA_shift,cD_shift,wavename); % discrete inverse wavelet transform of the shifted version
        s2 = [s_shift(2:end) s_shift(1)]; % reshifting the shifted version 
        
        approx(b*l_blocks+1:b*l_blocks+l_blocks) = (s1+s2)/2; % reconstruct the approximation of the next level
    end
end
x = approx;

x = x/NormCoeff+DCVal;           

dodWavelet = x(1:length(dod));

%% ALL FINE UNTIL HERE
% except approx issues on low significance digits (e-15)

%%
%
dodWavelet_py = importdata('/home/bizzego/tmp/nirs_wav.txt');

plot(dod)
hold on
plot(dodWavelet)
hold on
plot(dodWavelet_py)
hold off

%%
