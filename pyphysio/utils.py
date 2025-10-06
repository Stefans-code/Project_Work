# coding=utf-8
# from __future__ import division
import numpy as _np
import xarray as _xr
from scipy.signal import welch as _welch, periodogram as _periodogram, \
    detrend as _detrend
# import pycwt.wavelet as wave
import pywt as _pywt
#TODO replace with pywavelets
from sklearn.decomposition import PCA as _PCA
from ._base_algorithm import _Algorithm
from .filters import _Filter

class Diff(_Filter): #xarray done
    """
    Computes the differences between adjacent samples.

    Optional parameters
    -------------------
    degree : int, >0, default = 1
        Sample interval to compute the differences
    
    Returns
    -------
    signal : 
        Differences signal. 

    """

    def __init__(self, degree=1):
        assert degree > 0, "The degree value should be positive"
        _Filter.__init__(self, degree=degree)

    def algorithm(self, signal):
        """
        Calculates the differences between consecutive values
        """
        
        signal_values = signal.values.ravel()
        params = self._params
        degree = params['degree']

        sig_1 = signal_values[:-degree]
        sig_2 = signal_values[degree:]

        diff = sig_2 - sig_1
        out = _np.ones(len(signal_values))*diff[-1]
        out[:len(diff)] = diff

        return out
    
class PeakDetection(_Filter): #xarray done
    """
    Estimate the maxima and the minima in the signal (in particular for periodic signals).

    Parameters
    ----------
    delta : float or list
        Threshold for the detection of the peaks. If it is a list it must have the same length of the signal.
        
    Optional parameters
    -------------------
    refractory : float, >=0, default = 0
        Seconds to skip after a detected paek to look for new peaks.
    start_max : boolean, default = True
        Whether to start looking for a maximum or (False) for a minimum.

    Returns
    -------
    maxp : numpy.array
        Array containing indexes of the maxima
    minp : numpy.array
        Array containing indexes of the minima
    maxv : numpy.array
        Array containing values of the maxima
    minv : numpy.array
        Array containing values of the minima
    """

    def __init__(self, delta, refractory=0, return_peaks = True):
        delta = _np.array(delta)
        assert delta.ndim <= 1, "Delta value should be 1 or 0-dimensional"
        assert delta.all() > 0, "Delta value/s should be positive"
        assert refractory >= 0, "Refractory value should be non negative"
        _Filter.__init__(self, delta=delta, refractory=refractory, return_peaks=return_peaks)
        
    def algorithm(self, signal):
        params = self._params
        refractory = params['refractory']
        if refractory == 0:  # if 0 then do not skip samples
            refractory = 1
        else:  # else transform the refractory from seconds to samples
            refractory = refractory * signal.p.get_sampling_freq()
        # look_for_max = params['start_max']
        delta = params['delta']
        return_peaks = params['return_peaks']
        
        signal_values = signal.p.get_values().ravel()
        
        if return_peaks == False: #looking for valleys
            signal_values = -signal_values
            
        max_idxs = []
        max_vals = []

        scalar = delta.ndim == 0
        if scalar:
            d = delta
        else:
            assert len(delta) == len(signal), "delta vector's length differs from signal's one, returning empty."
        
        
        mx_candidate_idx = 0
        mx_candidate_val = signal_values[mx_candidate_idx]
        i_activation_max = mx_candidate_idx

        mn_candidate_idx = 0
        mn_candidate_val = signal_values[mn_candidate_idx]
        
        look_max = True
        
        for i in range(1, len(signal_values)):
            sample = signal_values[i]
            if not scalar:
                d = delta[i]

            #if value is greater, then update the candidate max
            if sample > mx_candidate_val:
                mx_candidate_val = sample
                mx_candidate_idx = i
            if sample < mn_candidate_val:
                mn_candidate_val = sample
                mn_candidate_idx = i
            
            #if we are looking for the max,
            #and we are outside the refractory period,
            #and current value is lower than (candidate max - d)
            #we validate the candidate maximum and store it
            #and we update the candidate minimum
            if look_max:
                if i >= i_activation_max and sample < mx_candidate_val - d:  
                    max_idxs.append(mx_candidate_idx)
                    max_vals.append(mx_candidate_val)
                    i_activation_max = i + refractory
    
                    mn_candidate_val = sample
                    mx_candidate_idx = i
    
                    look_max = False
            
            else: #we are looking for a min
                if sample > mn_candidate_val + d:  # new min
                    mx_candidate_val = sample
                    mx_candidate_idx = i

                    look_max = True
    
        out = _np.ones_like(signal_values)*_np.nan
        
        if return_peaks:
            out[max_idxs] = _np.array(max_vals)
        else:
            out[max_idxs] = -1*_np.array(max_vals)
        
        return out

class SignalRange(_Filter): #xarray done
    """
    Estimate the local range of the signal by sliding windowing

    Parameters
    ----------
    win_len : float, >0
        Length of the window  in seconds
    win_step : float, >0
        Shift to start the next window in seconds

    Optional parameters
    -------------------    
    smooth : boolean, default=True
        Whether to convolve the result with a gaussian window

    Returns
    -------
    deltas : numpy.array
        Local range of the signal
    """

    def __init__(self, win_len, win_step, smooth=True):
        assert win_len > 0, "Window length should be positive"
        assert win_step > 0, "Window step should be positive"
        _Filter.__init__(self, win_len=win_len, win_step=win_step, smooth=smooth)
        
    def algorithm(self, signal):
        params = self._params
        win_len = params['win_len']
        win_step = params['win_step']
        smooth = params['smooth']

        fsamp = signal.p.get_sampling_freq()
        idx_len = int(win_len * fsamp)
        idx_step = int(win_step * fsamp)
        
        signal_values = signal.values

        # print('>>> signalrange')
        deltas = _np.zeros(len(signal_values))
        
        if len(signal) < idx_len:
            print("Input signal is shorter than the window length.")
            deltas = deltas + (_np.max(signal_values) - _np.min(signal_values))
        else:
            windows = _np.arange(0, len(signal_values) - idx_len + 1, idx_step)
            

            curr_delta = 0
            for start in windows:
                portion_curr = signal_values[start: start + idx_len]
                curr_delta = _np.max(portion_curr) - _np.min(portion_curr)
                deltas[start:start + idx_len] = curr_delta

            deltas[windows[-1] + idx_len:] = curr_delta
            
            if smooth:
                win_len = int(win_len*2*fsamp)
                deltas = _np.convolve(deltas, _np.ones(win_len)/win_len, mode='same')
                deltas = deltas[:len(signal_values)]
            # print('<<< signalrange')
            
        return deltas

class PSD(_Algorithm): #xarray done
    """
    Estimate the power spectral density (PSD) of the signal.

    Parameters
    ----------
    method : str
        Method to estimate the PSD. Available methods: 'welch', 'period'
        
    Optional parameters
    -------------------
    
    nfft : int, >0, default=2048
        Number of samples of the PSD
    window : str, default = 'hamming'
        Type of window
    min_order : int, >0, default=18
        Minimum order of the model to be tested for psd_method='ar'
    max_order : int, >0, default=25
        Maximum order of the model to be tested for psd_method='ar'
    normalize : boolean, default = True
        Whether to normalize the PSD
    remove_mean : boolean, default = True
        Whether to remove the mean from the signal before estimating the PSD
    
    Returns
    -------
    freq : numpy.array
        Frequencies
    psd : numpy.array
        Power Spectrum Density
    """

    def __init__(self, method, nfft=2048, window='hamming', min_order=10, max_order=30,
                 remove_mean=True, scaling='density', **kwargs):
        
        _method_list = ['welch', 'period', 'ar']
        _window_list = ['hamming', 'blackman', 'hanning', 'bartlett', 'boxcar']

        assert method in _method_list, "Parameter method should be in " + _method_list.__repr__()
        assert nfft > 0, "nfft value should be positive"
        assert window in _window_list, "Parameter window type should be in " + _window_list.__repr__()
        if method == "ar":
            assert min_order > 0, "Minimum order for the AR method should be positive"
            assert max_order > 0, "Maximum order for the AR method should be positive"
        
        assert scaling in ['density', 'spectrum']
        _Algorithm.__init__(self, method=method, nfft=nfft, window=window, min_order=min_order,
                       max_order=max_order, remove_mean=remove_mean, scaling=scaling, **kwargs)
        
        self.required_dims = ['time']
    
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        
        nfft = self._params['nfft']
        N = int(nfft/2 + 1)
        fsamp = signal.p.get_sampling_freq()
        freqs = _np.linspace(start=0, stop=fsamp / 2, num=N)
        
        template = self.__compute_template__(signal, {'time':1, 'freq': freqs})
        return(chunk_dict, template)
   
    def algorithm(self, signal):
        # print('----->', self.name)
        params = self._params
        method = params['method']
        nfft = params['nfft'] if "nfft" in params else None
        window = params['window']
        remove_mean = params['remove_mean']
        scaling = params['scaling']
        
        fsamp = signal.p.get_sampling_freq()
        
        signal_values = signal.values.ravel()
        
        if remove_mean:
            signal_values = signal_values - _np.mean(signal_values)

        if method == 'period':
            freqs, psd = _periodogram(signal_values, fs=fsamp, window = window, 
                                      nfft=nfft, return_onesided=True, scaling=scaling)

        elif method == 'welch':
            freqs, psd = _welch(signal_values, fs=fsamp, window=window, 
                                nfft=nfft, return_onesided=True, scaling=scaling)

        elif method == 'ar':
            raise NotImplementedError
            
            #TODO CHECK THAT IT IS CORRECT
            '''
            # print("Using AR method: results might not be comparable with other methods")
            #methods derived from: https://github.com/mpastell/pyageng
            def autocorr(x, lag=30):
                c = _np.correlate(x, x, 'full')
                mid = len(c)//2
                acov = c[mid:mid+lag]
                acor = acov/acov[0]
                return(acor)
                
            def aryw(x, order=30):
                x = x - _np.mean(x)
                ac = autocorr(x, order+1)
                R = _linalg.toeplitz(ac[:order])
                r = ac[1:order+1]
                params = _np.linalg.inv(R).dot(r)
                return(params)
                
            def AIC_yule(signal_values, order):
                #this is from library spectrum: https://github.com/cokelaer/spectrum
                N = len(signal_values)
                assert N>=order, "The number of samples in the signal should be >= to the model order"
                
                C = _np.correlate(signal_values, signal_values, mode='full')/N
                r = C[N-1:]
                
                T0  = r[0]
                T = r[1:]
                
                A = _np.zeros(order, dtype=float)
                P = T0
                
                for k in range(0, order):
                    save = T[k]
                    if k == 0:
                        temp = -save / P
                    else:
                        for j in range(0, k):
                            save = save + A[j] * T[k-j-1]
                        temp = -save / P
                    
                    P = P * (1. - temp**2.)
                    A[k] = temp
                
                    khalf = (k+1)//2
                    for j in range(0, khalf):
                        kj = k-j-1
                        save = A[j]
                        A[j] = save + temp * A[kj]
                        if j != kj:
                            A[kj] += temp*save
                
                res = N * _np.log(P) + 2*(order + 1)
                return(res)
            
            min_order = params['min_order']
            max_order = params['max_order']
            
            if len(signal_values) <= max_order:
                # print("Input signal too short: try another 'method', a lower 'max_order', or a longer signal")
                freqs = _np.linspace(start=0, stop=fsamp / 2, num=1024)
                p = _np.repeat(_np.nan, 1024)
                return _np.squeeze(freqs), _np.squeeze(p)
            
            signal_values = signal.p.main_signal.values.ravel()
            orders = _np.arange(min_order, max_order + 1)
            aics = [AIC_yule(signal_values, x) for x in orders]
            best_order = orders[_np.argmin(aics)]

            params = aryw(signal_values, best_order)
            a = _np.concatenate([_np.ones(1), -params])
            w, P = _freqz(1, a, whole = False, worN = nfft)
            
            psd = 2*_np.abs(P)/fsamp
            
            freqs = _np.linspace(start=0, stop=fsamp / 2, num=len(psd))
            '''

        # NORMALIZE
        if scaling == 'density':
            psd /= len(psd)
      
        # out = signal.copy(deep=True)
        # out = out.expand_dims({'freq':freqs}, axis=0)[:,0]
        # out.name = signal.name+'_'+self.name
        # out.values = _np.expand_dims(psd,[1,2])
        
        
        psd = psd[_np.newaxis, :]
        # print(psd.shape)
        
        # out = signal.copy(deep=True)
        # out = out.expand_dims({'freq':freqs}, axis=0)[:,0]
        # out = out.drop('time')
        # # out.name = signal.name+'_'+self.name
        # # print(out)
        
        # out.values = psd
        # print('<-----', self.name)
        return psd
        # return out


    
class Wavelet(_Algorithm):
    """

    """
    def __init__(self, wtype = 'cmor_1.15-1.0',
                 freqs = None,
                 minScale = 2,
                 nNotes = 12,
                 detrend=True,
                 normalize=False,
                 compute_coi=False):
            
        
        _Algorithm.__init__(self, wtype = wtype, freqs = freqs,
                            minScale = minScale, nNotes = nNotes,
                            detrend=detrend, normalize=normalize,
                            compute_coi=compute_coi)
        self.required_dims = ['time']
        
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        
        self._compute_scales(signal)
        fsamp = signal.p.get_sampling_freq()
        freqs = self._params['freqs_nyq']*fsamp
        
        template = self.__compute_template__(signal, {'freq': freqs})        
        return(chunk_dict, template)
    
    def _compute_coi(self, W):
        W_out = W.copy(deep=True)
        N = W.sizes['time']
        
        freqs_nyq = self._params['freqs_nyq']
        coif_ = 1/(2*_np.arange(1, N//2))

        # coif_ = fsamp/(2*np.arange(1, N//2))
        min_coif = coif_[-1]
        coif = _np.zeros(N) + min_coif
        coif[:len(coif_)] = coif_
        coif[-len(coif_):] = coif_[::-1]
        
        freqs = W.coords['freq'].values
        for i, i_t in enumerate(W.coords['time']):
            idx_na = _np.where(freqs_nyq < coif[i])[0]
            W_out.loc[{'time': [i_t], 'freq': freqs[idx_na]}] = _np.nan
        return(W_out)
            
    
    def _compute_scales(self, signal):
        params = self._params
        freqs = params['freqs']
        wtype = params['wtype']
        
        signal_values = signal.p.get_values()
        fsamp = signal.p.get_sampling_freq()
        
        if freqs is None: #users want the algoritm to compute the scales
            minScale = params['minScale']
            nNotes = params['nNotes']
            
            # The scales as of Mallat 1999
            # minScale = 2 # / wavelet.flambda()
            N = signal_values.shape[0]
            nOctaves = int(_np.round(_np.log2(N/2) / (1/nNotes)))
            scales = minScale * 2 ** (_np.arange(0, nOctaves + 1) * (1/nNotes))
            freqs_nyq = _pywt.scale2frequency(wtype, scales)
            
        else: #user provided the frequencies
            freqs = _np.array(freqs)
            #check correct order of frequencies
            assert freqs[0]>freqs[-1]
            assert (_np.diff(freqs)<0).all()
            freqs_nyq = freqs/fsamp
            scales = _pywt.frequency2scale(wtype, freqs_nyq)
            
        scales = _np.sort(scales)
        
        self._params['freqs_nyq'] = freqs_nyq
        self._params['scales'] = scales
        
    
    def algorithm(self, signal):
        if 'scales' not in self._params:
            self._compute_scales(signal)
        
        params = self._params
        #get signal values and info
        signal_values = signal.p.get_values().ravel()
        fsamp = signal.p.get_sampling_freq()
        N = len(signal_values)
        
        #remove linear drift
        detrend = params['detrend']
        if detrend:
            signal_values = _detrend(signal_values, type='linear')
        
        #compute wavelet
        wtype = params['wtype']
        scales = params['scales']
        
        W, freqs_nyq = _pywt.cwt(signal_values, scales, wavelet=wtype)
        
        self._params['freqs_nyq'] = freqs_nyq
        
        #normalize computed W
        normalize = params['normalize']
        if normalize:
            scaleMatrix = _np.ones([1, N]) * scales[:, None]
            W = W**2 / scaleMatrix
        
        #compute coi and assign na outside
        compute_coi = params['compute_coi']
        if compute_coi:
            W = self._compute_coi(W)
                
        
        W = W.T
        # W = W[:, _np.newaxis, :]
        
        return W
        # W = _np.expand_dims(W,[2,3])
        
        # out = signal.copy(deep=True)
        
        # out = out.expand_dims({'freq':freqs.astype(_np.float64)}, axis=0)
        # # out.name = signal.name+'_'#+self.name
        
        # out.values = W
        # return out
    
    # def __get_template__(self, signal):
    #     self._compute_scales(signal)
        
    #     signal_values = signal.p.get_values()
    #     N = signal_values.shape[0]
    #     fsamp = signal.p.get_sampling_freq()
        
    #     scales = self._params['scales']
    #     # if 'freqs' in self._params:
    #     #     freqs = self._params['freqs']
    #     # else:
    #     freqs = self._params['freqs_nyq']*fsamp

    #     out = _np.zeros(shape=(len(scales), N,
    #                            signal.sizes['channel'], 
    #                            signal.sizes['component']))

    #     out = _xr.DataArray(out, 
    #                         dims=('freq', 'time', 'channel', 'component'),
    #                         coords = {'freq': freqs,
    #                                   'time': signal.coords['time'].values,
    #                                   'channel': signal.coords['channel'],
    #                                   'component': signal.coords['component']})

    #     return {'channel': 1, 'component':1}, out
        

class Maxima(_Filter): 
    """
    Find all local maxima in the signal

    Parameters
    ----------
    win_len : float, >0
        Length of window in seconds (method = 'windowing')
    win_step : float, >0
        Shift of the window to start the next window in seconds (method = 'windowing')
    method : str
        Method to detect the maxima. Available methods: 'complete' or 'windowing'. 'complete' finds all the local
         maxima, 'windowing' uses a runnning window to find the global maxima in each window.
    
    Optional parameters
    -------------------
    refractory : float, >0, default=0
        Seconds to skip after a detected maximum to look for new maxima, when method = 'complete'. 

    Returns
    -------
    idx_maxs : array
        Array containing indexes of the maxima
    val_maxs : array
        Array containing values of the maxima
    """

    def __init__(self, method='complete', refractory=0, win_len=None, win_step=None):
        assert method in ['complete', 'windowing'], "Method not valid"
        assert refractory >= 0, "Refractory time value should be positive (or 0 to deactivate)"
        
        if method == 'windowing':
            assert win_len > 0, "Window length should be positive"
            assert win_step > 0, "Window step should be positive"
        _Filter.__init__(self, method=method, refractory=refractory, win_len=win_len, win_step=win_step)
        self.required_dims = ['time']
        
    def algorithm(self, signal):
        params = self._params
        method = params['method']
        signal_values = signal.values.ravel()
        
        if method == 'complete':
            refractory = params['refractory']
            if refractory != 0:
                refractory = int(refractory * signal.p.get_sampling_freq())
            
            idx_maxs = []
            prev = signal_values[0]
            k = 1
            while k < len(signal_values) - 1 - refractory:
                curr = signal_values[k]
                nxt = signal_values[k + 1]
                if (curr >= prev) and (curr > nxt):
                    idx_maxs.append(k)
                    #update next k
                    if refractory > 0:
                        k = k + refractory
                    else:
                        k = k + 2
                    prev = signal_values[k - 1]
                else:  # continue
                    prev = signal_values[k]
                    k += 1
            idx_maxs = _np.array(idx_maxs).astype(int)
        elif method == 'windowing':
            fsamp = signal.p.get_sampling_freq()

            winlen = int(params['win_len'] * fsamp)
            winstep = int(params['win_step'] * fsamp)

            assert winlen >= 3, "Window length is less that 3 samples"
            assert winstep >= 1, "Window step is less than 1 sample"

            idx_maxs = [_np.nan]
            if winlen < len(signal):
                idx_start = _np.arange(0, len(signal) - winlen + 1, winstep)
            else:
                idx_start = [0]

            for idx_st in idx_start:
                idx_sp = idx_st + winlen
                if idx_sp > len(signal_values):
                    idx_sp = len(signal_values)
                curr_win = signal_values[idx_st: idx_sp]
                curr_idx_max = _np.argmax(curr_win) + idx_st
                
                # peak not already detected & peak not at the beginnig/end of the window:
                if curr_idx_max != idx_maxs[-1] and curr_idx_max != idx_st and curr_idx_max != idx_sp - 1:
                    idx_maxs.append(curr_idx_max)
            idx_maxs = idx_maxs[1:]
            
        out = _np.ones_like(signal.values)*_np.nan
        out[idx_maxs] = signal.values[idx_maxs]
        return out
        
class Minima(Maxima): 
    """
    Find all local minima in the signal

    Parameters
    ----------
    method : str
        Method to detect the minima. Available methods: 'complete' or 'windowing'. 'complete' finds all the local
        minima, 'windowing' uses a runnning window to find the global minima in each window.
    win_len : float, >0
        Length of window in seconds (method = 'windowing')
    win_step : float, >0
        Shift of the window to start the next window in seconds (method = 'windowing')

    Optional parameters
    -------------------
    refractory : float, >0, default = 0
        Seconds to skip after a detected minimum to look for new minima, when method = 'complete'. 

    Returns
    -------
    idx_mins : array
        Array containing indexes of the minima
    val_mins : array
        Array containing values of the minima
    """
    
    def algorithm(self, signal):
        result = super().algorithm(-signal)
        return(-result)

class PCA(_Algorithm):
    """
    """

    def __init__(self, n_out=1, dimension='channel'):
        _Algorithm.__init__(self, n_out=n_out, dimension=dimension)
        self.required_dims = ['time', dimension]
    
    def __get_template__(self, signal):
        dimension = self._params['dimension']
        assert dimension in signal.dims
        n_out = self._params['n_out']
        
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal, {dimension: n_out})
        return(chunk_dict, template)
        
    def algorithm(self, signal):
        pca = _PCA(n_components=self._params['n_out'])
        orig_channels = _np.squeeze(signal.p.get_values())
        out_channels = pca.fit_transform(orig_channels)
        return(out_channels)

#TODO IF NEEDED
# class Durations(_Algorithm):
#     """
#     Compute durations of events starting from their start and stop indexes

#     Parameters:
#     -----------
#     starts : array
#         Start indexes along the data
#     stops : array
#         Stop indexes along the data

#     Return:
#     -------
#     durations : array
#         durations of the events
#     """

#     def __init__(self, starts, stops):
#         starts = _np.array(starts)
#         assert starts.ndim == 1
#         stops = _np.array(stops)
#         assert stops.ndim == 1
#         _Algorithm.__init__(self, starts=starts, stops=stops)

    
#     def algorithm(self, signal):
#         params = self._params
#         starts = params["starts"]
#         stops = params["stops"]

#         fsamp = signal.get_sampling_freq()
#         durations = []
#         for I in range(len(starts)):
#             if (stops[I] > 0) & (starts[I] >= 0):
#                 durations.append((stops[I] - starts[I]) / fsamp)
#             else:
#                 durations.append(_np.nan)
#         return durations

# class Slopes(_Algorithm):
#     """
#     Compute rising slope of peaks

#     Parameters:
#     -----------
#     starts : array
#         Start of the peaks indexes
#     peaks : array
#         Peaks indexes

#     Return:
#     -------
#     slopes : array
#         Rising slopes the peaks
#     """

#     def __init__(self, starts, peaks):
#         starts = _np.array(starts)
#         assert starts.ndim == 1
#         peaks = _np.array(peaks)
#         assert peaks.ndim == 1
#         _Algorithm.__init__(self, starts=starts, peaks=peaks)

    
#     def algorithm(cls, data, params):
#         starts = params["starts"]
#         peaks = params["peaks"]

#         fsamp = data.get_sampling_freq()
#         slopes = []
#         for I in range(len(starts)):
#             if peaks[I] > 0 & starts[I] >= 0:
#                 dy = data[peaks[I]] - data[starts[I]]
#                 dt = (peaks[I] - starts[I]) / fsamp
#                 slopes.append(dy / dt)
#             else:
#                 slopes.append(_np.nan)
#         return slopes

class PeakSelection(_Filter):
    """
    Identify the start and the end indexes of each peak in the signal, using derivatives.

    Parameters
    ----------
    win_pre : float, >0
        Duration (in seconds) of interval before the peak that is considered to find the start of the peak
    win_post : float, >0
        Duration (in seconds) of interval after the peak that is considered to find the end of the peak
    
    Returns
    -------
    starts : array
        Array containing start indexes
    ends : array
        Array containing end indexes
    """

    def __init__(self, win_pre, win_post):
        assert win_pre > 0, "Window pre peak value should be positive"
        assert win_post > 0, "Window post peak value should be positive"
        _Filter.__init__(self, win_pre=win_pre, win_post=win_post)
        self.required_dims = ['time']
    
    def __get_template__(self, signal):
        chunk_dict, template = super().__get_template__(signal)
        template['peaks'] = signal['peaks']

        return (chunk_dict, template)
    
    def __mapper_func__(self, signal_in, **kwargs):
        out = super().__mapper_func__(signal_in, **kwargs)
        out['peaks'] = signal_in['peaks']
        return(out)
    
    def algorithm(self, signal):
        assert 'peaks' in signal.coords, "Signal coords should contain peaks"
        
        params = self._params
        peaks = signal['peaks']
        i_peaks = _np.where(_np.isnan(peaks)==False)[0]
        i_pre_max = int(params['win_pre'] * signal.p.get_sampling_freq())
        i_post_max = int(params['win_post'] * signal.p.get_sampling_freq())
        
        signal_values = signal.p.get_values().ravel()
        dd = _np.convolve(_np.diff(signal_values)//1.0, _np.ones(2)/2)

        i_start = []
        i_stop = []
        
        for idx_max in i_peaks:
            idx_pre = idx_max-1
            while ((dd[idx_pre]>-0.5) and ((idx_max-idx_pre) <= i_pre_max)) and (idx_pre>0):
                idx_pre -=1
            idx_pre = idx_pre + 1
            i_start.append(idx_pre)
            
            idx_post = idx_max+1
            while ((dd[idx_post]<-0.5) and ((idx_post-idx_max) <= i_post_max)) and (idx_post<(len(signal_values)-1)):
                idx_post +=1

            i_stop.append(idx_post)
        
        sig_out = _np.zeros(len(signal_values))
        
        for i_st, i_sp in zip(i_start, i_stop):
            sig_out[i_st:i_sp] = 1
        
        return sig_out
