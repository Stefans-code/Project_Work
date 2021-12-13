# coding=utf-8
# from __future__ import division
import numpy as _np
import xarray as _xr


from scipy.signal import welch as _welch, periodogram as _periodogram, freqz as _freqz
import pycwt.wavelet as wave
from scipy import linalg as _linalg

from . import Algorithm as _Algorithm
from ..signal import create_signal

_xr.set_options(keep_attrs = True)


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

class Diff(_Algorithm): #xarray done
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
        _Algorithm.__init__(self, degree=degree)
        self.dimensions = {'time' : 0}

    
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
    
class PeakDetection(_Algorithm): #xarray done
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

    def __init__(self, delta, refractory=0, start_max=True, return_peaks = True):
        delta = _np.array(delta)
        assert delta.ndim <= 1, "Delta value should be 1 or 0-dimensional"
        assert delta.all() > 0, "Delta value/s should be positive"
        assert refractory >= 0, "Refractory value should be non negative"
        _Algorithm.__init__(self, delta=delta, refractory=refractory, start_max=start_max, return_peaks=return_peaks)
        self.dimensions = {'time' : 0}

    def __finalize__(self, res_sig, arr_windows):
        return __finalize_special__(res_sig)
        
    
    def algorithm(self, signal):
        params = self._params
        refractory = params['refractory']
        if refractory == 0:  # if 0 then do not skip samples
            refractory = 1
        else:  # else transform the refractory from seconds to samples
            refractory = refractory * signal.p.get_sampling_freq()
        look_for_max = params['start_max']
        delta = params['delta']

        return_peaks = params['return_peaks']
        minp = []
        maxp = []

        minv = []
        maxv = []

        scalar = delta.ndim == 0
        if scalar:
            d = delta

        s = signal.values.ravel()
        if not scalar and len(delta) != len(signal):
            print("delta vector's length differs from signal's one, returning empty.")
        else:
            mn_pos_candidate = mx_pos_candidate = 0
            mn_candidate = mx_candidate = s[0]

            i_activation_min = 0
            i_activation_max = 0

            for i in range(1, len(s)):
                sample = s[i]
                if not scalar:
                    d = delta[i]

                if sample > mx_candidate:
                    mx_candidate = sample
                    mx_pos_candidate = i
                if sample < mn_candidate:
                    mn_candidate = sample
                    mn_pos_candidate = i

                if look_for_max:
                    if i >= i_activation_max and sample < mx_candidate - d:  # new max
                        maxp.append(mx_pos_candidate)
                        maxv.append(mx_candidate)
                        i_activation_max = i + refractory

                        mn_candidate = sample
                        mn_pos_candidate = i

                        look_for_max = False
                else:
                    if i >= i_activation_min and sample > mn_candidate + d:  # new min
                        minp.append(mn_pos_candidate)
                        minv.append(mn_candidate)
                        i_activation_min = i + refractory

                        mx_candidate = sample
                        mx_pos_candidate = i

                        look_for_max = True
        
        out = _np.ones_like(signal.values)*_np.nan
        
        if return_peaks:
            out[maxp] = signal.values[maxp]
        else:
            out[minp] = signal.values[minp]
        
        out_xarray = signal.copy(data = out)

        return out_xarray

class SignalRange(_Algorithm): #xarray done
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
        _Algorithm.__init__(self, win_len=win_len, win_step=win_step, smooth=smooth)
        self.dimensions = {'time' : 0}

    
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
        if len(signal) < idx_len:
            print("Input signal is shorter than the window length.")
            return _np.max(signal) - _np.min(signal)
        else:
            windows = _np.arange(0, len(signal_values) - idx_len + 1, idx_step)
            deltas = _np.zeros(len(signal_values))

            curr_delta = 0
            for start in windows:
                portion_curr = signal_values[start: start + idx_len]
                curr_delta = _np.max(portion_curr) - _np.min(portion_curr)
                deltas[start:start + idx_len] = curr_delta

            deltas[windows[-1] + idx_len:] = curr_delta

            if smooth:
                win_len = int(win_len*2*fsamp)
                deltas = _np.convolve(deltas, _np.ones(win_len)/win_len, mode='same')
            # print('<<< signalrange')
            return deltas

class PSD(_Algorithm): #xarray done
    """
    Estimate the power spectral density (PSD) of the signal.

    Parameters
    ----------
    method : str
        Method to estimate the PSD. Available methods: 'welch', 'fft', 'ar'
        
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

    def __init__(self, method, nfft=2048, window='hamming', min_order=10, max_order=30, normalize=False,
                 remove_mean=True, **kwargs):
        
        _method_list = ['welch', 'fft', 'ar']
        _window_list = ['hamming', 'blackman', 'hanning', 'bartlett', 'none']

        assert method in _method_list, "Parameter method should be in " + _method_list.__repr__()
        assert nfft > 0, "nfft value should be positive"
        assert window in _window_list, "Parameter window type should be in " + _window_list.__repr__()
        if method == "ar":
            assert min_order > 0, "Minimum order for the AR method should be positive"
            assert max_order > 0, "Maximum order for the AR method should be positive"
        
        _Algorithm.__init__(self, method=method, nfft=nfft, window=window, min_order=min_order,
                       max_order=max_order, normalize=normalize, remove_mean=remove_mean, **kwargs)
        
        self.dimensions = 'special'
        
    # TODO (Feature - Issue #15): consider point below:
    # A density spectrum considers the amplitudes per unit frequency.
    # Density spectra are used to compare spectra with different frequency resolution as the
    # magnitudes are not influenced by the resolution because it is per Hertz. The amplitude
    # spectra on the other hand depend on the chosen frequency resolution.

    def __finalize__(self, res_sig, arr_window):
        return __finalize_special__(res_sig)
    
    def algorithm(self, signal):
        # print('----->', self.name)
        params = self._params
        method = params['method']
        nfft = params['nfft'] if "nfft" in params else None
        window = params['window']
        normalize = params['normalize']
        remove_mean = params['remove_mean']

        fsamp = signal.p.get_sampling_freq()
        
        signal_values = signal.values.ravel()
        
        if remove_mean:
            signal_values = signal_values - _np.mean(signal_values)

        if method == 'fft':
            freqs, psd = _periodogram(signal_values, fs=fsamp, window = window, nfft=nfft, return_onesided=True)

        elif method == 'welch':
            freqs, psd = _welch(signal_values, fsamp, window=window, return_onesided=True, nfft=nfft)

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
            '''

        freqs = _np.linspace(start=0, stop=fsamp / 2, num=len(psd))

        # NORMALIZE
        if normalize:
            psd /= _np.sum(psd)
      
        # out = signal.copy(deep=True)
        # out = out.expand_dims({'freq':freqs}, axis=0)[:,0]
        # out.name = signal.name+'_'+self.name
        # out.values = _np.expand_dims(psd,[1,2])
        
        psd = _np.expand_dims(psd,[1, 2])
        
        out = signal.copy(deep=True)
        out = out.expand_dims({'freq':freqs}, axis=0)[:,0]
        out = out.drop('time')
        # out.name = signal.name+'_'+self.name
        # print(out)
        
        out.values = psd
        # print('<-----', self.name)
        return out

    def __get_template__(self, signal):
        nfft = self._params['nfft']
        N = int(nfft/2 + 1)
        out = _np.zeros(shape=(N, #1,
                               signal.sizes['channel'], 
                               signal.sizes['component']))
        
        fsamp = signal.p.get_sampling_freq()
        
        freqs = _np.linspace(start=0, stop=fsamp / 2, num=N)
        
        out = _xr.DataArray(out, dims=('freq', 'channel', 'component'), #'time', 'channel', 'component'),
                            coords = {'freq': freqs,
                                      # 'time': [signal.coords['time'].values[0]],
                                      'channel': signal.coords['channel'],
                                      'component': signal.coords['component']})
        # print(out)
        return {'channel': 1, 'component':1}, out
    
class Wavelet(_Algorithm): #xarray dones
    """
    TODO
    """
    def __init__(self, detrend=True, mother = None, J = None, **kwargs):
        mother = wave.Morlet(6) if mother is None else mother
        
        if J is None:
            J = 68 #default value in cwt function
        else:
            assert J>1
            
        
        _Algorithm.__init__(self, detrend = detrend, mother = mother, J=J, **kwargs)
        self.dimensions = 'special'
        
    def __finalize__(self, res_sig, arr_window):
        return __finalize_special__(res_sig)
    
    def algorithm(self, signal):
        params = self._params
        
        fsamp = signal.p.get_sampling_freq()
        t = signal.p.get_times()
        t0 = signal.p.get_start_time()
        dt = 1/fsamp
        
        detrend = params['detrend']
        signal_values = signal.values.ravel()
        
        if detrend:
            #% detrend
            p = _np.polyfit(t - t0, signal_values, 1)
            signal_values = signal_values - _np.polyval(p, t - t0)
            
        #% wavelet
        J = params['J']
        freqs = _np.logspace(1, _np.log10(fsamp/2), J+1)[::-1]
        
        mother = params['mother']
        w, scales, freqs, coi, fft, fftfreqs = wave.cwt(signal_values, dt, 
                                                        wavelet=mother,
                                                        J=J,
                                                        freqs=freqs)
        
        power = (_np.abs(w)) ** 2
        power /= scales[:, None]
        power = _np.expand_dims(power,[2,3])

        out = signal.copy(deep=True)
        out = out.expand_dims({'freq':freqs}, axis=0)
        # out.name = signal.name+'_'#+self.name
        
        out.values = power
        return out
    
    def __get_template__(self, signal):
        J = self._params['J']
        N = signal.shape[0]
        out = _np.zeros(shape=(J+1, N,
                               signal.sizes['channel'], 
                               signal.sizes['component']))
        
        fsamp = signal.p.get_sampling_freq()
        freqs = _np.logspace(1, _np.log10(fsamp/2), J+1)[::-1]
        
        out = _xr.DataArray(out, dims=('freq', 'time', 'channel', 'component'),
                            coords = {'freq': freqs,
                                      'time': signal.coords['time'].values,
                                      'channel': signal.coords['channel'],
                                      'component': signal.coords['component']})
        # print(out)
        return {'channel': 1, 'component':1}, out
        

class Maxima(_Algorithm): #xarray done
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
        _Algorithm.__init__(self, method=method, refractory=refractory, win_len=win_len, win_step=win_step)
        self.dimensions = {'time' : 0}
    
    def __finalize__(self, res_sig, arr_window):
        return __finalize_special__(res_sig)
    
    def algorithm(self, signal):
        params = self._params
        method = params['method']
        signal_values = signal.values.ravel()
        
        if method == 'complete':
            refractory = params['refractory']
            if refractory == 0:
                refractory = 1
            else:
                refractory = refractory * signal.p.get_sampling_freq()
            idx_maxs = []
            prev = signal_values[0]
            k = 1
            while k < len(signal_values) - 1 - refractory:
                curr = signal_values[k]
                nxt = signal_values[k + 1]
                if (curr >= prev) and (curr >= nxt):
                    idx_maxs.append(k)
                    prev = signal_values[k + 1 + refractory]
                    k = k + 2 + refractory
                else:  # continue
                    prev = signal_values[k]
                    k += 1
            idx_maxs = _np.array(idx_maxs).astype(int)
        elif method == 'windowing':
            fsamp = signal.p.get_sampling_freq()
            
            winlen = int(params['win_len'] * fsamp)
            winstep = int(params['win_step'] * fsamp)

            # TODO (Andrea): check that winlen > 2
            # TODO (Andrea): check that winstep >= 1

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
                curr_max = _np.max(curr_win)

                # peak not already detected & peak not at the beginnig/end of the window:
                if curr_idx_max != idx_maxs[-1] and curr_idx_max != idx_st and curr_idx_max != idx_sp - 1:
                    idx_maxs.append(curr_idx_max)
            idx_maxs = idx_maxs[1:]
            
        out = _np.ones_like(signal.values)*_np.nan
        out[idx_maxs] = signal.values[idx_maxs]
        out_xarray = signal.copy(data = out)
        return out_xarray
        
class Minima(_Algorithm): #xarray done
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

    def __init__(self, method='complete', refractory=0, win_len=None, win_step=None):
        assert method in ['complete', 'windowing'], "Method not valid"
        assert refractory >= 0, "Refractory time value should be positive (or 0 to deactivate)"
        
        if method == 'windowing':
            assert win_len > 0, "Window length should be positive"
            assert win_step > 0, "Window step should be positive"
        _Algorithm.__init__(self, method=method, refractory=refractory, win_len=win_len, win_step=win_step)
        
        
    def __finalize__(self, res_sig, arr_window):
        return __finalize_special__(res_sig)
    
    def algorithm(self, signal):
        params = self._params
        max_alg = Maxima(**params) 
        #TODO: check
        result = max_alg.algorithm(-signal)
        return(result)

#TODO from here
class BootstrapEstimation(_Algorithm):
    """
    Perform a bootstrapped estimation of given statistical indicator
    
    Parameters
    ----------
    func : numpy function
        Function to use in the bootstrapping. Must accept data as input
        
    Optional parameters
    -------------------
    
    n : int, >0, default = 100
        Number of iterations
    k : float, (0,1), default = 0.5
        Portion of data to be used at each iteration
    
    Returns
    -------
    estim : float
        Bootstrapped estimate
    
    """

    def __init__(self, func, n=100, k=0.5):
        from types import FunctionType as Func
        assert isinstance(func, Func), "Parameter function should be a function (types.FunctionType)"
        assert n > 0, "n should be positive"
        assert 0 < k <= 1, "k should be between (0 and 1]"
        _Algorithm.__init__(self, func=func, n=n, k=k)

    
    def algorithm(self, signal):
        params = self._params
        signal = _np.asarray(signal)
        l = len(signal)
        func = params['func']
        niter = int(params['n'])
        k = params['k']

        estim = []
        for i in range(niter):
            ixs = _np.arange(l)
            ixs_p = _np.random.permutation(ixs)
            sampled_data = signal[ixs_p[:int(round(k * l))]]
            curr_est = func(sampled_data)
            estim.append(curr_est)
        estim = _np.sort(estim)
        return estim[int(len(estim) / 2)]

class Durations(_Algorithm):
    """
    Compute durations of events starting from their start and stop indexes

    Parameters:
    -----------
    starts : array
        Start indexes along the data
    stops : array
        Stop indexes along the data

    Return:
    -------
    durations : array
        durations of the events
    """

    def __init__(self, starts, stops):
        starts = _np.array(starts)
        assert starts.ndim == 1
        stops = _np.array(stops)
        assert stops.ndim == 1
        _Algorithm.__init__(self, starts=starts, stops=stops)

    
    def algorithm(self, signal):
        params = self._params
        starts = params["starts"]
        stops = params["stops"]

        fsamp = signal.get_sampling_freq()
        durations = []
        for I in range(len(starts)):
            if (stops[I] > 0) & (starts[I] >= 0):
                durations.append((stops[I] - starts[I]) / fsamp)
            else:
                durations.append(_np.nan)
        return durations

class Slopes(_Algorithm):
    """
    Compute rising slope of peaks

    Parameters:
    -----------
    starts : array
        Start of the peaks indexes
    peaks : array
        Peaks indexes

    Return:
    -------
    slopes : array
        Rising slopes the peaks
    """

    def __init__(self, starts, peaks):
        starts = _np.array(starts)
        assert starts.ndim == 1
        peaks = _np.array(peaks)
        assert peaks.ndim == 1
        _Algorithm.__init__(self, starts=starts, peaks=peaks)

    
    def algorithm(cls, data, params):
        starts = params["starts"]
        peaks = params["peaks"]

        fsamp = data.get_sampling_freq()
        slopes = []
        for I in range(len(starts)):
            if peaks[I] > 0 & starts[I] >= 0:
                dy = data[peaks[I]] - data[starts[I]]
                dt = (peaks[I] - starts[I]) / fsamp
                slopes.append(dy / dt)
            else:
                slopes.append(_np.nan)
        return slopes

class BeatOutliers(_Algorithm):
    """
    Detects outliers in the IBI signal. 
    
    Optional parameters
    -------------------
    
    cache : int, >0,  default=3
        Number of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI
    
    Returns
    -------
    id_bad_ibi : numpy.array
        Identifiers of wrong beats
    
    Notes
    -----
    It only detects outliers. You should manually remove outliers using FixIBI
    
    """

    def __init__(self, ibi_median=0, cache=3, sensitivity=0.25):
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Algorithm.__init__(self, ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)

   
    def algorithm(self, signal):
        params = self._params
        cache, sensitivity, ibi_median = params["cache"], params["sensitivity"], params["ibi_median"]

        if ibi_median == 0:
            ibi_expected = float(_ma.median(signal))
        else:
            ibi_expected = float(ibi_median)

        id_bad_ibi = []
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        # missings = []
        idx_ibi = signal.get_indices()
        ibi = signal.get_values()
        for i in range(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)

            curr_ibi = ibi[i]

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            # missings.append([idx_ibi[i-1],idx_ibi[i]])

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                counter_bad = 0
            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                counter_bad = 0

        return id_bad_ibi

class FixIBI(_Algorithm):
    """
    Corrects the IBI series removing abnormal IBI
    
    Parameters
    ----------
    idx_bad_ibi : array
        Identifiers of abnormal beats
   
    Returns
    -------
    ibi : Unevenly Signal
        Corrected IBI
            
    """

    def __init__(self, idx_bad_ibi):
        idx_bad_ibi = _np.array(idx_bad_ibi)
        assert idx_bad_ibi.ndim == 1
        _Algorithm.__init__(self, id_bad_ibi=idx_bad_ibi)

    
    def algorithm(self, signal):
        params = self._params
        
        id_bad = params['id_bad_ibi']
        if len(id_bad) == 0:
            return(signal)
        
        idx_ibi = signal.get_indices()
        ibi = signal.get_values()
        idx_ibi_nobad = _np.delete(idx_ibi, id_bad)
        ibi_nobad = _np.delete(ibi, id_bad)
        idx_ibi = idx_ibi_nobad.astype(int)
        ibi = ibi_nobad
        return _Signal(values = ibi, 
                       sampling_freq = signal.get_sampling_freq(), 
                       start_time = signal.get_start_time(),
                       info = signal.get_info(), 
                       x_values=idx_ibi, x_type='indices')

class PeakSelection(_Algorithm):
    """
    Identify the start and the end indexes of each peak in the signal, using derivatives.

    Parameters
    ----------
    indices : array, >=0
        Array containing indexes (first column) and values (second column) of the maxima
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

    def __init__(self, indices, win_pre, win_post):
        indices = _np.array(indices)
        assert indices.ndim < 2, "Parameter indices has to be 1 or 0-dimensional"
        assert indices.all() >= 0, "Parameter indices contains negative values"
        assert win_pre > 0, "Window pre peak value should be positive"
        assert win_post > 0, "Window post peak value should be positive"
        _Algorithm.__init__(self, indices=indices, win_pre=win_pre, win_post=win_post)

    
    def algorithm(self, signal):
        params = self._params
        i_peaks = params['indices']
        i_pre_max = int(params['win_pre'] * signal.get_sampling_freq())
        i_post_max = int(params['win_post'] * signal.get_sampling_freq())

        ZERO = 0.01

        i_start = _np.empty(len(i_peaks), int)
        i_stop = _np.empty(len(i_peaks), int)

        signal_dt = Diff()(signal)
        for i in range(len(i_peaks)):
            i_pk = int(i_peaks[i])

            if i_pk < i_pre_max:
                i_st = 0
                i_sp = i_pk + i_post_max
            elif i_pk >= len(signal_dt) - i_post_max:
                i_st = i_pk - i_pre_max
                i_sp = len(signal_dt) - 1
            else:
                i_st = i_pk - i_pre_max
                i_sp = i_pk + i_post_max

            # find START
            signal_dt_pre = signal_dt[i_st:i_pk]
            i_pre = len(signal_dt_pre) - 1

            # OR below is to allow small fluctuations (?)

            while i_pre > 0 and (signal_dt_pre[i_pre] > 0 or abs(signal_dt_pre[i_pre]) <= ZERO):
                i_pre -= 1

            i_start[i] = i_st + i_pre + 1

            # find STOP
            signal_dt_post = signal_dt[i_pk: i_sp]
            i_post = 1

            # OR below is to allow small fluctuations (?)
            while i_post < len(signal_dt_post) - 1 and (
                            signal_dt_post[i_post] < 0 or abs(signal_dt_post[i_post]) <= ZERO):
                i_post += 1

            i_stop[i] = i_pk + i_post

        return i_start, i_stop