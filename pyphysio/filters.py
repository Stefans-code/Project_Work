import numpy as _np
import scipy.stats as _stats
try:
    from scipy.signal import gaussian as _gaussian
except:
    from scipy.signal.windows import gaussian as _gaussian

from scipy.signal import filtfilt as _filtfilt, \
    filter_design as _filter_design, iirfilter as _iirfilter, \
        deconvolve as _deconvolve, firwin as _firwin, \
            iirnotch as _iirnotch, lfilter as _lfilter
from ._base_algorithm import _Algorithm
from .utils import SignalRange as _SignalRange

class Normalize(_Algorithm):
    """
    A class for normalizing signals using various methods.

    This class provides functionality to normalize signals using different normalization methods, including mean subtraction, standardization, min-max scaling, and custom scaling.

    Parameters:
        norm_method (str, optional): The normalization method to be used. Defaults to 'standard'.
            - 'mean': Subtract the mean from each value.
            - 'standard': Standardize the signal by subtracting the mean and dividing by the standard deviation.
            - 'min': Subtract the minimum value from each value.
            - 'maxmin': Scale the values to the range [0, 1] by subtracting the minimum value and dividing by the difference between the maximum and minimum values.
            - 'custom': Scale the values by subtracting a custom bias and dividing by a custom range.
        norm_bias (float, optional): The bias value used for custom scaling. Defaults to 0.
        norm_range (float, optional): The range value used for custom scaling. Must not be zero. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the base class constructor.

    Methods:
        algorithm(signal, **kwargs): Normalize the given signal using the specified normalization method.

    Raises:
        AssertionError: If norm_method is not one of 'mean', 'standard', 'min', 'maxmin', 'custom'.
        AssertionError: If norm_range is zero when norm_method is 'custom'.
        ValueError: If an unsupported norm_method is specified.
    """

    def __init__(self, norm_method='standard', norm_bias=0, norm_range=1, **kwargs):
        assert norm_method in ['mean', 'standard', 'min', 'maxmin', 'custom'],\
            "norm_method must be one of 'mean', 'standard', 'min', 'maxmin', 'custom'"
        if norm_method == "custom":
            assert norm_range != 0, "norm_range must not be zero"
        _Algorithm.__init__(self, norm_method=norm_method, norm_bias=norm_bias, norm_range=norm_range, **kwargs)
        self.dimensions = {'time' : 0}

    def algorithm(self, signal, **kwargs):
        from .indicators.timedomain import Mean as _Mean, StDev as _StDev, Min as _Min, Max as _Max
        params = self._params
        method = params['norm_method']
        signal_values = signal.values
        if method == "mean":
            return signal_values - _Mean()(signal).values
        elif method == "standard":
            mean = _Mean()(signal).values
            std = _StDev()(signal).values
            result = (signal_values - mean) / std
            return(result)
            
        elif method == "min":
            return signal_values - _Min()(signal).values
        elif method == "maxmin":
            return (signal_values - _Min()(signal).values) / \
                (_Max()(signal).values - _Min()(signal).values)
        elif method == "custom":
            result = (signal_values - params['norm_bias']) / params['norm_range']
            return result
        else:
            raise ValueError

class IIRFilter(_Algorithm):
    """
    Infinite Impulse Response (IIR) Filter implementation.

    This class implements an IIR filter algorithm for signal filtering. It supports different filter types
    such as Butterworth, Chebyshev Type I, Chebyshev Type II, elliptic, and Bessel filters. The filter can
    operate in various modes including lowpass, highpass, bandpass, and bandstop.

    Parameters:
        fp (float or array_like): Passband edge frequencies. For a lowpass or highpass filter, it should be a
            single value. For a bandpass or bandstop filter, it should be a tuple of two values representing
            the lower and upper passband edge frequencies.
        fs (float or array_like, optional): Stopband edge frequencies. Required for bandpass and bandstop
            filters. For a lowpass or highpass filter, it is ignored. If not provided, the stopband edge
            frequencies are set to None.
        btype (str, optional): Filter type. Can be one of the following: 'lowpass', 'highpass', 'bandpass',
            or 'bandstop'. Defaults to 'bandpass'.
        order (int, optional): Filter order. Defaults to 3.
        loss (float, optional): Passband loss (ripple) in decibels (dB). Must be a positive value.
            Defaults to 0.1.
        att (float, optional): Stopband attenuation in decibels (dB). Must be a positive value and greater than
            the loss value. Defaults to 40.
        ftype (str, optional): Filter type. Can be one of the following: 'butter', 'cheby1', 'cheby2', 'ellip',
            or 'bessel'. Defaults to 'cheby1'.
        safe (bool, optional): Whether to enable safe mode. When safe mode is enabled, if the filter parameters
            result in no solution, the original signal is returned. Defaults to True.

    Attributes:
        dimensions (dict): Dictionary specifying the dimensions of the filter. In this case, only the 'time'
            dimension is used, which is set to 0.

    Methods:
        algorithm(signal):
            Apply the IIR filter algorithm to the input signal and return the filtered signal.

    Notes:
        - The IIRFilter class inherits from the _Algorithm base class.
        - The _Algorithm base class provides common functionality and is not defined in this documentation.
        - The IIRFilter class uses functions from numpy, scipy, and other modules.

    """

    def __init__(self, fp, fs=None, btype='bandpass', order=3, loss=.1, att=40, ftype='cheby1', safe=True):
        assert loss > 0, "Loss value should be positive"
        assert att > 0, "Attenuation value should be positive"
        assert att > loss, "Attenuation value should be greater than loss value"
        assert ftype in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'],\
            "Filter type must be in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']"
        _Algorithm.__init__(self, fp=fp, fs=fs, btype=btype, order=order, 
                            loss=loss, att=att, ftype=ftype, safe=safe)
        self.dimensions = {'time' : 0}

    def algorithm(self, signal):
        # print('----->', self.name)
        # print(signal.shape)
        params = self._params
        fsamp = signal.p.get_sampling_freq()
        fp, fs, btype, order = params["fp"], params["fs"], params["btype"], params["order"]
        loss, att, ftype = params["loss"], params["att"], params["ftype"]
        safe = params["safe"]
        
        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        wp = fp / nyq
        assert (wp<1).all(), f"invalid fp for given sampling frequency {fsamp}"
        
        if fs is None:
            b, a = _iirfilter(order, wp, btype=btype, rp=loss, rs=att, analog=False, ftype=ftype)
        else:
            fs = _np.array(fs)
            ws = fs / nyq
            assert (ws<1).all(), f"invalid fs for given sampling frequency {fsamp}"
        
            b, a = _filter_design.iirdesign(wp, ws, loss, att, ftype=ftype, output="ba")
        

        sig_filtered = _filtfilt(b, a, signal.values.ravel(), axis=0)

        if safe:
            if _np.isnan(sig_filtered[0]):
                print('Filter parameters allow no solution. Returning original signal.')
                return signal.values

        # print('<-----', self.name)
        return sig_filtered

class NotchFilter(_Algorithm):
    """
    NotchFilter is a class that implements a notch filter algorithm to remove a specific frequency component from a signal.

    Parameters:
    -----------
    f : float
        The frequency of the notch filter in Hz. Must be greater than 0.
    Q : float, optional
        The quality factor of the notch filter. Higher values result in a narrower bandwidth. Must be greater than 0. Default is 30.
    safe : bool, optional
        A flag indicating whether to handle unsafe filter parameters. If set to True, and the filter parameters result in no valid solution, the original signal will be returned instead. Default is True.
    """

    def __init__(self, f, Q=30, safe=True):
        assert f > 0
        assert Q > 0
        _Algorithm.__init__(self, f=f, Q=Q, safe=safe)
        self.dimensions = {'time' : 0}

    def algorithm(self, signal):
        params = self._params
        fsamp = signal.p.get_sampling_freq()
        f = params["f"]
        Q = params["Q"]
        safe = params["safe"]
        
        b, a = _iirnotch(f, Q, fsamp)
        
        sig_filtered = _filtfilt(b, a, signal.values.ravel(), axis=0)

        if safe:
            if _np.isnan(sig_filtered[0]):
                print('Filter parameters allow no solution. Returning original signal.')
                return signal.values
        
        return sig_filtered
        
class FIRFilter(_Algorithm):
    """
    Finite Impulse Response (FIR) filter class for signal processing.

    Parameters:
    -----------
    fp : float or array_like
        Cutoff frequency or frequencies for the filter. For a lowpass or highpass
        filter, a single value should be provided. For a bandpass or bandstop filter,
        a list or array of two values should be provided.
    fs : float or array_like, optional
        Stop frequency or frequencies for the filter. If not specified, a lowpass or
        highpass filter is created. For a bandpass or bandstop filter, a list or array
        of two values should be provided.
    order : int, optional
        Order of the filter. Default is 5.
    btype : str, optional
        Type of filter. Possible values are 'lowpass', 'highpass', 'bandpass', and
        'bandstop'. Default is 'lowpass'.
    att : float, optional
        Attenuation value in decibels (dB). Default is 40.
    wtype : str, optional
        Type of window to use in filter design. Currently, only 'hamming' window is
        supported. Default is 'hamming'.
    safe : bool, optional
        If True, check if the filter parameters allow a valid solution. If not, return
        the original signal. If False, no check is performed and the filter is applied
        regardless of the parameters. Default is True.
    
    Note:
    -----
    This class inherits from the _Algorithm class.

    """

    def __init__(self, fp, fs=None, order=5, btype='lowpass', att=40, wtype='hamming', safe=True):
        assert att > 0, "Attenuation value should be positive"
        assert wtype in ['hamming'],\
            "Window type must be in ['hamming']"
        _Algorithm.__init__(self, fp=fp, fs=fs, order=order, btype=btype,
                            att=att, wtype=wtype, safe=True)
        self.dimensions = {'time' : 0}

    def algorithm(self, signal):
        params = self._params
        fsamp = signal.p.get_sampling_freq()
        fp, fs, order = params["fp"], params["fs"], params["order"]
        btype, att, wtype = params["btype"], params["att"], params["wtype"]
        safe = params["safe"]
        fp = _np.array(fp)
                
        if fp.ndim == 0:
            fp = _np.expand_dims(fp, 0)
    
        if fs is None:
            pass_zero = btype
            N = order+1
        
        else:
            fs = _np.array(fs)
            if fs.ndim == 0:
                fs = _np.expand_dims(fs, 0)
    
            # d1 = 10**(loss/10)
            # d2 = 10**(att/10)
            Dsamp = _np.min(abs(fs-fp))/fsamp
            
            # from https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need
            # N = int(2/3*_np.log10(1/(10*d1*d2))*fsamp/Dsamp)
            N = int(att/(22*Dsamp))
                           
            pass_zero=True
                      
            if fp[0]>fs[0]:
                pass_zero=False
            
        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        wp = fp / nyq
        
        if N%2 ==0:
            N+=1
            
        b = _firwin(N, wp, window=wtype, pass_zero=pass_zero)
        signal_values = signal.values.ravel()
        sig_filtered = _lfilter(b, 1.0, signal_values)
        sig_filtered[0:N] = sig_filtered[N]
        sig_out = _np.ones(len(signal_values)) * sig_filtered[-1]
        
        idx_ = N//2
        sig_out[:-idx_] = sig_filtered[idx_:]
        
        if safe:
            if _np.isnan(sig_filtered[0]):
                print('Filter parameters allow no solution. Returning original signal.')
                return signal_values
        
        return sig_out

class KalmanFilter(_Algorithm):
    """
    Implements a Kalman Filter algorithm for signal processing.

    This class applies the Kalman Filter algorithm to a given signal. The Kalman Filter is an optimal estimation algorithm that combines measurements and a prediction model to estimate the state of a system. It is commonly used in signal processing and control systems.

    Parameters
    ----------
    R : float
        The measurement noise covariance. Must be a positive value.
    ratio : float
        The ratio used to calculate the process noise covariance. Must be greater than 1.
    win_len : float, optional
        The length of the sliding window used to estimate the process noise covariance. Must be a positive value. Defaults to 1.
    win_step : float, optional
        The step size of the sliding window used to estimate the process noise covariance. Must be a positive value. Defaults to 0.5.

    """

    def __init__(self, R, ratio, win_len=1, win_step=0.5):
        assert R > 0, "R should be positive"
        assert ratio > 1, "ratio should be >1"
        assert win_len > 0, "Window length value should be positive"
        assert win_step > 0, "Window step value should be positive"
        
        _Algorithm.__init__(self, R=R, ratio=ratio, win_len=win_len, win_step=win_step)
        self.dimensions = {'time' : 0}
        
    def algorithm(self, signal):
        params = self._params
        R = params['R']
        ratio = params['ratio']
        win_len = params['win_len']
        win_step = params['win_step']
        
        sz = len(signal)
        
        rr = _SignalRange(win_len, win_step)(signal).values
        Q = _np.nanmedian(rr)/ratio
            
        P = 1
        
        x_out = signal.values.ravel()
        for k in range(1,sz):
                x_ = x_out[k-1]
                P_ = P + Q
            
                # measurement update
                K = P_ / (P_ + R)
                x_out[k] = x_ + K * (x_out[k] - x_)
                P = (1 - K ) * P_

        return(x_out)

class RemoveSpikes(_Algorithm):
    #TODO: see MA removal using spline in fnirs specialized
    def __init__(self, K=2, N=1, dilate=0, D=0.95, method='step'):
        assert K > 0, "K should be positive"
        assert isinstance(N, int) and N>0, "N value not valid"
        assert dilate>=0, "dilate should be >= 0.0"
        assert D>=0, "D should be >= 0.0"
        assert method in ['linear', 'step']
        _Algorithm.__init__(self, K=K, N=N, dilate=dilate, D=D, method=method)
        self.dimensions = {'time' : 0}
    
    def algorithm(self, signal):
        params = self._params
        K = params['K']
        N = params['N']
        dilate = params['dilate']
        D = params['D']
        method = params['method']
        fs = signal.p.get_sampling_freq()
        
        
        s = signal.values.ravel()
        sig_diff = abs(s[N:] - s[:-N])
        ds_mean = _np.nanmean(sig_diff)
        
        idx_spikes = _np.where(sig_diff>K*ds_mean)[0]+N//2
        spikes = _np.zeros(len(s))
        spikes[idx_spikes] = 1
        win = _np.ones(1+int(2*dilate*fs))
        spikes = _np.convolve(spikes, win, 'same')
        idx_spikes = _np.where(spikes>0)[0]
        
        x_out = signal.values.ravel()
        
        #TODO check linear connector method
        if method == 'linear':
            diff_idx_spikes = _np.diff(idx_spikes)
            new_spike = _np.where(diff_idx_spikes > 1)[0] + 1
            new_spike = _np.r_[0, new_spike, -1]
            for I in range(len(new_spike)-1):
                IDX_START = idx_spikes[new_spike[I]] -1
                IDX_STOP = idx_spikes[new_spike[I+1]-1] +1
                
                L = IDX_STOP - IDX_START + 1
                x_start = x_out[IDX_START]
                x_stop = x_out[IDX_STOP]
                coefficient = (x_stop - x_start)/ L
                
                x_out[IDX_START:IDX_STOP+1] = coefficient*_np.arange(L) + x_start
        else:
            for IDX in idx_spikes:
                delta = x_out[IDX] - x_out[IDX-1]
                x_out[IDX:] = x_out[IDX:] - D*delta
        
        return(x_out)

class ConvolutionalFilter(_Algorithm):
    """
    A class representing a convolutional filter algorithm.

    Parameters:
    -----------
    irftype : str
        The type of impulse response function (IRF) to use. Must be one of ['gauss', 'rect', 'triang', 'dgauss', 'custom'].
    win_len : int, optional
        The window length value for the IRF. Required when `irftype` is not 'custom'. Default is 0.
    irf : array-like, optional
        The custom impulse response function to use. Required when `irftype` is 'custom'. Default is None.
    normalize : bool, optional
        Flag indicating whether to normalize the impulse response function. Default is True.

    """
    def __init__(self, irftype, win_len=0, irf=None, normalize=True):
        assert irftype in ['gauss', 'rect', 'triang', 'dgauss', 'custom'],\
            "IRF type must be in ['gauss', 'rect', 'triang', 'dgauss', 'custom']"
        assert irftype == 'custom' or win_len > 0, "Window length value should be positive"
        _Algorithm.__init__(self, irftype=irftype, win_len=win_len, irf=irf, normalize=normalize)
        self.dimensions = {'time' : 0}

    # TODO (Andrea): TEST normalization and results
    def algorithm(self, signal):
        params = self._params
        irftype = params["irftype"]
        normalize = params["normalize"]

        fsamp = signal.p.get_sampling_freq()
        irf = None
        if irftype == 'custom':
            assert 'irf' in params, "'irf' parameter should be defined when irftype = 'custom'"
                
            irf = _np.array(params["irf"])
            n = len(irf)
        else:
            assert 'win_len' in params, "'win_len' should be defined when irftype is not 'custom'"
                
            n = int(params['win_len'] * fsamp)

            if irftype == 'gauss':
                std = _np.floor(n / 8)
                irf = _gaussian(n, std)
            elif irftype == 'rect':
                irf = _np.ones(n)

            elif irftype == 'triang':
                irf_1 = _np.arange(n // 2)
                irf_2 = irf_1[-1] - _np.arange(n // 2)
                if n % 2 == 0:
                    irf = _np.r_[irf_1, irf_2]
                else:
                    irf = _np.r_[irf_1, irf_1[-1] + 1, irf_2]
            elif irftype == 'dgauss':
                std = _np.round(n / 8)
                g = _gaussian(n, std)
                irf = _np.diff(g)

        # NORMALIZE
        if normalize:
            # irf = irf / (_np.sum(irf) * len(irf) / fsamp)
            irf = irf / _np.sum(irf)
        
        s = signal.values.ravel()
        
        signal_ = _np.r_[_np.ones(n) * s[0], s, _np.ones(n) * s[-1]]  # TESTME

        signal_f = _np.convolve(signal_, irf, mode='same')

        signal_out = signal_f[n:-n]
        return signal_out

class DeConvolutionalFilter(_Algorithm):
    """
    Class for performing deconvolution using different methods.

    Parameters
    ----------
    irf : array_like
        The Impulse Response Function (IRF) to be used for deconvolution.
    normalize : bool, optional
        Flag indicating whether to normalize the IRF before deconvolution.
        Defaults to True.
    deconv_method : {'fft', 'sps'}, optional
        The deconvolution method to be used. 'fft' is based on the computation of the Fast Fourier
        Transform; 'sps' uses the implementation in scipy.signal.    
    """
    def __init__(self, irf, normalize=True, deconv_method='sps'):
        assert deconv_method in ['fft', 'sps'], "Deconvolution method not valid"
        _Algorithm.__init__(self, irf=irf, normalize=normalize, deconv_method=deconv_method)
        self.dimensions = {'time' : 0}

    def algorithm(self, signal):
        params = self._params
        irf = params["irf"]
        normalize = params["normalize"]
        deconvolution_method = params["deconv_method"]

        fsamp = signal.p.get_sampling_freq()
        s = signal.values.ravel()
        if normalize:
            irf = irf / (_np.sum(irf) * len(irf) / fsamp)
        if deconvolution_method == 'fft':
            l = len(s)
            fft_signal = _np.fft.fft(s, n=l)
            fft_irf = _np.fft.fft(irf, n=l)
            out = abs(_np.fft.ifft(fft_signal / fft_irf))
            out[0] = out[1]
            out[-1] = out[-2]
        elif deconvolution_method == 'sps':
            print('sps based deconvolution needs to be tested. Use carefully.')
            out_dec, _ = _deconvolve(s, irf)
            
            #fix size
            #TODO half before, half after?
            out = _np.ones(len(signal))*out_dec[-1]
            out[:len(out_dec)] = out_dec
        else:
            print('Deconvolution method not implemented. Returning original signal.')
            out = s
        return out




class Prewhitening(_Algorithm):
    """
    """

    def __init__(self, order=50, optimize=True, **kwargs):
        _Algorithm.__init__(self, order=order, optimize=optimize, **kwargs)
        self.dimensions = {'time' : 0}

    def algorithm(self, signal, **kwargs):
        from statsmodels.tsa.ar_model import AutoReg as _AutoReg
        
        params = self._params
        order = params['order']
        optimize = params['optimize']
        
        signal_values = signal.p.get_values().ravel()
        
        if optimize:
            bic_ = []
            for i in _np.arange(1, order+1):
                model = _AutoReg(signal_values, lags=i)
                model_fit = model.fit()
                bic_.append(model_fit.bic)
            order_final = _np.argmin(bic_)+1
        else:
            order_final = order
            
        model = _AutoReg(signal_values, lags=order_final)
        model_fit = model.fit()
        
        prewhit_signal = signal_values
        prewhit_signal[order_final:] = model_fit.resid
        
        return(prewhit_signal)

'''
# TODO: check and convert to xarray

class DenoiseEDA(_Algorithm):
    """
    Remove noise due to sensor displacement from the EDA signal.
    
    Parameters
    ----------
    threshold : float, >0
        Threshold to detect the noise
        
    Optional parameters
    -------------------
    
    win_len : float, >0, default = 2
        Length of the window
   
    Returns
    -------
    signal : EvenlySignal
        De-noised signal
            
    """

    def __init__(self, threshold, win_len=2):
        assert threshold > 0, "Threshold value should be positive"
        assert win_len > 0, "Window length value should be positive"
        _Algorithm.__init__(self, threshold=threshold, win_len=win_len)

    @classmethod
    def algorithm(self, signal):
        params = self._params
        threshold = params['threshold']
        win_len = params['win_len']

        s = signal.values.ravel()
        # remove fluctuations
        noise = ConvolutionalFilter(irftype='triang', win_len=win_len, normalize=True)(abs(_np.diff(s)))

        # identify noisy portions
        idx_ok = _np.where(noise <= threshold)[0]

        # fix start and stop of the signal for the following interpolation
        if idx_ok[0] != 0:
            idx_ok = _np.r_[0, idx_ok].astype(int)

        if idx_ok[-1] != len(signal) - 1:
            idx_ok = _np.r_[idx_ok, len(signal) - 1].astype(int)

        denoised = _Signal(signal[idx_ok], sampling_freq=signal.get_sampling_freq(),
                           start_time = signal.get_start_time(),
                           x_values=idx_ok, x_type='indices')

        # interpolation
        signal_out = denoised.fill('linear')
        return signal_out
'''    
