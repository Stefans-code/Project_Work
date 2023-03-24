import numpy as _np
import scipy.stats as _stats
from scipy.signal import gaussian as _gaussian, filtfilt as _filtfilt, \
    filter_design as _filter_design, iirfilter as _iirfilter, \
        deconvolve as _deconvolve, firwin as _firwin, \
            iirnotch as _iirnotch, lfilter as _lfilter
from ._base_algorithm import _Algorithm
from .utils import SignalRange as _SignalRange

class Normalize(_Algorithm):
    """
    Normalizes the input signal using the general formula: (signal - BIAS) / RANGE.

    Parameters
    ----------
    norm_method : str, optional
        Method for normalization. Available methods are:
        * 'mean' - remove the mean [BIAS = mean(signal); RANGE = 1]
        * 'standard' - standardization [BIAS = mean(signal); RANGE = std(signal)]
        * 'min' - remove the minimum [BIAS = min(signal); RANGE = 1]
        * 'maxmin' - maxmin normalization [BIAS = min(signal); RANGE = (max(signal) - min(signal))]
        * 'custom' - custom, bias and range are manually defined [BIAS = bias, RANGE = range].
        Default is 'standard'.
    norm_bias : float, optional
        Bias for custom normalization. Default is 0.
    norm_range : float, optional
        Range for custom normalization. Must not be zero if norm_method is 'custom'. Default is 1.
    **kwargs : dict
        Additional keyword arguments to pass to _Algorithm.__init__().

    Returns
    -------
    signal : numpy.ndarray
        The normalized signal.

    Raises
    ------
    ValueError
        If norm_method is not one of 'mean', 'standard', 'min', 'maxmin', or 'custom'.

    Notes
    -----
    This class inherits from _Algorithm.

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
    Filter the input signal using an Infinite Impulse Response filter.

    Parameters
    ----------
    fp : list or float
        The pass frequencies
    fs : list or float
        The stop frequencies
    
    Optional parameters
    -------------------
    loss : float, >0, default = 0.1
        Loss tolerance in the pass band
    att : float, >0, default = 40
        Minimum attenuation required in the stop band.
    ftype : str, default = 'butter'
        Type of filter. Available types: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'

    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    Notes
    -----
    This is a wrapper of *scipy.signal.filter_design.iirdesign*. Refer to `scipy.signal.filter_design.iirdesign`
    for additional information
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
    Filter the input signal using an Infinite Impulse Response filter.

    Parameters
    ----------
    f : float
        The frequency to be removed
    Q : float
        Quality
    
    Returns
    -------
    signal : Signal
        Filtered signal

    Notes
    -----
    This is a wrapper of *scipy.signal.iirnotch*. Refer to `scipy.signal.iirnotch`
    for additional information
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
    Filter the input signal using a Finite Impulse Response filter.

    Parameters
    ----------
    fp : list or float
        The pass frequencies
    fs : list or float
        The stop frequencies
    
    Optional parameters
    -------------------
    loss : float, >0, default = 0.1
        Loss tolerance in the pass band
    att : float, >0, default = 40
        Minimum attenuation required in the stop band.
    wtype : str, default = 'hamming'
        Type of filter. Available types: 'hamming'

    Returns
    -------
    signal : EvenlySignal
        Filtered signal

    Notes
    -----
    This is a wrapper of *scipy.signal.firwin*. Refer to `scipy.signal.firwin`
    for additional information
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

class ImputeNAN(_Algorithm):
    def __init__(self, win_len=5, allnan='nan'):
        assert win_len>0, "win_len should be >0"
        _Algorithm.__init__(self, win_len = win_len)
        self.dimensions = {'time' : 0}
        
    def algorithm(self, signal):
        def group_consecutives(vals, step=1):
            """Return list of consecutive lists of numbers from vals (number list)."""
            run = []
            result = [run]
            expect = None
            for v in vals:
                if (v == expect) or (expect is None):
                    run.append(v)
                else:
                    run = [v]
                    result.append(run)
                expect = v + step
            return result

        #%
        params = self._params
        win_len = params['win_len']*signal.p.get_sampling_freq()
                
        s = signal.values.ravel()
        if _np.isnan(s).all():
            return(signal.values)
            
        idx_nan = _np.where(_np.isnan(s))[0]
        segments = group_consecutives(idx_nan)

        #%
        if len(segments[0])>=1:
            for i_seg, SEG in enumerate(segments):
                idx_st = SEG[0]
                idx_sp = SEG[-1]
                idx_win_pre = _np.arange(-int(win_len/2), 0, 1)+idx_st
                idx_win_pre = idx_win_pre[_np.where(idx_win_pre>0)[0]] #not before signal start

                STD = []
                if len(idx_win_pre)>=3:
                    STD.append(_np.nanstd(s[idx_win_pre]))
                
                idx_win_post = _np.arange(0, int(win_len/2))+idx_sp+1
                idx_win_post = idx_win_post[_np.where(idx_win_post<len(s))[0]]
                
                if len(idx_win_post)>=3:
                    STD.append(_np.nanstd(s[idx_win_post]))
                
                if len(STD)>0 and not (_np.isnan(STD).all()):
                    STD = _np.nanmin(STD)
                else:
                    STD = 0
                    
                idx_win = _np.hstack([idx_win_pre, idx_win_post]).astype(int)
                idx_win = idx_win[_np.where(~_np.isnan(s[idx_win]))[0]] # remove nans
                
                if len(idx_win)>3:
                    R = _stats.linregress(idx_win, s[idx_win])
                    s_nan = _np.array(SEG)*R[0]+R[1] + _np.random.normal(scale=STD, size = len(SEG))
                else:
                    s_nan = _np.nanmean(s)*_np.ones(len(SEG))
                s[SEG] = s_nan
        
        return(s)

class RemoveSpikes(_Algorithm):
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
    Filter a signal by convolution with a given impulse response function (IRF).

    Parameters
    ----------
    irftype : str
        Type of IRF to be generated. 'gauss', 'rect', 'triang', 'dgauss', 'custom'.
    win_len : float, >0 (> 8/fsamp for 'gaussian')
        Duration of the generated IRF in seconds (if irftype is not 'custom')
    
    Optional parameters
    -------------------
    irf : numpy.array
        IRF to be used if irftype is 'custom'
    normalize : boolean, default = True
        Whether to normalizes the IRF to have unitary area
    
    Returns
    -------
    signal : EvenlySignal
        Filtered signal

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
                if n < 8:
                    # TODO (Andrea): test, sometimes it returns nan
                    print(f"'win_len' too short to generate a gaussian IRF, expected > {str(_np.ceil(8 / fsamp))}")
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
            irf = irf / _np.sum(irf)
        
        s = signal.values.ravel()
        
        signal_ = _np.r_[_np.ones(n) * s[0], s, _np.ones(n) * s[-1]]  # TESTME

        signal_f = _np.convolve(signal_, irf, mode='same')

        signal_out = signal_f[n:-n]
        return signal_out

class DeConvolutionalFilter(_Algorithm):
    """
    Filter a signal by deconvolution with a given impulse response function (IRF).

    Parameters
    ----------
    irf : numpy.array
        IRF used to deconvolve the signal
    
    Optional parameters
    -------------------
    
    normalize : boolean, default = True
        Whether to normalize the IRF to have unitary area
    deconv_method : str, default = 'sps'
        Available methods: 'fft', 'sps'. 'fft' uses the fourier transform, 'sps' uses the scipy.signal.deconvolve
         function
        
    Returns
    -------
    signal : EvenlySignal
        Filtered signal

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

        s = signal.values.ravel()
        if normalize:
            irf = irf / _np.sum(irf)
        if deconvolution_method == 'fft':
            l = len(s)
            fft_signal = _np.fft.fft(s, n=l)
            fft_irf = _np.fft.fft(irf, n=l)
            out = _np.fft.ifft(fft_signal / fft_irf)
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
