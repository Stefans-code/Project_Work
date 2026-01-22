import pytest
import numpy as np
from pyphysio.signal import create_signal
import pyphysio.filters as filt


class TestNormalize:
    """Tests for Normalize filter."""
    
    def test_normalize_standard(self):
        """Test standardization normalization method."""
        data = np.random.uniform(0, 100, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='standard')
        result = normalizer(signal)
        
        assert result.values.shape == signal.values.shape
        assert np.abs(np.nanmean(result.values)) < 1e-10
        assert np.abs(np.nanstd(result.values) - 1.0) < 1e-10
    
    def test_normalize_mean(self):
        """Test mean subtraction normalization."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='mean')
        result = normalizer(signal)
        
        expected = data - np.mean(data)
        np.testing.assert_array_almost_equal(result.values.ravel(), expected)
    
    def test_normalize_min(self):
        """Test min subtraction normalization."""
        data = np.array([5, 10, 15, 20], dtype=float)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='min')
        result = normalizer(signal)
        
        expected = data - np.min(data)
        np.testing.assert_array_almost_equal(result.values.ravel(), expected)
    
    def test_normalize_maxmin(self):
        """Test min-max normalization (scaling to [0, 1])."""
        data = np.array([0, 50, 100], dtype=float)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='maxmin')
        result = normalizer(signal)
        
        expected = np.array([0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result.values.ravel(), expected)
    
    def test_normalize_custom(self):
        """Test custom normalization with bias and range."""
        data = np.array([10, 20, 30], dtype=float)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='custom', norm_bias=10, norm_range=10)
        result = normalizer(signal)
        
        expected = (data - 10) / 10
        np.testing.assert_array_almost_equal(result.values.ravel(), expected)
    
    def test_normalize_invalid_method(self):
        """Test that invalid normalization method raises assertion."""
        with pytest.raises(AssertionError):
            filt.Normalize(norm_method='invalid')
    
    def test_normalize_custom_zero_range(self):
        """Test that custom normalization with zero range raises assertion."""
        with pytest.raises(AssertionError):
            filt.Normalize(norm_method='custom', norm_range=0)
    
    def test_normalize_multidimensional(self):
        """Test normalization with multidimensional signals."""
        data = np.random.uniform(0, 100, (1000, 5, 2))
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='standard')
        result = normalizer(signal)
        
        assert result.values.shape == signal.values.shape


class TestIIRFilter:
    """Tests for IIR filter."""
    
    def test_iirfilter_lowpass(self):
        """Test IIR lowpass filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=20, btype='lowpass')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape
        assert not np.any(np.isnan(result.values))
    
    def test_iirfilter_highpass(self):
        """Test IIR highpass filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=5, btype='highpass')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape
        assert not np.any(np.isnan(result.values))
    
    def test_iirfilter_bandpass(self):
        """Test IIR bandpass filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=[20, 30], btype='bandpass')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape
        assert not np.any(np.isnan(result.values))
    
    def test_iirfilter_bandstop(self):
        """Test IIR bandstop (notch-like) filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=[20, 30], btype='bandstop')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_iirfilter_ftype_butter(self):
        """Test IIR with Butterworth filter type."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=20, btype='lowpass', ftype='butter')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_iirfilter_ftype_cheby1(self):
        """Test IIR with Chebyshev Type 1 filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=20, btype='lowpass', ftype='cheby1')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_iirfilter_invalid_ftype(self):
        """Test that invalid filter type raises assertion."""
        with pytest.raises(AssertionError):
            filt.IIRFilter(fp=20, ftype='invalid')
    
    def test_iirfilter_invalid_loss(self):
        """Test that non-positive loss raises assertion."""
        with pytest.raises(AssertionError):
            filt.IIRFilter(fp=20, loss=-0.1)
    
    def test_iirfilter_invalid_att(self):
        """Test that non-positive attenuation raises assertion."""
        with pytest.raises(AssertionError):
            filt.IIRFilter(fp=20, att=-40)
    
    def test_iirfilter_att_not_greater_than_loss(self):
        """Test that attenuation must be greater than loss."""
        with pytest.raises(AssertionError):
            filt.IIRFilter(fp=20, loss=50, att=40)
    
    def test_iirfilter_with_stopband(self):
        """Test IIR filter with explicit stopband specification."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        iir = filt.IIRFilter(fp=20, fs=25, btype='lowpass')
        result = iir(signal)
        
        assert result.values.shape == signal.values.shape


class TestNotchFilter:
    """Tests for Notch filter."""
    
    def test_notchfilter_basic(self):
        """Test basic notch filter operation."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        notch = filt.NotchFilter(f=50)
        result = notch(signal)
        
        assert result.values.shape == signal.values.shape
        assert not np.any(np.isnan(result.values))
    
    def test_notchfilter_custom_Q(self):
        """Test notch filter with custom quality factor."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        notch = filt.NotchFilter(f=50, Q=60)
        result = notch(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_notchfilter_low_frequency(self):
        """Test notch filter at low frequency."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        notch = filt.NotchFilter(f=1)
        result = notch(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_notchfilter_invalid_frequency(self):
        """Test that non-positive frequency raises assertion."""
        with pytest.raises(AssertionError):
            filt.NotchFilter(f=-50)
    
    def test_notchfilter_invalid_Q(self):
        """Test that non-positive Q raises assertion."""
        with pytest.raises(AssertionError):
            filt.NotchFilter(f=50, Q=-30)


class TestFIRFilter:
    """Tests for FIR filter."""
    
    def test_firfilter_lowpass(self):
        """Test FIR lowpass filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        fir = filt.FIRFilter(fp=20, btype='lowpass', order=10)
        result = fir(signal)
        
        assert result.values.shape == signal.values.shape
        assert not np.any(np.isnan(result.values))
    
    def test_firfilter_highpass(self):
        """Test FIR highpass filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        fir = filt.FIRFilter(fp=5, btype='highpass', order=10)
        result = fir(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_firfilter_bandpass(self):
        """Test FIR bandpass filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        fir = filt.FIRFilter(fp=[20, 30], fs=[15, 35], btype='bandpass', order=20)
        result = fir(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_firfilter_bandstop(self):
        """Test FIR bandstop filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        fir = filt.FIRFilter(fp=[15, 35], fs=[20, 30], btype='bandstop', order=20)
        result = fir(signal)
        
        assert result.values.shape == signal.values.shape
    
    def test_firfilter_invalid_attenuation(self):
        """Test that non-positive attenuation raises assertion."""
        with pytest.raises(AssertionError):
            filt.FIRFilter(fp=20, att=-40)
    
    def test_firfilter_invalid_window(self):
        """Test that invalid window type raises assertion."""
        with pytest.raises(AssertionError):
            filt.FIRFilter(fp=20, wtype='invalid')


class TestKalmanFilter:
    """Tests for Kalman filter."""
    
    def test_kalmanfilter_basic(self):
        """Test basic Kalman filter operation."""
        data = np.random.uniform(0, 1, 1000) + np.random.normal(0, 0.1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        kalman = filt.KalmanFilter(R=100, Q=1)
        result = kalman(signal)
        
        assert result.shape == signal.values.ravel().shape
        assert not np.any(np.isnan(result))
    
    def test_kalmanfilter_different_parameters(self):
        """Test Kalman filter with different R and Q values."""
        data = np.random.uniform(0, 1, 500)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        kalman = filt.KalmanFilter(R=50, Q=10)
        result = kalman(signal)
        
        assert result.shape == (500,)
    
    def test_kalmanfilter_invalid_R(self):
        """Test that non-positive R raises assertion."""
        with pytest.raises(AssertionError):
            filt.KalmanFilter(R=-100, Q=1)
    
    def test_kalmanfilter_invalid_Q(self):
        """Test that non-positive Q raises assertion."""
        with pytest.raises(AssertionError):
            filt.KalmanFilter(R=100, Q=-1)


class TestRemoveSpikes:
    """Tests for RemoveSpikes filter."""
    
    def test_removespikes_default(self):
        """Test basic spike removal with default parameters."""
        data = np.random.uniform(0, 1, 1000)
        # Add some artificial spikes
        data[100] += 10
        data[500] += 10
        
        signal = create_signal(data, sampling_freq=100, name='test')
        
        remover = filt.RemoveSpikes(K=2)
        result = remover(signal)
        
        assert result.shape == signal.values.ravel().shape
        assert not np.any(np.isnan(result))
    
    def test_removespikes_step_method(self):
        """Test spike removal with step method."""
        data = np.ones(1000)
        data[100:110] += 5  # Add a spike region
        signal = create_signal(data, sampling_freq=100, name='test')
        
        remover = filt.RemoveSpikes(K=2, method='step', D=0.9)
        result = remover(signal)
        
        assert result.shape == (1000,)
    
    def test_removespikes_linear_method(self):
        """Test spike removal with linear method."""
        data = np.ones(1000)
        data[100] += 5
        data[200] += 5
        signal = create_signal(data, sampling_freq=100, name='test')
        
        remover = filt.RemoveSpikes(K=2, method='linear', N=1)
        result = remover(signal)
        
        assert result.shape == (1000,)
    
    def test_removespikes_with_dilation(self):
        """Test spike removal with dilation window."""
        data = np.ones(1000) + np.random.normal(0, 0.1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        remover = filt.RemoveSpikes(K=3, dilate=0.1)
        result = remover(signal)
        
        assert result.shape == (1000,)
    
    def test_removespikes_invalid_K(self):
        """Test that non-positive K raises assertion."""
        with pytest.raises(AssertionError):
            filt.RemoveSpikes(K=-2)
    
    def test_removespikes_invalid_N(self):
        """Test that invalid N raises assertion."""
        with pytest.raises(AssertionError):
            filt.RemoveSpikes(N=-1)
    
    def test_removespikes_invalid_dilate(self):
        """Test that negative dilate raises assertion."""
        with pytest.raises(AssertionError):
            filt.RemoveSpikes(dilate=-0.1)
    
    def test_removespikes_invalid_D(self):
        """Test that negative D raises assertion."""
        with pytest.raises(AssertionError):
            filt.RemoveSpikes(D=-0.5)
    
    def test_removespikes_invalid_method(self):
        """Test that invalid method raises assertion."""
        with pytest.raises(AssertionError):
            filt.RemoveSpikes(method='invalid')


class TestConvolutionalFilter:
    """Tests for Convolutional filter."""
    
    def test_convolutional_gauss(self):
        """Test Gaussian convolutional filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        conv = filt.ConvolutionalFilter('gauss', win_len=0.5)
        result = conv(signal)
        
        assert result.shape == signal.values.ravel().shape
        assert not np.any(np.isnan(result))
    
    def test_convolutional_rect(self):
        """Test rectangular convolutional filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        conv = filt.ConvolutionalFilter('rect', win_len=0.5)
        result = conv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_convolutional_triang(self):
        """Test triangular convolutional filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        conv = filt.ConvolutionalFilter('triang', win_len=0.5)
        result = conv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_convolutional_dgauss(self):
        """Test derivative of Gaussian convolutional filter."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        conv = filt.ConvolutionalFilter('dgauss', win_len=0.5)
        result = conv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_convolutional_custom(self):
        """Test custom convolutional filter with user-defined impulse response."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        custom_irf = np.array([0.25, 0.5, 0.25])
        conv = filt.ConvolutionalFilter('custom', irf=custom_irf)
        result = conv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_convolutional_no_normalize(self):
        """Test convolutional filter without normalization."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        conv = filt.ConvolutionalFilter('rect', win_len=0.5, normalize=False)
        result = conv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_convolutional_invalid_irf_type(self):
        """Test that invalid IRF type raises assertion."""
        with pytest.raises(AssertionError):
            filt.ConvolutionalFilter('invalid', win_len=0.5)
    
    def test_convolutional_missing_win_len(self):
        """Test that missing win_len for non-custom filter raises assertion."""
        with pytest.raises(AssertionError):
            filt.ConvolutionalFilter('gauss', win_len=0)
    
    def test_convolutional_custom_missing_irf(self):
        """Test that custom filter with missing irf raises error when applied."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        conv = filt.ConvolutionalFilter('custom')
        with pytest.raises((AssertionError, KeyError, TypeError)):
            conv(signal)


class TestDeConvolutionalFilter:
    """Tests for Deconvolutional filter."""
    
    def test_deconvolutional_fft(self):
        """Test deconvolutional filter using FFT method."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        irf = np.array([0.5, 0.3, 0.2])
        deconv = filt.DeConvolutionalFilter(irf, deconv_method='fft')
        result = deconv(signal)
        
        assert result.shape == signal.values.ravel().shape
        assert not np.any(np.isnan(result))
    
    def test_deconvolutional_sps(self):
        """Test deconvolutional filter using SPS method."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        irf = np.array([0.5, 0.3, 0.2])
        deconv = filt.DeConvolutionalFilter(irf, deconv_method='sps')
        result = deconv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_deconvolutional_no_normalize(self):
        """Test deconvolutional filter without normalization."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        irf = np.array([0.5, 0.3, 0.2])
        deconv = filt.DeConvolutionalFilter(irf, normalize=False, deconv_method='fft')
        result = deconv(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_deconvolutional_invalid_method(self):
        """Test that invalid deconvolution method raises assertion."""
        with pytest.raises(AssertionError):
            filt.DeConvolutionalFilter([0.5, 0.3], deconv_method='invalid')


class TestPrewhitening:
    """Tests for Prewhitening filter."""
    
    def test_prewhitening_optimize(self):
        """Test prewhitening with AR model optimization."""
        # Generate an autocorrelated signal
        data = np.cumsum(np.random.normal(0, 1, 500))
        signal = create_signal(data, sampling_freq=100, name='test')
        
        whitener = filt.Prewhitening(p=1, optimize=True, pmin=1, pmax=5)
        result = whitener(signal)
        
        assert result.shape == signal.values.ravel().shape
        assert hasattr(whitener, 'f'), "Filter coefficients should be stored"
    
    def test_prewhitening_no_optimize(self):
        """Test prewhitening without AR model optimization."""
        data = np.random.normal(0, 1, 500)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        whitener = filt.Prewhitening(p=3, optimize=False)
        result = whitener(signal)
        
        assert result.shape == signal.values.ravel().shape
    
    def test_prewhitening_different_orders(self):
        """Test prewhitening with different AR orders."""
        data = np.random.normal(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        for order in [1, 3, 5]:
            whitener = filt.Prewhitening(p=order, optimize=False)
            result = whitener(signal)
            assert result.shape == (1000,)


class TestMultipleFiltros:
    """Integration tests with multiple filters."""
    
    def test_all_filters_on_1d_signal(self):
        """Test all filters work on 1D signals."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        filters = [
            filt.Normalize(),
            filt.ConvolutionalFilter('rect', 0.5),
            filt.DeConvolutionalFilter([0.01, 0.005], deconv_method='fft'),
            filt.FIRFilter(20, order=10),
            filt.IIRFilter(20, btype='lowpass'),
            filt.KalmanFilter(100, 100),
            filt.NotchFilter(25),
            filt.RemoveSpikes(),
            filt.Prewhitening(pmin=1, pmax=3)
        ]
        
        for f in filters:
            result = f(signal)
            assert result is not None
    
    def test_all_filters_on_multidimensional_signal(self):
        """Test all filters work on multidimensional signals."""
        data = np.random.uniform(0, 1, (1000, 5, 2))
        signal = create_signal(data, sampling_freq=100, name='test')
        
        filters = [
            filt.Normalize(),
            filt.ConvolutionalFilter('rect', 0.5),
            filt.IIRFilter(20, btype='lowpass'),
        ]
        
        for f in filters:
            result = f(signal)
            assert result.values.ndim == signal.values.ndim
    
    def test_sequential_filtering(self):
        """Test applying multiple filters sequentially."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        # Apply filters in sequence
        signal = filt.Normalize()(signal)
        signal = filt.IIRFilter(20, btype='lowpass')(signal)
        signal = filt.RemoveSpikes()(signal)
        
        assert signal is not None
        assert signal.values.shape[0] == 1000
    
    def test_filters_preserve_signal_properties(self):
        """Test that filters preserve basic signal properties."""
        data = np.random.uniform(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        # Most filters should not change the length
        f = filt.IIRFilter(20, btype='lowpass')
        result = f(signal)
        assert len(result) == 1000


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_filter_with_very_short_signal(self):
        """Test filters with very short signals."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize()
        result = normalizer(signal)
        assert len(result.values.ravel()) == 5
    
    def test_filter_with_nan_values(self):
        """Test filter behavior with NaN values."""
        data = np.array([1, 2, np.nan, 4, 5], dtype=float)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='mean')
        result = normalizer(signal)
        # Should still produce output (may contain NaN)
        assert len(result.values.ravel()) == 5
    
    def test_filter_with_zero_signal(self):
        """Test filters with all-zero signal."""
        data = np.zeros(1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='mean')
        result = normalizer(signal)
        # Zero signal minus mean (zero) should be zero
        np.testing.assert_array_almost_equal(result.values.ravel(), np.zeros(1000))
    
    def test_filter_with_constant_signal(self):
        """Test filters with constant signal."""
        data = np.ones(1000) * 5
        signal = create_signal(data, sampling_freq=100, name='test')
        
        normalizer = filt.Normalize(norm_method='min')
        result = normalizer(signal)
        # Constant signal minus its min should be zero
        np.testing.assert_array_almost_equal(result.values.ravel(), np.zeros(1000))
    
    def test_different_sampling_frequencies(self):
        """Test filters with different sampling frequencies."""
        sampling_freqs = [10, 50, 100, 500, 1000]
        
        for sf in sampling_freqs:
            data = np.random.uniform(0, 1, 1000)
            signal = create_signal(data, sampling_freq=sf, name='test')
            
            f = filt.Normalize()
            result = f(signal)
            assert len(result.values.ravel()) == 1000