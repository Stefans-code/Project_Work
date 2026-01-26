"""
Tests for the utils module including signal processing utilities.
"""
import numpy as np
import pytest
from pyphysio.signal import create_signal
from pyphysio.utils import Diff, Wavelet, PCA, PeakDetection, PeakSelection, Maxima, PSD, SignalRange
from pyphysio import TestData

# TODO-AI [PRIORITY: HIGH]: Use accessor `signal.p.get_values()` for numeric checks.
# TODO-AI [PRIORITY: MEDIUM]: Reduce extremely large test arrays to speed CI.
# TODO-AI [PRIORITY: MEDIUM]: Seed random inputs via `rng` fixture to avoid flakiness.


class TestDiffFilter:
    """Test the Diff filter (difference calculation)."""

    def test_diff_basic(self):
        """Test basic difference calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=1)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        assert result is not None
        assert result.shape == signal.shape

    def test_diff_degree_1(self):
        """Test difference with degree 1."""
        data = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        signal = create_signal(data, sampling_freq=1)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        # Differences should be [2, 3, 4, 5, ...]
        assert result is not None

    def test_diff_degree_2(self):
        """Test difference with degree 2."""
        data = np.arange(100, dtype=float)
        signal = create_signal(data, sampling_freq=1)
        diff_filter = Diff(degree=2)
        result = diff_filter(signal)
        assert result is not None
        assert result.shape == signal.shape

    def test_diff_invalid_degree(self):
        """Test that invalid degree raises error."""
        with pytest.raises(AssertionError):
            Diff(degree=0)
        with pytest.raises(AssertionError):
            Diff(degree=-1)

    def test_diff_with_real_data(self):
        """Test difference filter with real test data."""
        ecg = TestData.ecg(return_signal=True)
        diff_filter = Diff(degree=1)
        result = diff_filter(ecg)
        assert result is not None
        assert result.shape == ecg.shape

    def test_diff_multidimensional(self):
        """Test difference filter with multidimensional data."""
        data = np.random.uniform(size=(1000, 5, 2))
        signal = create_signal(data, sampling_freq=100)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        assert result is not None

    def test_diff_preserves_shape(self):
        """Test that diff preserves signal shape."""
        for shape in [(100,), (100, 5), (100, 5, 2)]:
            data = np.random.uniform(size=shape)
            signal = create_signal(data, sampling_freq=10)
            diff_filter = Diff(degree=1)
            result = diff_filter(signal)
            assert result.shape == signal.shape


class TestSignalProcessingChain:
    """Test chaining multiple processing operations."""

    def test_diff_chain(self):
        """Test chaining multiple diff operations."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        diff1 = Diff(degree=1)
        result1 = diff1(signal)
        diff2 = Diff(degree=1)
        result2 = diff2(result1)
        assert result2 is not None
        assert result2.shape == signal.shape

    def test_diff_with_varying_signal_sizes(self):
        """Test diff filter with different signal sizes."""
        sizes = [10, 100, 1000, 10000]
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=100)
            diff_filter = Diff(degree=1)
            result = diff_filter(signal)
            assert result.shape[0] == size

    def test_diff_with_constant_signal(self):
        """Test diff filter with constant signal."""
        data = np.ones(100) * 5.0
        signal = create_signal(data, sampling_freq=1)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        # Differences of constant signal should be close to zero
        assert np.abs(np.nanmean(result.values)) < 0.1

    def test_diff_with_linear_signal(self):
        """Test diff filter with linear signal."""
        data = np.linspace(0, 100, 100)
        signal = create_signal(data, sampling_freq=1)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        # Differences should be approximately constant
        assert result is not None


class TestUtilsNumericalStability:
    """Test numerical stability of utility functions."""

    def test_diff_with_large_values(self):
        """Test diff filter with large values."""
        data = np.random.uniform(1e6, 1e7, size=1000)
        signal = create_signal(data, sampling_freq=100)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        assert not np.any(np.isnan(result.values))

    def test_diff_with_small_values(self):
        """Test diff filter with very small values."""
        data = np.random.uniform(1e-10, 1e-9, size=1000)
        signal = create_signal(data, sampling_freq=100)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        assert not np.any(np.isinf(result.values))

    def test_diff_with_mixed_sign_values(self):
        """Test diff filter with mixed positive and negative values."""
        data = np.concatenate([
            np.linspace(-100, 0, 500),
            np.linspace(0, 100, 500)
        ])
        signal = create_signal(data, sampling_freq=100)
        diff_filter = Diff(degree=1)
        result = diff_filter(signal)
        assert result is not None


class TestWavelet:
    """Test Wavelet transform utility."""

    def test_wavelet_2d_signal(self):
        """Test wavelet transform on 2D signal."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        wavelet = Wavelet()
        result = wavelet(signal)
        
        assert result is not None
        assert 'freq' in result.coords

    def test_wavelet_freq_coordinates(self):
        """Test that wavelet result has frequency coordinates."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        wavelet = Wavelet()
        result = wavelet(signal)
        
        freqs = result.freq.values
        assert freqs is not None
        assert len(freqs) > 0

    def test_wavelet_compute_coi(self):
        """Test wavelet COI (Cone of Influence) computation."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        wavelet = Wavelet()
        result = wavelet(signal)
        coi = wavelet._compute_coi(result)
        
        assert coi is not None
        assert coi.shape == result.shape

    def test_wavelet_dimension_increase(self):
        """Test that wavelet increases dimensionality by 1."""
        sizes = [1000, (1000, 1), (1000, 5)]
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=100)
            
            wavelet = Wavelet()
            result = wavelet(signal)
            
            assert result.p.get_values().ndim == signal.p.get_values().ndim + 1


class TestPCA:
    """Test Principal Component Analysis utility."""

    def test_pca_4d_signal(self):
        """Test PCA on 4D signal."""
        data = np.random.uniform(size=(1000, 5, 6, 7))
        signal = create_signal(data, sampling_freq=100)
        
        pca = PCA(dimension='dimension_4')
        result = pca(signal)
        
        assert result is not None
        assert result.shape[0] == signal.shape[0]

    def test_pca_dimension_reduction(self):
        """Test that PCA reduces specified dimension."""
        data = np.random.uniform(size=(1000, 5, 6, 7))
        signal = create_signal(data, sampling_freq=100)
        
        pca = PCA(dimension='dimension_4')
        result = pca(signal)
        
        # Result should have reduced dimension_4 size
        assert result.shape[-1] < signal.shape[-1]

    def test_pca_with_3d_data(self):
        """Test PCA with 3D signal."""
        data = np.random.uniform(size=(1000, 5, 6))
        signal = create_signal(data, sampling_freq=100)
        
        pca = PCA(dimension='component')
        result = pca(signal)
        
        assert result is not None


class TestPeakDetection:
    """Test Peak Detection utility."""

    def test_peak_detection_basic(self):
        """Test basic peak detection."""
        freqs = np.arange(0, 10)
        t = np.arange(0, 20, 0.05)
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        peak_detector = PeakDetection(0.1, return_peaks=True)
        peaks = peak_detector(signal)
        
        assert peaks is not None
        assert peaks.shape == signal.shape

    def test_peak_detection_sinusoid_count(self):
        """Test that peak detection correctly identifies sinusoid peaks."""
        freqs = np.arange(0, 10)
        t = np.arange(0, 20, 0.05)
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        peak_detector = PeakDetection(0.1, return_peaks=True)
        peaks = peak_detector(signal)
        
        for i in np.arange(1, len(freqs)):
            peaks_ch = peaks.sel(channel=i).dropna(dim='time')
            t_max = peaks_ch.p.get_times()
            expected_count = int(20 * freqs[i])
            assert len(t_max) == expected_count, \
                f"Expected {expected_count} peaks for frequency {freqs[i]}, got {len(t_max)}"

    def test_peak_detection_dimension_preservation(self):
        """Test that peak detection preserves signal dimensions."""
        sizes = [(100,), (100, 5), (100, 5, 2)]
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=20)
            
            peak_detector = PeakDetection(0.1, return_peaks=True)
            peaks = peak_detector(signal)
            
            assert peaks.p.get_values().ndim == signal.p.get_values().ndim


class TestPeakSelection:
    """Test Peak Selection utility."""

    def test_peak_selection_basic(self):
        """Test basic peak selection."""
        freqs = np.arange(0, 10)
        t = np.arange(0, 20, 0.05)
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        peak_detector = PeakDetection(0.1, return_peaks=True)
        peaks = peak_detector(signal)
        
        signal_with_peaks = signal.copy(deep=True)
        signal_with_peaks['peaks'] = peaks
        
        peak_selector = PeakSelection(win_pre=1, win_post=1)
        result = peak_selector(signal_with_peaks)
        
        assert result is not None

    def test_peak_selection_window_parameters(self):
        """Test peak selection with different window parameters."""
        data = np.random.uniform(size=(1000, 5))
        signal = create_signal(data, sampling_freq=20)
        
        peak_detector = PeakDetection(0.1, return_peaks=True)
        peaks = peak_detector(signal)
        
        signal_with_peaks = signal.copy(deep=True)
        signal_with_peaks['peaks'] = peaks
        
        for win_pre, win_post in [(1, 1), (2, 2), (0.5, 0.5)]:
            peak_selector = PeakSelection(win_pre=win_pre, win_post=win_post)
            result = peak_selector(signal_with_peaks)
            assert result is not None


class TestMaxima:
    """Test Maxima detection utility."""

    def test_maxima_basic(self):
        """Test basic maxima detection."""
        data = np.random.uniform(size=(1000, 5))
        signal = create_signal(data, sampling_freq=100)
        
        maxima = Maxima()
        result = maxima(signal)
        
        assert result is not None
        assert result.shape == signal.shape

    def test_maxima_sinusoid_count(self):
        """Test maxima detection on sinusoids."""
        freqs = np.arange(1, 10)
        t = np.arange(0, 20, 0.05)
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        maxima = Maxima()
        result = maxima(signal)
        
        for i in np.arange(1, len(freqs)):
            res_ch = result.sel(channel=i).dropna(dim='time')
            t_max = res_ch.p.get_times()
            expected_count = int(20 * freqs[i])
            assert len(t_max) == expected_count, \
                f"Expected {expected_count} maxima for frequency {freqs[i]}, got {len(t_max)}"

    def test_maxima_dimension_preservation(self):
        """Test that maxima preserves signal dimensions."""
        sizes = [(100,), (100, 5), (100, 5, 2)]
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=100)
            
            maxima = Maxima()
            result = maxima(signal)
            
            assert result.p.get_values().ndim == signal.p.get_values().ndim


class TestPSD:
    """Test Power Spectral Density computation."""

    def test_psd_welch_method(self):
        """Test PSD with Welch method."""
        freqs = np.arange(1, 10)
        t = np.arange(0, 20, 0.05)
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        psd = PSD('welch')
        result = psd(signal)
        
        assert result is not None
        assert 'freq' in result.coords

    def test_psd_welch_peak_detection(self):
        """Test that PSD correctly identifies frequency peaks with Welch method."""
        freqs = np.arange(2, 10)  # Start from 2 Hz to avoid edge cases
        t = np.arange(0, 40, 0.05)  # Longer signal for better frequency resolution
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        psd = PSD('welch')
        result = psd(signal)
        
        for i in np.arange(len(freqs)):
            idx_max = np.argmax(result.p.get_values()[:, i])
            detected_freq = result.coords['freq'].values[idx_max]
            # Allow 1 Hz tolerance due to frequency resolution limits
            assert abs(detected_freq - freqs[i]) < 1.0, \
                f"Expected peak near {freqs[i]} Hz, detected at {detected_freq} Hz"

    def test_psd_period_method(self):
        """Test PSD with period method."""
        freqs = np.arange(1, 10)
        t = np.arange(0, 20, 0.05)
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        psd = PSD('period')
        result = psd(signal)
        
        assert result is not None
        assert 'freq' in result.coords

    def test_psd_period_peak_detection(self):
        """Test that PSD correctly identifies peaks with period method."""
        freqs = np.arange(2, 10)  # Start from 2 Hz to avoid edge cases
        t = np.arange(0, 40, 0.05)  # Longer signal for better frequency resolution
        data = np.array([np.sin(2*np.pi*x*t) for x in freqs]).T
        signal = create_signal(data, sampling_freq=20)
        
        psd = PSD('period')
        result = psd(signal)
        
        for i in np.arange(len(freqs)):
            idx_max = np.argmax(result.p.get_values()[:, i])
            detected_freq = result.coords['freq'].values[idx_max]
            # Allow 1 Hz tolerance due to frequency resolution limits
            assert abs(detected_freq - freqs[i]) < 1.0, \
                f"Expected peak near {freqs[i]} Hz, detected at {detected_freq} Hz"

    def test_psd_dimension_increase(self):
        """Test that PSD increases dimensionality by 1."""
        sizes = [1000, (1000, 1), (1000, 5)]
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=100)
            
            psd = PSD('welch')
            result = psd(signal)
            
            assert result.p.get_values().ndim == signal.p.get_values().ndim + 1


class TestSignalRange:
    """Test Signal Range detection utility."""

    def test_signal_range_basic(self):
        """Test basic signal range detection."""
        ampl = np.arange(1, 11)
        t = np.arange(0, 20, 0.05)
        data = np.array([A*np.sin(2*np.pi*t) for A in ampl]).T
        signal = create_signal(data, sampling_freq=20)
        
        sigrange = SignalRange(1, 0.5)
        result = sigrange(signal)
        
        assert result is not None
        assert result.shape == signal.shape

    def test_signal_range_amplitude_detection(self):
        """Test that signal range correctly detects amplitude."""
        ampl = np.arange(1, 11)
        t = np.arange(0, 20, 0.05)
        data = np.array([A*np.sin(2*np.pi*t) for A in ampl]).T
        signal = create_signal(data, sampling_freq=20)
        
        sigrange = SignalRange(1, 0.5)
        result = sigrange(signal)
        
        for i in np.arange(1, len(ampl)):
            sigrange_ch = result.sel(channel=i).dropna(dim='time')
            max_range = np.max(sigrange_ch.p.get_values())
            expected_range = 2 * ampl[i]
            assert abs(max_range - expected_range) < 0.001, \
                f"Expected range {expected_range}, got {max_range} for amplitude {ampl[i]}"

    def test_signal_range_dimension_preservation(self):
        """Test that signal range preserves signal dimensions."""
        sizes = [(100,), (100, 5), (100, 5, 2)]
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=20)
            
            sigrange = SignalRange(1, 0.5)
            result = sigrange(signal)
            
            assert result.p.get_values().ndim == signal.p.get_values().ndim


class TestUtilsMultipleSizes:
    """Test utilities with multiple signal sizes."""

    def test_utilities_multiple_sizes(self):
        """Test all utilities work with various signal sizes."""
        sizes = [1000, (1000,), (1000, 1), (1000, 1, 1),
                 (1000, 5), (1000, 5, 2), (1000, 2, 3, 5)]
        
        for size in sizes:
            data = np.random.uniform(size=size)
            signal = create_signal(data, sampling_freq=100)
            
            # PSD
            psd = PSD('welch')
            pwd = psd(signal)
            assert pwd.p.get_values().ndim == signal.p.get_values().ndim + 1
            
            # Diff
            diff = Diff()
            diff_result = diff(signal)
            assert diff_result.p.get_values().ndim == signal.p.get_values().ndim
            
            # Maxima
            maxx = Maxima()
            max_result = maxx(signal)
            assert max_result.p.get_values().ndim == signal.p.get_values().ndim
            
            # PeakDetection
            peaks = PeakDetection(0.1, return_peaks=True)
            peaks_result = peaks(signal)
            assert peaks_result.p.get_values().ndim == signal.p.get_values().ndim
            
            # SignalRange
            sigrange = SignalRange(1, 0.5)
            sigrange_result = sigrange(signal)
            assert sigrange_result.p.get_values().ndim == signal.p.get_values().ndim
            
            # Wavelet
            wavelet = Wavelet()
            wave_result = wavelet(signal)
            assert wave_result.p.get_values().ndim == signal.p.get_values().ndim + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
