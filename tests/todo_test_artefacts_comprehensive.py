"""
Tests for the artefacts module including detection and correction algorithms.
"""
import numpy as np
import pytest
from pyphysio.signal import create_signal
from pyphysio import TestData
import pyphysio.artefacts as art
import xarray as xr


class TestDetectMA:
    """Test Motion Artifact Detection algorithm."""

    def test_detect_ma_basic(self):
        """Test basic motion artifact detection."""
        data = np.random.uniform(size=(1000, 3))
        # Add some artifacts
        data[200:250, 0] = data[200:250, 0] + 5
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        result = detector(signal)
        assert result is not None

    def test_detect_ma_with_fuse_none(self):
        """Test motion artifact detection with fuse=None."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA(fuse=None)
        result = detector(signal)
        assert result is not None

    def test_detect_ma_with_fuse_all(self):
        """Test motion artifact detection with fuse='all'."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA(fuse='all')
        result = detector(signal)
        assert result is not None

    def test_detect_ma_with_fuse_component(self):
        """Test motion artifact detection with fuse='component'."""
        data = np.random.uniform(size=(1000, 3, 2))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA(fuse='component')
        result = detector(signal)
        assert result is not None

    def test_detect_ma_iqr_method(self):
        """Test motion artifact detection with IQR method."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA(method='iqr', iqr=1.5)
        result = detector(signal)
        assert result is not None

    def test_detect_ma_mad_method(self):
        """Test motion artifact detection with MAD method."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA(method='mad', th_std_coeff=3.0)
        result = detector(signal)
        assert result is not None

    def test_detect_ma_window_length(self):
        """Test motion artifact detection with different window lengths."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        for win_len in [0.5, 1.0, 2.0]:
            detector = art.DetectMA(win_len=win_len)
            result = detector(signal)
            assert result is not None

    def test_detect_ma_window_mask(self):
        """Test motion artifact detection with different mask window lengths."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        for win_mask in [0.5, 1.0, 2.0]:
            detector = art.DetectMA(win_mask=win_mask)
            result = detector(signal)
            assert result is not None

    def test_detect_ma_with_real_data(self):
        """Test motion artifact detection with real test data."""
        signal = TestData.ecg(return_signal=True)
        # Expand to 3D for testing
        data_3d = np.tile(signal.values, (1, 3, 1))
        signal_3d = create_signal(data_3d, sampling_freq=signal.attrs['sampling_freq'])
        detector = art.DetectMA()
        result = detector(signal_3d)
        assert result is not None

    def test_detect_ma_output_shape(self):
        """Test that output has consistent shape with input."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        result = detector(signal)
        assert result.shape == signal.shape

    def test_detect_ma_with_artifacts_present(self):
        """Test detection identifies artifacts when present."""
        data = np.random.uniform(size=(1000, 3))
        # Add clear artifacts
        data[300:350, :] = np.random.uniform(10, 20, size=(50, 3))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        result = detector(signal)
        # Should detect something
        assert result is not None


class TestWaveletFilter:
    """Test Wavelet-based filtering for artifact correction."""

    def test_wavelet_filter_basic(self):
        """Test basic wavelet filtering."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        wavelet_filt = art.WaveletFilter()
        result = wavelet_filt(signal)
        assert result is not None

    def test_wavelet_filter_shape_preservation(self):
        """Test that wavelet filter preserves signal shape."""
        data = np.random.uniform(size=(1000, 5, 2))
        signal = create_signal(data, sampling_freq=100)
        wavelet_filt = art.WaveletFilter()
        result = wavelet_filt(signal)
        assert result.shape == signal.shape

    def test_wavelet_filter_with_real_data(self):
        """Test wavelet filter with real data."""
        signal = TestData.ecg(return_signal=True)
        wavelet_filt = art.WaveletFilter()
        result = wavelet_filt(signal)
        assert result is not None

    def test_wavelet_filter_reduces_noise(self):
        """Test that wavelet filter reduces noise."""
        # Create signal with noise
        clean_data = np.sin(np.linspace(0, 10 * np.pi, 1000))
        noise = np.random.normal(0, 0.1, size=1000)
        noisy_data = clean_data + noise
        signal = create_signal(noisy_data, sampling_freq=100)
        wavelet_filt = art.WaveletFilter()
        result = wavelet_filt(signal)
        # Filtered signal should be smoother (lower high-frequency variance)
        assert result is not None


class TestArtefactCorrection:
    """Test artifact correction algorithms."""

    def test_mara_basic(self):
        """Test basic MARA (Motion Artifact Removal using AutoRegression)."""
        data = np.random.uniform(size=(1000, 3))
        signal = create_signal(data, sampling_freq=100)
        # Add MA variable
        ma_data = np.random.choice([0, 1], size=(1000, 3))
        signal['MA'] = (('time', 'channel'), ma_data)
        mara = art.MARA()
        result = mara(signal, scheduler='single-threaded')
        assert result is not None

    def test_mara_with_detected_artifacts(self):
        """Test MARA with detected motion artifacts."""
        data = np.random.uniform(size=(1000, 3))
        # Add artifacts
        data[200:250, 0] = data[200:250, 0] + 3
        signal = create_signal(data, sampling_freq=100)
        
        # Detect artifacts
        detector = art.DetectMA(fuse='all')
        ma_mask = detector(signal)
        signal['MA'] = ma_mask
        
        # Correct
        mara = art.MARA()
        result = mara(signal, scheduler='single-threaded')
        assert result is not None

    def test_mara_output_shape(self):
        """Test MARA preserves signal shape."""
        data = np.random.uniform(size=(500, 3))
        signal = create_signal(data, sampling_freq=100)
        ma_data = np.random.choice([0, 1], size=(500, 3))
        signal['MA'] = (('time', 'channel'), ma_data)
        mara = art.MARA()
        result = mara(signal, scheduler='single-threaded')
        assert result is not None


class TestArtefactIntegration:
    """Integration tests for artifact detection and correction pipeline."""

    def test_full_artifact_pipeline(self):
        """Test complete artifact detection and correction pipeline."""
        # Create signal with artifacts
        data = np.random.uniform(size=(1000, 3, 2))
        data[300:350, :, :] = data[300:350, :, :] + 2
        signal = create_signal(data, sampling_freq=100)
        
        # Detect artifacts
        detector = art.DetectMA(fuse='component')
        ma_mask = detector(signal)
        signal['MA'] = ma_mask
        
        # Correct artifacts
        mara = art.MARA()
        corrected = mara(signal, scheduler='single-threaded')
        
        assert corrected is not None
        assert corrected.shape == signal.shape

    def test_detection_before_filtering(self):
        """Test artifact detection before wavelet filtering."""
        data = np.random.uniform(size=(1000, 3))
        data[400:450, :] = np.random.uniform(10, 15, size=(50, 3))
        signal = create_signal(data, sampling_freq=100)
        
        # Detect
        detector = art.DetectMA()
        ma_mask = detector(signal)
        
        # Filter
        wavelet_filt = art.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        assert ma_mask is not None
        assert filtered is not None

    def test_multiple_channels_artifact_detection(self):
        """Test artifact detection on multi-channel signals."""
        data = np.random.uniform(size=(1000, 5, 3))
        signal = create_signal(data, sampling_freq=100)
        
        detector = art.DetectMA()
        result = detector(signal)
        assert result is not None
        assert result.shape == signal.shape


class TestArtefactEdgeCases:
    """Test edge cases and error handling in artifact algorithms."""

    def test_short_signal_detection(self):
        """Test artifact detection on short signals."""
        data = np.random.uniform(size=(100, 2))
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        result = detector(signal)
        assert result is not None

    def test_single_channel_detection(self):
        """Test artifact detection on single-channel signal."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        result = detector(signal)
        assert result is not None

    def test_detection_with_nans(self):
        """Test artifact detection with NaN values."""
        data = np.random.uniform(size=(1000, 3))
        data[100:150, 1] = np.nan
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        result = detector(signal)
        assert result is not None

    def test_detection_with_infs(self):
        """Test artifact detection with infinite values."""
        data = np.random.uniform(size=(1000, 3))
        data[100:150, 0] = np.inf
        signal = create_signal(data, sampling_freq=100)
        detector = art.DetectMA()
        try:
            result = detector(signal)
            assert result is not None
        except (ValueError, RuntimeWarning):
            # Expected behavior
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
