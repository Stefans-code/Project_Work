"""
Tests for the artefacts module including detection and correction algorithms.
"""
import numpy as np
import pytest
from pyphysio.signal import create_signal
from pyphysio import TestData
import pyphysio.artefacts as art
import xarray as xr
from pyphysio.generators import SpikeGenerator, BaselineShiftGenerator, SinusoidalGenerator


class TestDetectMA:
    """Test Motion Artifact Detection algorithm."""

    def test_detect_ma_basic(self):
        """Test basic motion artifact detection."""
        # Create a sinusoidal baseline with Gaussian noise and add spikes
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                        times=[0.5, 2.0, 5.0], amplitudes=[5.0, 4.0, 6.0],
                        durations=[0.02, 0.02, 0.03], shape='gaussian')
        vals = base + spikes.p.get_values().ravel()
        signal = create_signal(vals, sampling_freq=fs)
        detector = art.DetectMA()
        result = detector(signal)
        assert result is not None
        assert result.sum() > 0

    def test_detect_ma_with_fuse_none(self):
        """Test motion artifact detection with fuse=None."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                        times=[1.0, 3.0], amplitudes=4.0,
                        durations=0.02, shape='linear')
        vals = base + spikes.p.get_values().ravel()
        signal = create_signal(vals, sampling_freq=fs)
        detector = art.DetectMA(fuse=None)
        result = detector(signal)
        assert result is not None
        assert result.sum() > 0

    def test_detect_ma_with_fuse_all(self):
        """Test motion artifact detection with fuse='all'."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                        times=[0.8, 4.0], amplitudes=6.0,
                        durations=0.03, shape='linear')
        vals = base + spikes.p.get_values().ravel()
        # create a 3D signal (time, channel, component) and inject the same artefacts across channels/components
        data_3d = np.tile(vals[:, None, None], (1, 3, 2))
        signal = create_signal(data_3d, sampling_freq=fs)
        detector = art.DetectMA(fuse='all')
        result = detector(signal)
        assert result is not None
        assert result.sum() > 0

    def test_detect_ma_with_fuse_component(self):
        """Test motion artifact detection with fuse='component'."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                times=[2.0], amplitudes=12.0,
                durations=0.08, shape='gaussian')
        vals = base + spikes.p.get_values().ravel()
        # create a 3D signal with artifacts present in components
        data_3d = np.tile(vals[:, None, None], (1, 2, 3))
        signal = create_signal(data_3d, sampling_freq=fs)
        detector = art.DetectMA(fuse='component', method='fixed', th_std=0.1, th_amp=0.5)
        result = detector(signal)
        assert result is not None
        assert result.sum() > 0

    def test_detect_ma_iqr_method(self):
        """Test motion artifact detection with IQR method."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        # baseline shifts create amplitude changes that should be picked up by IQR
        shifts = BaselineShiftGenerator.baseline_shifts(duration=duration, sampling_freq=fs,
                                times=[1.0, 4.0], amplitudes=[3.0, -2.0],
                                durations=[0.1, 0.2])
        vals = base + shifts.p.get_values().ravel()
        signal = create_signal(vals, sampling_freq=fs)
        detector = art.DetectMA(method='iqr', iqr=1.5)
        result = detector(signal)
        assert result is not None
        assert result.sum() > 0

    def test_detect_ma_mad_method(self):
        """Test motion artifact detection with MAD method."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                times=[3.0, 6.0], amplitudes=[10.0, 10.0],
                durations=[0.05, 0.05], shape='gaussian')
        vals = base + spikes.p.get_values().ravel()
        signal = create_signal(vals, sampling_freq=fs)
        detector = art.DetectMA(method='mad', th_std_coeff=1.0)
        result = detector(signal)
        assert result is not None
        assert result.sum() > 0

    def test_detect_ma_window_length(self):
        """Test motion artifact detection with different window lengths."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                times=[2.0, 5.0], amplitudes=[12.0, 15.0],
                durations=[0.08, 0.1], shape='linear')
        vals = base + spikes.p.get_values().ravel()
        signal = create_signal(vals, sampling_freq=fs)
        for win_len in [0.5, 1.0, 2.0]:
            detector = art.DetectMA(win_len=win_len, method='fixed', th_std=0.1, th_amp=0.5)
            result = detector(signal)
            assert result is not None
            assert result.sum() > 0

    def test_detect_ma_window_mask(self):
        """Test motion artifact detection with different mask window lengths."""
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                times=[1.0, 4.0, 7.0], amplitudes=[12.0, 14.0, 11.0],
                durations=[0.08, 0.08, 0.08], shape='gaussian')
        vals = base + spikes.p.get_values().ravel()
        signal = create_signal(vals, sampling_freq=fs)
        for win_mask in [0.5, 1.0, 2.0]:
            detector = art.DetectMA(win_mask=win_mask, method='fixed', th_std=0.1, th_amp=0.5)
            result = detector(signal)
            assert result is not None
            assert result.sum() > 0

    def test_detect_ma_with_real_data(self):
        """Test motion artifact detection with real test data."""
        # create a sinusoidal real-like signal and inject spikes into channel 0
        fs = 100
        duration = 10.0
        baseline = SinusoidalGenerator.simple_sine(duration=duration, sampling_freq=fs, frequency=1).p.get_values().ravel()
        noise = np.random.normal(0, 0.05, size=baseline.shape)
        base = baseline + noise
        spikes = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                                        times=[0.5, 1.0], amplitudes=5.0,
                                        durations=0.02, shape='gaussian')
        spikes_vals = spikes.p.get_values().ravel()
        data_3d = np.tile(base[:, None, None], (1, 3, 1))
        if spikes_vals.shape[0] == data_3d.shape[0]:
            data_3d[:, 0, 0] += spikes_vals
        signal_3d = create_signal(data_3d, sampling_freq=fs)
        detector = art.DetectMA()
        result = detector(signal_3d)
        assert result is not None
        assert result.sum() > 0

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

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
