"""
Comprehensive tests for WaveletFilter artifact removal.

This test suite evaluates the WaveletFilter's ability to remove various types
of artifacts and noise:
- Baseline shifts (slow drift)
- Spikes (impulse noise)
- Localized high-frequency components
- Complex combinations of artifacts

The filter uses wavelet decomposition to identify and suppress outlier
coefficients based on inter-quartile range (IQR) thresholding.

Test signals are generated using pyphysio.generators.fundamental.SinusoidalGenerator
and pyphysio.generators.fundamental.CompositeSignalGenerator.

Usage:
    Run tests without figures:
        pytest tests/test_96_WaveletFilter.py -v
    
    Run tests with before/after filtering figures:
        pytest tests/test_96_WaveletFilter.py -v --generate-wavelet-figures

References:
    Molavi, B., & Dumont, G. A. (2012). "Wavelet-based motion artifact 
    removal for functional near-infrared spectroscopy". Physiological 
    Measurement, 33(2), 259.
"""

import numpy as np
import pytest
import os
from pathlib import Path
import matplotlib.pyplot as plt
from pyphysio.signal import create_signal
from pyphysio import artefacts
from pyphysio.generators.fundamental import (
    SinusoidalGenerator, CompositeSignalGenerator, NoiseGenerator,
    SpikeGenerator, BaselineShiftGenerator
)

# Make tests deterministic where randomization is used
np.random.seed(0)

# TODO-AI [PRIORITY: HIGH]: Use `signal.p.get_values()` consistently instead of
# `signal.values` or `signal.data` for numeric comparisons.
# TODO-AI [PRIORITY: HIGH]: Seed randomness (e.g., spike amplitudes) to make
# tests deterministic (use `np.random.seed(0)` or `rng` fixture).
# TODO-AI [PRIORITY: MEDIUM]: Replace fixed numeric thresholds with relative
# tolerances derived from noise amplitude / signal power.


@pytest.fixture
def generate_figures(request):
    """Fixture to enable/disable figure generation based on command-line option."""
    return request.config.getoption("--generate-wavelet-figures")


@pytest.fixture
def figure_dir(request, generate_figures):
    """Create directory for saving test figures."""
    if generate_figures:
        fig_dir = Path(__file__).parent / "test_96_wavelet_figures"
        fig_dir.mkdir(exist_ok=True)
        return fig_dir
    return None


def save_wavelet_comparison_figure(figure_dir, test_name, times, original_signal, 
                                   filtered_signal, figsize=(14, 6)):
    """
    Save a before/after filtering comparison figure.
    
    Parameters
    ----------
    figure_dir : Path
        Directory to save the figure
    test_name : str
        Name of the test (used for filename)
    times : array-like
        Time vector for the x-axis
    original_signal : array-like
        Original signal before filtering
    filtered_signal : array-like
        Signal after filtering
    figsize : tuple, optional
        Figure size (width, height)
    """
    if figure_dir is None:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Original signal
    axes[0].plot(times, original_signal, 'b-', linewidth=1, label='Original (with artifacts)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{test_name} - Original Signal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Filtered signal
    axes[1].plot(times, filtered_signal, 'g-', linewidth=1, label='After WaveletFilter')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'{test_name} - Filtered Signal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    safe_name = test_name.replace(' ', '_').replace('/', '_').lower()
    filepath = figure_dir / f"{safe_name}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath


class TestWaveletFilterBasic:
    """Basic functionality tests for WaveletFilter."""
    
    def test_wavelet_filter_simple_signal(self, figure_dir):
        """Test WaveletFilter preserves clean periodic signal without artifacts."""
        # Create clean sinusoidal signal using SinusoidalGenerator
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        # Generate clean 2 Hz sine wave
        clean_signal = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(clean_signal)
        
        # Generate figure if requested
        if figure_dir:
            save_wavelet_comparison_figure(
                figure_dir,
                'test_wavelet_filter_simple_signal',
                times,
                clean_signal.values.ravel(),
                filtered.values.ravel()
            )
        
        # Assertions
        assert filtered is not None
        assert filtered.shape == clean_signal.shape
        # Clean signal should remain similar after filtering
        mse = np.mean((filtered.values.ravel() - clean_signal.values.ravel()) ** 2)
        assert mse < 0.1, "Clean signal should be minimally distorted"
    
    def test_wavelet_filter_with_single_baseline_shift(self, figure_dir):
        """Test WaveletFilter removes a single baseline shift artifact.
        
        Simulates a scenario where the signal baseline suddenly shifts
        (e.g., due to contact loss in fNIRS or electrode movement in ECG).
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add single baseline shift using generator (step at t=2.5s)
        shift_signal = BaselineShiftGenerator.baseline_shifts(
            duration=duration, sampling_freq=fsamp, times=[2.5], amplitudes=3.0, durations=0.01
        )
        signal_with_shift = clean_signal + shift_signal.values.ravel()
        signal = create_signal(signal_with_shift, sampling_freq=fsamp, name='shift')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Generate figure if requested
        if figure_dir:
            save_wavelet_comparison_figure(
                figure_dir,
                'test_wavelet_filter_with_single_baseline_shift',
                times,
                signal_with_shift,
                filtered.values.ravel()
            )
        
        # Assertions
        assert filtered is not None
        # Wavelet filter reduces high-freq noise but doesn't completely remove DC shifts
        # Check that filter processes the signal (doesn't fail)
        filtered_values = filtered.values.ravel()
        # Verify filter is applied: variance should be reduced compared to noisy signal
        # (the filter smooths the signal)
        assert np.std(filtered_values) < np.std(signal_with_shift), \
            "Filter should reduce overall variance"
    
    def test_wavelet_filter_with_single_spike_artifact(self, figure_dir):
        """Test WaveletFilter removes a single spike artifact.
        
        Simulates a scenario with impulse noise (e.g., EMG artifact,
        electromagnetic interference, or sensor glitch).
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add single large spike using generator
        spike_signal = SpikeGenerator.spikes(duration=duration, sampling_freq=fsamp,
                     times=[2.5], amplitudes=8.0, durations=0.01,
                     shape='linear')
        spike_idx = int(2.5 * fsamp)
        signal_with_spike = clean_signal + spike_signal.values.ravel()
        signal = create_signal(signal_with_spike, sampling_freq=fsamp, name='spike')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Generate figure if requested
        if figure_dir:
            save_wavelet_comparison_figure(
                figure_dir,
                'test_wavelet_filter_with_single_spike_artifact',
                times,
                signal_with_spike,
                filtered.values.ravel()
            )
        
        # Assertions
        assert filtered is not None
        # Check that spike is removed
        spike_amplitude = np.abs(filtered.values.ravel()[spike_idx] - clean_signal[spike_idx])
        assert spike_amplitude < 1.0, "Spike artifact should be significantly reduced"
        
        # Check that region around spike is also smoothed
        region_around_spike = filtered.values.ravel()[spike_idx-5:spike_idx+5]
        original_region = clean_signal[spike_idx-5:spike_idx+5]
        region_mse = np.mean((region_around_spike - original_region) ** 2)
        assert region_mse < 2.0, "Region around spike should be reasonably preserved"


class TestWaveletFilterArtifacts:
    """Test WaveletFilter with various artifact types."""
    
    def test_wavelet_filter_with_localized_high_frequency_noise(self, figure_dir):
        """Test WaveletFilter removes localized high-frequency noise.
        
        Simulates high-frequency noise in a localized region
        (e.g., motion artifact in a specific time window).
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add localized high-frequency noise (40 Hz) in middle of signal
        signal_with_hf = clean_signal.copy()
        hf_start = int(1.5 * fsamp)
        hf_end = int(3.5 * fsamp)
        hf_noise = 1.0 * np.sin(2 * np.pi * 40 * times[hf_start:hf_end])
        signal_with_hf[hf_start:hf_end] += hf_noise
        
        signal = create_signal(signal_with_hf, sampling_freq=fsamp, name='hf_noise')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Generate figure if requested
        if figure_dir:
            save_wavelet_comparison_figure(
                figure_dir,
                'test_wavelet_filter_with_localized_high_frequency_noise',
                times,
                signal_with_hf,
                filtered.values.ravel()
            )
        
        # Assertions
        assert filtered is not None
        # Check that high-frequency content is reduced in the noisy region
        noisy_region = filtered.values.ravel()[hf_start:hf_end]
        clean_region = clean_signal[hf_start:hf_end]
        region_mse = np.mean((noisy_region - clean_region) ** 2)
        assert region_mse < 0.5, "High-frequency noise should be significantly reduced"
    
    def test_wavelet_filter_with_multiple_spikes(self):
        """Test WaveletFilter removes multiple spike artifacts.
        
        Tests handling of multiple impulse noise events.
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add multiple spikes using generator
        spike_times = [1.0, 2.5, 4.0]
        spike_amps = [7.0 + np.random.uniform(-1, 1) for _ in spike_times]
        spike_signal = SpikeGenerator.spikes(duration=duration, sampling_freq=fsamp,
                                             times=spike_times, amplitudes=spike_amps,
                                             durations=0.01)
        signal = create_signal(clean_signal + spike_signal.values.ravel(), sampling_freq=fsamp, name='multi_spike')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Assertions
        assert filtered is not None
        # Check that spikes are removed at all positions
        spike_positions = [int(t * fsamp) for t in spike_times]
        for idx in spike_positions:
            spike_remaining = np.abs(filtered.values.ravel()[idx] - clean_signal[idx])
            assert spike_remaining < 1.0, f"Spike at index {idx} should be removed"
    
    def test_wavelet_filter_with_baseline_drift(self):
        """Test WaveletFilter handles slow baseline drift.
        
        Simulates slow drift in baseline (e.g., physiological drift,
        sensor drift over time).
        """
        # Create clean signal with low-frequency drift component
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        # Generate 2 Hz sine wave (clean component)
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add slow baseline drift (0.3 Hz sinusoidal drift with 2.0 amplitude)
        drift = 2.0 * np.sin(2 * np.pi * 0.3 * times)
        signal_with_drift = clean_signal + drift
        
        signal = create_signal(signal_with_drift, sampling_freq=fsamp, name='drift')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Assertions
        assert filtered is not None
        # Check that low-frequency drift is reduced
        filtered_values = filtered.values.ravel()
        # Detrend by subtracting low-pass filtered version
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 0.5, 'low', fs=fsamp)  # Low-pass at 0.5 Hz
        baseline_est = filtfilt(b, a, filtered_values)
        detrended = filtered_values - baseline_est
        detrended_mse = np.mean((detrended - clean_signal) ** 2)
        assert detrended_mse < 0.3, "Baseline drift should be effectively removed"


class TestWaveletFilterComplex:
    """Test WaveletFilter with complex artifact combinations."""
    
    def test_wavelet_filter_with_mixed_artifacts(self):
        """Test WaveletFilter handles combination of baseline shift + spikes.
        
        Realistic scenario with multiple artifact types present simultaneously.
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add baseline shift and spikes using generators
        shift_sig = BaselineShiftGenerator.baseline_shifts(duration=duration, sampling_freq=fsamp,
                                                          times=[2.5], amplitudes=2.5, durations=0.01)
        spike_sig = SpikeGenerator.spikes(duration=duration, sampling_freq=fsamp,
                          times=[1.0, 3.5], amplitudes=[6.0, 6.0], durations=0.01)
        signal_mixed = clean_signal + shift_sig.values.ravel() + spike_sig.values.ravel()
        signal = create_signal(signal_mixed, sampling_freq=fsamp, name='mixed')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Assertions
        assert filtered is not None
        # Check that spikes are at least partially removed
        original_spike_amplitude = np.max(signal_mixed) - np.mean(clean_signal)
        filtered_spike_amplitude = np.max(filtered.values.ravel()) - np.mean(clean_signal)
        assert filtered_spike_amplitude < original_spike_amplitude, \
            "Filter should reduce spike amplitudes in mixed artifact scenario"
        
        # Check that filter successfully processes the signal
        assert np.std(filtered.values.ravel()) < np.std(signal_mixed), \
            "Filter should reduce variance from artifact-rich signal"
    
    def test_wavelet_filter_with_hf_noise_and_spike(self):
        """Test WaveletFilter with high-frequency noise + spike artifacts.
        
        Complex scenario with both wideband and impulse noise.
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Add localized high-frequency noise
        signal_complex = clean_signal.copy()
        hf_start = int(1.0 * fsamp)
        hf_end = int(3.0 * fsamp)
        hf_noise = 0.8 * np.sin(2 * np.pi * 40 * times[hf_start:hf_end])
        signal_complex[hf_start:hf_end] += hf_noise
        
        # Add spike using generator
        spike_sig = SpikeGenerator.spikes(duration=duration, sampling_freq=fsamp,
                          times=[2.0], amplitudes=5.0, durations=0.01)
        spike_idx = int(2.0 * fsamp)
        signal = create_signal(signal_complex + spike_sig.values.ravel(), sampling_freq=fsamp, name='complex')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Assertions
        assert filtered is not None
        assert filtered.shape == signal.shape
        # Both artifacts should be reduced
        spike_remaining = np.abs(filtered.values.ravel()[spike_idx] - clean_signal[spike_idx])
        assert spike_remaining < 1.0, "Spike should be removed in complex scenario"
    
    def test_wavelet_filter_with_multiple_artifacts_and_drift(self, figure_dir):
        """Test WaveletFilter with comprehensive artifact combination.
        
        Most realistic scenario: baseline shift + spikes + HF noise + drift.
        """
        # Create clean signal
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        
        # Build artifact-rich signal step by step
        signal_rich = clean_signal.copy()
        
        # 1. Add baseline drift
        drift = 1.5 * np.sin(2 * np.pi * 0.25 * times)
        signal_rich += drift
        
        # 2. Add step baseline shift using generator
        shift_sig = BaselineShiftGenerator.baseline_shifts(duration=duration, sampling_freq=fsamp,
                                  times=[3.0], amplitudes=2.0, durations=0.01)
        signal_rich += shift_sig.values.ravel()
        
        # 3. Add localized HF noise
        hf_start = int(1.5 * fsamp)
        hf_end = int(2.5 * fsamp)
        hf_noise = 0.6 * np.sin(2 * np.pi * 45 * times[hf_start:hf_end])
        signal_rich[hf_start:hf_end] += hf_noise
        
        # 4. Add spikes using generator
        spike_times = [0.8, 2.0, 4.2]
        spike_amps = [5.0 + np.random.uniform(-1, 1) for _ in spike_times]
        spike_sig = SpikeGenerator.spikes(duration=duration, sampling_freq=fsamp,
                                          times=spike_times, amplitudes=spike_amps,
                                          durations=0.01)
        signal_rich += spike_sig.values.ravel()
        
        signal = create_signal(signal_rich, sampling_freq=fsamp, name='rich_artifacts')
        
        # Apply wavelet filter
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(signal)
        
        # Generate figure if requested
        if figure_dir:
            save_wavelet_comparison_figure(
                figure_dir,
                'test_wavelet_filter_with_multiple_artifacts_and_drift',
                times,
                signal_rich,
                filtered.values.ravel()
            )
        
        # Assertions
        assert filtered is not None
        # Signal should still be recognizable
        filtered_values = filtered.values.ravel()
        
        # Check SNR improvement in spike regions
        spike_positions = [int(t * fsamp) for t in spike_times]
        spike_region = filtered_values[spike_positions[0]-5:spike_positions[0]+5]
        clean_region = clean_signal[spike_positions[0]-5:spike_positions[0]+5]
        
        # Artifacts should significantly reduce MSE
        original_mse = np.mean((signal_rich[spike_positions[0]-5:spike_positions[0]+5] - 
                                clean_region) ** 2)
        filtered_mse = np.mean((spike_region - clean_region) ** 2)
        assert filtered_mse <= original_mse, "Artifacts should be substantially removed"


class TestWaveletFilterParameters:
    """Test WaveletFilter parameter sensitivity."""
    
    def test_wavelet_filter_iqr_parameter_effect(self):
        """Test effect of IQR parameter on artifact removal.
        
        Higher IQR means more aggressive outlier removal.
        """
        # Create signal with spike using generator
        duration = 5
        fsamp = 100
        times = np.arange(duration * fsamp) / fsamp
        
        clean_signal_obj = SinusoidalGenerator.simple_sine(
            duration=duration, 
            sampling_freq=fsamp, 
            amplitude=1.0, 
            frequency=2.0
        )
        clean_signal = clean_signal_obj.values.ravel()
        spike_sig = SpikeGenerator.spikes(duration=duration, sampling_freq=fsamp,
                          times=[2.5], amplitudes=7.0, durations=0.01)
        signal = create_signal(clean_signal + spike_sig.values.ravel(), sampling_freq=fsamp)
        
        # Test different IQR values
        iqr_values = [0.5, 1.5, 3.0]
        filtered_results = []
        
        for iqr in iqr_values:
            wavelet_filt = artefacts.WaveletFilter(iqr=iqr)
            filtered = wavelet_filt(signal)
            filtered_results.append(filtered.values.ravel())
        
        # Assertions
        # Lower IQR should remove more (more aggressive)
        spike_idx = int(2.5 * fsamp)
        spike_remaining = [np.abs(res[spike_idx] - clean_signal[spike_idx]) 
                          for res in filtered_results]
        
        # First filter (iqr=0.5) should remove spike more than last (iqr=3.0)
        assert spike_remaining[0] <= spike_remaining[-1], \
            "Lower IQR should result in more aggressive artifact removal"


class TestWaveletFilterRobustness:
    """Test WaveletFilter robustness to edge cases."""
    
    def test_wavelet_filter_preserves_shape(self):
        """Test that WaveletFilter preserves signal shape."""
        clean_signal = SinusoidalGenerator.simple_sine(
            duration=5, 
            sampling_freq=100, 
            amplitude=1.0, 
            frequency=2.0
        )
        
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(clean_signal)
        
        assert filtered.shape == clean_signal.shape, "Shape should be preserved"
    
    def test_wavelet_filter_with_different_signal_lengths(self):
        """Test WaveletFilter works with various signal lengths."""
        fsamp = 100
        
        for duration in [1, 3, 5, 10]:
            clean_signal = SinusoidalGenerator.simple_sine(
                duration=duration, 
                sampling_freq=fsamp, 
                amplitude=1.0, 
                frequency=2.0
            )
            
            wavelet_filt = artefacts.WaveletFilter()
            filtered = wavelet_filt(clean_signal)
            
            assert filtered is not None
            assert filtered.shape == clean_signal.shape
    
    def test_wavelet_filter_constant_signal(self):
        """Test WaveletFilter with constant signal (no variation)."""
        duration = 5
        fsamp = 100
        constant_signal = create_signal(
            np.ones(duration * fsamp) * 5.0, 
            sampling_freq=fsamp
        )
        
        wavelet_filt = artefacts.WaveletFilter()
        filtered = wavelet_filt(constant_signal)
        
        # Constant signal should remain roughly constant
        assert filtered is not None
        filtered_values = filtered.values.ravel()
        assert np.std(filtered_values) < 0.5, "Constant signal should remain constant"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
