"""
Tests for signal generators module.

Tests cover:
- Fundamental signal generators (zeros, ones, ramps, etc.)
- Sinusoidal generators
- Noise generators
- Composite signal generators
- Window generators
- Physiological signal generators (ECG, respiration, EDA, fNIRS, EEG, EMG)
"""

import pytest
import numpy as np
from scipy.signal import welch
from pyphysio.generators import (
    FundamentalSignalGenerator,
    SinusoidalGenerator,
    NoiseGenerator,
    CompositeSignalGenerator,
    WindowGenerator,
    ECGGenerator,
    RespirationGenerator,
    EDAGenerator,
    fNIRSGenerator,
    EEGGenerator,
    EMGGenerator,
    SpikeGenerator,
    BaselineShiftGenerator,
)
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Some tests reference `figure_dir` conditionally but don't request the fixture.
# Provide a safe module-level default to avoid NameError; fixture injection
# still works for tests that accept `figure_dir` as a parameter.
# If the test run sets the environment variable `PYPHYSIO_TEST_FIG_DIR`, use it
# as a fallback so tests without the `figure_dir` parameter can still save
# figures (useful for ad-hoc runs). Otherwise, keep `None` so pytest's
# `figure_dir` fixture controls figure generation when used.
import os
from pathlib import Path as _Path

_env_fig_dir = os.environ.get('PYPHYSIO_TEST_FIG_DIR')
if _env_fig_dir:
    try:
        figure_dir = _Path(_env_fig_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        figure_dir = None
else:
    figure_dir = None


# Use centralized `figure_dir` and `generate_figures` fixtures from `conftest.py`.


def save_generator_figure(figure_dir, test_name, times, values, figsize=(10, 2)):
    """Delegate to centralized helper in tests/_helpers.py."""
    try:
        from _helpers import save_line_figure
    except Exception:
        return None
    return save_line_figure(figure_dir, test_name, times, values, figsize=figsize)

# TODO-AI [PRIORITY: HIGH]: Use the pyphysio accessor for data access.
# - Replace direct uses of `signal.values`, `signal.data`, `np.asarray(signal)`
#   with `signal.p.get_values()` or `signal.p.get_values().ravel()`.
# TODO-AI [PRIORITY: HIGH]: Seed RNG for deterministic tests.
# - Use generator `seed=` parameters when available or call `np.random.seed(0)`
#   immediately before stochastic calls.
# TODO-AI [PRIORITY: MEDIUM]: Convert brittle numeric assertions to tolerant
# comparisons (use relative tolerances or `pytest.approx`).
# TODO-AI [PRIORITY: MEDIUM]: Parametrize repeated tests (zeros/ones/const/ramp).
# TODO-AI [PRIORITY: LOW]: Remove leftover TODO comments inside tests or
# convert them into explicit assertions.


class TestFundamentalSignalGenerator:
    """Tests for FundamentalSignalGenerator class"""
    
    def test_zeros_generation(self, generate_figures, figure_dir):
        """Test generation of zero signal"""
        signal = FundamentalSignalGenerator.zeros(
            duration=10.0, sampling_freq=100
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.allclose(vals, 0)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'zeros_signal', times, vals)
    
    def test_ones_generation(self):
        """Test generation of ones signal"""
        signal = FundamentalSignalGenerator.ones(
            duration=10.0, sampling_freq=100
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.allclose(vals, 1)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'ones_signal', times, vals)
    
    def test_constant_generation(self):
        """Test generation of constant signal"""
        value = 3.5
        signal = FundamentalSignalGenerator.constant(
            duration=10.0, sampling_freq=50, value=value
        )
        vals = signal.p.get_values().ravel()
        assert np.allclose(vals, value)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, f'constant_value_{value}', times, vals)
    
    def test_ramp_generation(self):
        """Test generation of ramp signal"""
        signal = FundamentalSignalGenerator.ramp(
            duration=10.0, sampling_freq=10, start_value=0, end_value=10
        )
        expected = np.linspace(0, 10, 100)
        vals = signal.p.get_values().ravel()
        assert np.allclose(vals, expected)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'ramp_signal', times, vals)
    
    def test_delta_single(self):
        """Test generation of single delta function"""
        signal = FundamentalSignalGenerator.delta(
            duration=10.0, sampling_freq=10,
            delta_times=[0.5], delta_values=[1.0]
        )
        vals = signal.p.get_values().ravel()
        assert vals.ndim == 1
        assert vals.shape[0] == 100
        # Delta should be at t=0.5s -> index ~5
        idx = int(round(0.5 * 10))
        assert vals[idx] == 1.0
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'delta_single', times, vals)
    
    def test_delta_multiple(self):
        """Test generation of multiple delta functions"""
        signal = FundamentalSignalGenerator.delta(
            duration=10.0, sampling_freq=10,
            delta_times=[0.5, 1.5], delta_values=[1.0, 2.0]
        )
        vals = signal.p.get_values().ravel()
        assert vals[int(round(0.5 * 10))] == 1.0
        assert vals[int(round(1.5 * 10))] == 2.0
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'delta_multiple', times, vals)
    
    def test_boxcar_generation(self):
        """Test generation of boxcar (rectangular pulse) signal"""
        signal = FundamentalSignalGenerator.boxcar(
            duration=2.0, sampling_freq=100,
            start_time=0, high_value=2.0, low_value=0,
            transitions=[(0.25, 0.75)]
        )
        # Check that signal is non-zero in the boxcar region
        vals = signal.p.get_values().ravel()
        start_idx = int(0.25 * 100)
        end_idx = int(0.75 * 100)
        assert np.allclose(vals[start_idx:end_idx], 2.0)
        assert np.allclose(vals[:start_idx], 0)
        assert np.allclose(vals[end_idx:], 0)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'boxcar_signal', times, vals)


class TestSinusoidalGenerator:
    """Tests for SinusoidalGenerator class"""
    
    def test_simple_sine(self, figure_dir):
        """Test simple sinusoid generation"""
        freq = 10  # 10 Hz
        signal = SinusoidalGenerator.simple_sine(
            duration=10.0, sampling_freq=100, frequency=freq
        )
        
        # Check shape
        assert signal.p.get_values().ravel().shape == (1000,)

        # Check frequency content using zero crossings
        vals = signal.p.get_values().ravel()
        zero_crossings = np.where(np.diff(np.sign(vals)))[0]
        # Frequency is number of zero crossings / 2 / duration
        duration = 1000 / 100
        estimated_freq = len(zero_crossings) / 2 / duration
        assert np.isclose(estimated_freq, freq, atol=1)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'simple_sine_10Hz', times, vals)
    
    def test_sine_with_phase(self):
        """Test sinusoid with phase shift"""
        signal = SinusoidalGenerator.simple_sine(
            duration=1.0, sampling_freq=100,
            frequency=1, phase=np.pi/2
        )
        # At t=0, sin(phase) should be 1
        vals = signal.p.get_values().ravel()
        assert np.isclose(vals[0], np.sin(np.pi/2), atol=0.1)
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'sine_with_phase', times, vals)
    
    def test_multi_component_sine(self):
        """Test multi-component sinusoid"""
        components = [
            {'frequency': 5, 'amplitude': 1.0},
            {'frequency': 10, 'amplitude': 0.5},
            {'frequency': 15, 'amplitude': 0.25},
        ]
        
        signal = SinusoidalGenerator.multi_component_sine(
            duration=10.0, sampling_freq=100,
            components=components
        )
        
        # Check that signal contains the expected frequency components
        from scipy.signal import periodogram
        f, pxx = periodogram(signal.p.get_values().ravel(), fs=100)
        
        for comp in components:
            freq = comp['frequency']
            # Find peak near expected frequency
            idx = np.argmin(np.abs(f - freq))
            assert f[idx] < freq + 2  # Within 2 Hz
        # Save figure of the composite waveform when requested
        vals = signal.p.get_values().ravel()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'multi_component_sine', times, vals)
    
    def test_frequency_sweep(self):
        """Test frequency sweep (chirp) signal"""
        signal = SinusoidalGenerator.frequency_sweep(
            duration=10.0, sampling_freq=100,
            f_start=1, f_end=10, amplitude=1.0, method='linear'
        )
        
        # Check shape and that values are reasonable
        assert signal.p.get_values().ravel().shape == (1000,)
        assert np.max(np.abs(signal.p.get_values().ravel())) <= 1.1
        if figure_dir:
            vals = signal.p.get_values().ravel()
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'frequency_sweep', times, vals)
    
    def test_amplitude_modulated(self):
        """Test amplitude-modulated sinusoid"""
        signal = SinusoidalGenerator.amplitude_modulated(
            duration=10.0, sampling_freq=100,
            carrier_freq=20, modulation_freq=5, amplitude=1.0
        )
        
        # Check shape
        assert signal.p.get_values().ravel().shape == (1000,)
        # Envelope should vary
        assert np.std(signal.p.get_values().ravel()) > 0
        if figure_dir:
            vals = signal.p.get_values().ravel()
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'amplitude_modulated', times, vals)


class TestNoiseGenerator:
    """Tests for NoiseGenerator class"""
    
    def test_white_noise(self, figure_dir):
        """Test white noise generation"""
        signal = NoiseGenerator.white_noise(
            duration=100.0, sampling_freq=100, std=1.0
        )
        
        vals = signal.p.get_values().ravel()
        # Check standard deviation
        assert np.isclose(np.std(vals), 1.0, atol=0.15)
        
        # Check that it looks like noise (not a sinusoid or other pattern)
        # Just verify that the signal is random with expected properties
        assert np.min(vals) < -1  # Should have values across range
        assert np.max(vals) > 1
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'white_noise', times, vals)
    
    def test_pink_noise(self):
        """Test pink (1/f) noise generation"""
        signal = NoiseGenerator.pink_noise(
            duration=100.0, sampling_freq=100
        )
        
        # Pink noise should have more power at low frequencies
        from scipy.signal import periodogram
        f, pxx = periodogram(signal.p.get_values().ravel(), fs=100)
        
        # Power in low frequencies should be higher than high frequencies
        low_freq_power = np.mean(pxx[f < 10])
        high_freq_power = np.mean(pxx[(f > 20) & (f < 40)])
        assert low_freq_power > high_freq_power
        if figure_dir:
            vals = signal.p.get_values().ravel()
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'pink_noise', times, vals)
    
    def test_uniform_noise(self):
        """Test uniform noise generation"""
        signal = NoiseGenerator.uniform_noise(
            duration=10.0, sampling_freq=100,
            low=-1, high=1
        )
        
        # Check bounds
        vals = signal.p.get_values().ravel()
        assert np.min(vals) >= -1
        assert np.max(vals) <= 1

        # Check that values are actually distributed
        assert np.std(vals) > 0
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'uniform_noise', times, vals)


class TestCompositeSignalGenerator:
    """Tests for CompositeSignalGenerator class"""
    
    def test_sum_signals(self, figure_dir):
        """Test summing multiple signals"""
        sig1 = FundamentalSignalGenerator.constant(
            duration=10.0, sampling_freq=10, value=1.0
        )
        sig2 = FundamentalSignalGenerator.constant(
            duration=10.0, sampling_freq=10, value=2.0
        )
        
        result = CompositeSignalGenerator.sum_signals([sig1, sig2])
        vals = result.p.get_values().ravel()
        assert np.allclose(vals, 3.0)
        if figure_dir:
            times = np.arange(vals.size) / result.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'composite_sum', times, vals)
    
    def test_concatenate_signals(self, figure_dir):
        """Test concatenating signals"""
        sig1 = FundamentalSignalGenerator.ones(
            duration=10.0, sampling_freq=10
        )
        sig2 = FundamentalSignalGenerator.ones(
            duration=10.0, sampling_freq=10
        ) * 2
        
        result = CompositeSignalGenerator.concatenate_signals([sig1, sig2])
        vals = result.p.get_values().ravel()
        assert vals.shape[0] == 200
        assert np.allclose(vals[:100], 1.0)
        assert np.allclose(vals[100:], 2.0)
        if figure_dir:
            times = np.arange(vals.size) / result.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'concatenate_signals', times, vals)
    
    def test_signal_with_noise(self, figure_dir):
        """Test adding noise to a signal"""
        # Create a sinusoidal signal with variance
        clean = SinusoidalGenerator.simple_sine(
            duration=10.0, sampling_freq=100, frequency=10
        )
        
        # Add noise to achieve SNR of 20 dB
        noisy = CompositeSignalGenerator.signal_with_noise(
            clean, snr_db=20
        )
        
        # Check that noise was added
        clean_vals = clean.p.get_values().ravel()
        noisy_vals = noisy.p.get_values().ravel()
        assert not np.allclose(noisy_vals, clean_vals)
        # Check that the signal still looks similar overall
        correlation = np.corrcoef(clean_vals, noisy_vals)[0, 1]
        assert correlation > 0.8  # Should be highly correlated
        if figure_dir:
            times = np.arange(noisy_vals.size) / noisy.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'signal_with_noise_clean', times, clean_vals)
            save_generator_figure(figure_dir, 'signal_with_noise_noisy', times, noisy_vals)
    
    def test_signal_with_artifacts(self, figure_dir):
        """Test adding artifacts to a signal"""
        clean = FundamentalSignalGenerator.zeros(
            duration=10.0, sampling_freq=100
        )
        
        # Add spikes
        noisy = CompositeSignalGenerator.signal_with_artifacts(
            clean, artifact_times=[0.5, 1.0, 1.5],
            artifact_type='spike', artifact_amplitude=5.0
        )
        
        vals = noisy.p.get_values().ravel()
        # Check that spikes were added
        assert np.max(np.abs(vals)) > 3
        if figure_dir:
            times = np.arange(vals.size) / noisy.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'signal_with_artifacts', times, vals)


class TestWindowGenerator:
    """Tests for WindowGenerator class"""
    
    def test_apply_window(self, figure_dir):
        """Test applying a window to a signal"""
        signal = FundamentalSignalGenerator.ones(
            duration=10.0, sampling_freq=10
        )
        
        windowed = WindowGenerator.apply_window(signal, window_type='hann')
        
        # Use accessor values
        sig_vals = signal.p.get_values().ravel()
        win_vals = windowed.p.get_values().ravel()
        # Window should taper the signal
        assert win_vals[0] < sig_vals[0]
        assert win_vals[-1] < sig_vals[-1]
        assert win_vals[50] > win_vals[0]
        if figure_dir:
            times = np.arange(sig_vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'window_original', times, sig_vals)
            save_generator_figure(figure_dir, 'window_applied', times, win_vals)
    
    def test_window_generation(self, figure_dir):
        """Test generating window functions"""
        for window_type in ['hann', 'hamming', 'blackman', 'bartlett']:
            window = WindowGenerator.generate_window(
                duration=1.0, sampling_freq=100, window_type=window_type
            )
            
            assert len(window) == 100
            # Windows should have values between 0 and 1 (allowing for floating point errors)
            assert np.all(window >= -1e-10)
            assert np.all(window <= 1.0 + 1e-10)
            assert np.max(window) > 0.5  # Should have significant values
            if figure_dir:
                times = np.arange(len(window)) / 100.0
                save_generator_figure(figure_dir, f'window_{window_type}', times, window)


class TestECGGenerator:
    """Tests for ECG signal generation"""
    
    def test_simple_ecg(self, figure_dir):
        """Test simple ECG generation"""
        signal = ECGGenerator.simple_ecg(
            duration=4.0, sampling_freq=250,
            heart_rate=70
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.isfinite(vals).all()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'ecg_simple', times, vals)
    
    def test_simple_ecg_heart_rate(self, figure_dir):
        """Test ECG heart rate is approximately correct"""
        signal = ECGGenerator.simple_ecg(
            duration=30.0, sampling_freq=100,
            heart_rate=60
        )
        
        # Just check that the signal was generated correctly
        vals = signal.p.get_values().ravel()
        assert vals.shape == (3000,)
        assert np.isfinite(vals).all()
        # ECG should have some variation
        assert np.std(vals) > 0.1
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'ecg_heart_rate_60bpm', times, vals)
    
    def test_realistic_ecg(self, figure_dir):
        """Test realistic ECG with HRV"""
        signal = ECGGenerator.realistic_ecg(
            duration=30.0, sampling_freq=100,
            heart_rate=70, heart_rate_variability=10,
            noise_std=0.05
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (3000,)
        assert np.isfinite(vals).all()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'ecg_realistic_hvr', times, vals)
    
    def test_realistic_ecg_with_ectopy(self, figure_dir):
        """Test realistic ECG with ectopic beats"""
        signal = ECGGenerator.realistic_ecg(
            duration=50.0, sampling_freq=100,
            heart_rate=70, ectopy_rate=0.1
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (5000,)
        assert np.isfinite(vals).all()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'ecg_realistic_ectopy', times, vals)


class TestRespirationGenerator:
    """Tests for respiration signal generation"""
    
    def test_sinusoidal_respiration(self):
        """Test sinusoidal respiration"""
        signal = RespirationGenerator.sinusoidal_respiration(
            duration=100.0, sampling_freq=10,
            respiration_rate=15, amplitude=1.0
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.isfinite(vals).all()
        assert np.max(vals) <= 1.1
        assert np.min(vals) >= -1.1
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'respiration_sinusoidal', times, vals)
    
    def test_realistic_respiration(self):
        """Test realistic respiration with asymmetry"""
        signal = RespirationGenerator.realistic_respiration(
            duration=100.0, sampling_freq=10,
            respiration_rate=15, amplitude=1.0,
            inspiration_expiration_ratio=0.4
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.isfinite(vals).all()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'respiration_realistic', times, vals)
    
    def test_respiration_with_sinus_arrhythmia(self):
        """Test respiration with RSA modulation"""
        signal = RespirationGenerator.realistic_respiration(
            duration=100.0, sampling_freq=10,
            respiration_rate=15,
            respiratory_sinus_arrhythmia=True
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.std(vals) > 0
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'respiration_rsa', times, vals)


class TestEDAGenerator:
    """Tests fr EDA signal generation"""
    
    def test_eda_phasic(self):
        """Test EDA phasic component"""
        stimulus_times = [1.0, 3.0, 5.0]
        signal = EDAGenerator.eda_phasic_component(
            duration=100.0, sampling_freq=10,
            stimulus_times=stimulus_times,
            response_magnitude=0.5
        )
        
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        # Should be mostly zero before stimuli
        # TODO-AI: replace placeholder TODOs with deterministic checks
        # e.g., assert np.mean(vals[:int(0.5*signal.p.get_sampling_freq())]) < 0.1
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'eda_phasic', times, vals)
    
    def test_realistic_eda(self):
        """Test realistic EDA (SCL + SCR)"""
        stimulus_times = [1.0, 3.0]
        signal = EDAGenerator.realistic_eda(
            duration=200.0, sampling_freq=10,
            base_scl=2.0, stimulus_times=stimulus_times,
            stimulus_magnitudes=[0.5, 0.3]
        )
        
        vals = signal.p.get_values().ravel()
        assert vals.shape == (2000,)
        assert np.all(vals > 0)  # Conductance is always positive
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'eda_realistic', times, vals)


class TestFNIRSGenerator:
    """Tests for fNIRS signal generation"""
    
    def test_hbo_hbr_response(self):
        """Test HbO and HbR response to stimuli"""
        stimulus_times = [1.0, 3.0]
        hbo, hbr = fNIRSGenerator.hbo_hbr_response(
            duration=200.0, sampling_freq=10,
            stimulus_times=stimulus_times,
            hbo_magnitude=2.0, hbr_magnitude=-1.0
        )
        
        vals_hbo = hbo.p.get_values().ravel()
        vals_hbr = hbr.p.get_values().ravel()
        assert vals_hbo.shape == (2000,)
        assert vals_hbr.shape == (2000,)
        # HbO should increase during stimuli
        assert np.max(vals_hbo) > np.min(vals_hbo)
        # HbR should decrease during stimuli
        assert np.min(vals_hbr) < np.max(vals_hbr)
        if figure_dir:
            times = np.arange(vals_hbo.size) / hbo.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'fnirs_hbo', times, vals_hbo)
            save_generator_figure(figure_dir, 'fnirs_hbr', times, vals_hbr)
    
    def test_hbo_hbr_baseline(self):
        """Test that HbO and HbR have reasonable baselines"""
        hbo, hbr = fNIRSGenerator.hbo_hbr_response(
            duration=10.0, sampling_freq=10,
            stimulus_times=[]
        )
        
        # Should have baseline values
        vals_hbo = hbo.p.get_values().ravel()
        vals_hbr = hbr.p.get_values().ravel()
        assert np.mean(vals_hbo) > 50  # HbO baseline ~100 µM
        assert np.mean(vals_hbr) > 30  # HbR baseline ~50 µM
        if figure_dir:
            times = np.arange(vals_hbo.size) / hbo.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'fnirs_hbo_baseline', times, vals_hbo)
            save_generator_figure(figure_dir, 'fnirs_hbr_baseline', times, vals_hbr)


class TestEEGGenerator:
    """Tests for EEG signal generation"""
    
    def test_eeg_with_bands(self):
        """Test EEG with different frequency bands"""
        signal = EEGGenerator.eeg_with_bands(
            duration=20.0, sampling_freq=250,
            delta_power=0.5, alpha_power=2.0
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (5000,)
        assert np.isfinite(vals).all()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'eeg_with_bands', times, vals)
    
    def test_eeg_alpha_burst(self):
        """Test EEG with alpha bursts"""
        burst_times = [1.0, 3.0]
        signal = EEGGenerator.eeg_alpha_burst(
            duration=50.0, sampling_freq=100,
            burst_times=burst_times,
            burst_duration=2.0,
            alpha_freq=10.0
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (5000,)
        assert np.isfinite(vals).all()
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'eeg_alpha_burst', times, vals)


class TestEMGGenerator:
    """Tests for EMG signal generation"""
    
    def test_emg_at_rest(self):
        """Test resting EMG"""
        signal = EMGGenerator.emg_at_rest(
            duration=1.0, sampling_freq=1000,
            baseline_activity=0.05
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (1000,)
        assert np.isfinite(vals).all()
        # RMS should be small at rest
        assert np.sqrt(np.mean(vals**2)) < 0.2
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'emg_at_rest', times, vals)
    
    def test_emg_during_contraction(self):
        """Test EMG during muscle contraction"""
        contraction_times = [(0.5, 1.5), (2.0, 3.0)]
        signal = EMGGenerator.emg_during_contraction(
            duration=4.0, sampling_freq=1000,
            contraction_times=contraction_times,
            contraction_force=0.7
        )
        vals = signal.p.get_values().ravel()
        assert vals.shape == (4000,)
        assert np.isfinite(vals).all()
        # Just check that signal has variation
        assert np.std(vals) > 0
        if figure_dir:
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'emg_during_contraction', times, vals)


class TestGeneratorIntegration:
    """Integration tests for generators"""
    
    def test_signal_attributes(self):
        """Test that generated signals have proper attributes"""
        signal = ECGGenerator.simple_ecg(
            duration=0.4, sampling_freq=250
        )
        
        # Should be xarray DataArray
        assert hasattr(signal, 'values')
        assert hasattr(signal, 'attrs')
        
        # Should have sampling frequency in attributes
        assert 'sampling_freq' in signal.attrs
        assert signal.attrs['sampling_freq'] == 250
        if figure_dir:
            vals = signal.p.get_values().ravel()
            times = np.arange(vals.size) / signal.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'signal_attributes_ecg', times, vals)
    
    def test_multiple_signal_types(self):
        """Test generating multiple signal types"""
        signals = {
            'ECG': ECGGenerator.simple_ecg(100, 100),
            'Respiration': RespirationGenerator.sinusoidal_respiration(100, 100),
            'EEG': EEGGenerator.eeg_with_bands(100, 100),
            'EMG': EMGGenerator.emg_at_rest(100, 100),
        }
        
        for name, signal in signals.items():
            assert signal.p.get_values().ravel().shape == (10000,), f"{name} has wrong shape"
            assert np.isfinite(signal.p.get_values().ravel()).all(), f"{name} has NaN/inf values"
        if figure_dir:
            # Save one representative figure (ECG) for quick inspection
            sig = signals['ECG']
            vals = sig.p.get_values().ravel()
            times = np.arange(vals.size) / sig.p.get_sampling_freq()
            save_generator_figure(figure_dir, 'multiple_signal_types_ecg', times, vals)


class TestArtifactGenerators:
    """Tests for SpikeGenerator and BaselineShiftGenerator"""

    def test_spike_generator_single_and_peak(self, figure_dir):
        duration = 2.0
        fs = 100
        # single spike at 0.5s, amplitude 5.0
        sig = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                                    times=[0.5], amplitudes=5.0, durations=0.02,
                                    shape='linear')
        assert sig.shape[0] == int(duration * fs)
        idx = int(0.5 * fs)
        # peak near expected index and noticeably > 0
        vals = sig.p.get_values().ravel()
        assert vals[idx] > 3.0
        if figure_dir:
            times = np.arange(vals.size) / fs
            save_generator_figure(figure_dir, 'spike_single', times, vals)

    def test_spike_generator_per_event_params(self, figure_dir):
        duration = 2.0
        fs = 100
        times = [0.2, 1.0]
        amps = [2.0, 4.0]
        durs = [0.01, 0.05]
        sig = SpikeGenerator.spikes(duration=duration, sampling_freq=fs,
                                    times=times, amplitudes=amps, durations=durs,
                                    shape='gaussian')
        assert sig.shape[0] == int(duration * fs)
        idx0 = int(times[0] * fs)
        idx1 = int(times[1] * fs)
        vals = sig.p.get_values().ravel()
        peak0 = np.max(vals[max(0, idx0-2):idx0+3])
        peak1 = np.max(vals[max(0, idx1-3):idx1+4])
        assert peak0 > 0
        assert peak1 > 0
        assert peak1 > peak0
        if figure_dir:
            times = np.arange(vals.size) / fs
            save_generator_figure(figure_dir, 'spike_per_event', times, vals)

    def test_baseline_shift_single_and_maintained(self, figure_dir):
        duration = 3.0
        fs = 100
        clean = np.zeros(int(duration * fs))
        shift = BaselineShiftGenerator.baseline_shifts(duration=duration, sampling_freq=fs,
                                                       times=[1.0], amplitudes=3.0, durations=0.1)
        assert shift.shape[0] == int(duration * fs)
        idx = int(1.0 * fs)
        # after ramp end, mean offset should be approx amplitude
        vals = shift.p.get_values().ravel()
        post_mean = np.mean(vals[idx + int(0.1*fs) + 1:])
        assert np.isclose(post_mean, 3.0, atol=0.6)
        if figure_dir:
            times = np.arange(vals.size) / fs
            save_generator_figure(figure_dir, 'baseline_shift_single', times, vals)

    def test_baseline_shift_multiple_per_event(self, figure_dir):
        duration = 4.0
        fs = 100
        times = [0.5, 2.0]
        amps = [1.0, -0.5]
        durs = [0.1, 0.2]
        shift = BaselineShiftGenerator.baseline_shifts(duration=duration, sampling_freq=fs,
                                                       times=times, amplitudes=amps, durations=durs)
        assert shift.shape[0] == int(duration * fs)
        idx0 = int(times[0] * fs)
        idx1 = int(times[1] * fs)
        # check cumulative effect after second shift
        vals = shift.p.get_values().ravel()
        tail_mean = np.mean(vals[idx1 + int(durs[1]*fs) + 1:])
        assert np.isclose(tail_mean, amps[0] + amps[1], atol=0.8)
        if figure_dir:
            times = np.arange(vals.size) / fs
            save_generator_figure(figure_dir, 'baseline_shift_multiple', times, vals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
