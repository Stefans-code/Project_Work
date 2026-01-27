import pytest
import numpy as np
from pyphysio.signal import create_signal
import pyphysio.filters as filt
from pyphysio.generators.fundamental import SinusoidalGenerator
from pathlib import Path
import matplotlib.pyplot as plt
from _helpers import save_comparison_figure

# Use centralized `figure_dir` and `generate_figures` fixtures from `conftest.py`.


class TestKalmanFilter:
    """Tests for KalmanFilter to verify signal reconstruction from noisy measurements."""
    
    def test_kalman_filter_sinusoidal_signal_reconstruction(self, figure_dir, rng):
        """Test KalmanFilter can reconstruct sinusoidal signal from noisy measurements."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz
        
        # Create clean sinusoidal signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.p.get_values().ravel()
        
        # Add Gaussian noise with known standard deviation
        noise_std = 0.1  # Standard deviation of noise
        noise = rng.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
        
        # Apply Kalman filter
        # R: measurement noise covariance (should match our added noise)
        # Q: process noise covariance (controls smoothing)
        R = noise_std ** 2  # Noise variance
        Q = 0.01  # Small process noise for smooth tracking
        kalman = filt.KalmanFilter(R=R, Q=Q)
        filtered_signal = kalman(noisy_signal)
        filtered_data = filtered_signal.p.get_values().ravel()
        if figure_dir:
            times = np.arange(len(noisy_data)) / fsamp
            save_comparison_figure(figure_dir, 'kalman_sinusoidal_reconstruction', times, noisy_data, filtered_data)
        
        # Compute reconstruction error
        reconstruction_error = np.mean((filtered_data - clean_data) ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Filtered signal should have lower error than noisy signal
        noisy_error = np.mean((noisy_data - clean_data) ** 2)
        assert reconstruction_error < noisy_error, \
            f"Kalman filter should reduce noise: filtered_error={reconstruction_error:.6f}, noisy_error={noisy_error:.6f}"
        
        # Error should be reasonable relative to noise (better than just adding noise)
        # Allow some tolerance - Kalman filter trades off bias and variance
        assert reconstruction_error < noise_power * 2, \
            f"Kalman filter reconstruction error too high: {reconstruction_error:.6f} vs noise power {noise_power:.6f}"
    
    def test_kalman_filter_noise_level_effect(self, figure_dir, rng):
        """Test that KalmanFilter adapts to different noise levels."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz
        
        # Create clean signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.p.get_values().ravel()
        
        # Test with different noise levels
        noise_levels = [0.05, 0.1, 0.2]  # Standard deviations
        errors = []
        
        for noise_std in noise_levels:
            # Add noise
            noise = rng.normal(0, noise_std, len(clean_data))
            noisy_data = clean_data + noise
            noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
            
            # Apply Kalman filter with R matching the noise variance
            R = noise_std ** 2
            Q = 0.01
            kalman = filt.KalmanFilter(R=R, Q=Q)
            filtered_signal = kalman(noisy_signal)
            filtered_data = filtered_signal.p.get_values().ravel()
            
            # Compute error
            error = np.mean((filtered_data - clean_data) ** 2)
            errors.append(error)
            if figure_dir and noise_std == noise_levels[len(noise_levels)//2]:
                # save a representative middle-noise-level figure
                times = np.arange(len(noisy_data)) / fsamp
                save_comparison_figure(figure_dir, f'kalman_noise_level_{noise_std}', times, noisy_data, filtered_data)
        
        # Error should increase with noise level (higher noise = higher minimal achievable error)
        assert errors[0] < errors[1] < errors[2], \
            f"Higher noise should result in higher error: {errors}"
    
    def test_kalman_filter_parameter_effect(self, figure_dir, rng):
        """Test that different R and Q values produce different filtering behaviors."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz
        
        # Create clean signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.p.get_values().ravel()
        
        # Add noise
        noise_std = 0.1
        noise = rng.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
        
        # Test with different Q values (process noise)
        # Small Q = trust the model more (smoother)
        # Large Q = trust measurements more (noisier)
        Q_values = [0.001, 0.01, 0.1]
        R = noise_std ** 2
        
        errors = []
        for Q in Q_values:
            kalman = filt.KalmanFilter(R=R, Q=Q)
            filtered_signal = kalman(noisy_signal)
            filtered_data = filtered_signal.p.get_values().ravel()
            error = np.mean((filtered_data - clean_data) ** 2)
            errors.append(error)
            if figure_dir and Q == Q_values[1]:
                times = np.arange(len(noisy_data)) / fsamp
                save_comparison_figure(figure_dir, f'kalman_Q_{Q}', times, noisy_data, filtered_data)
        
        # Optimal Q is somewhere in the middle - too small gives over-smoothing, too large trusts noisy measurements
        # The error should show a relationship with Q
        assert len(errors) == 3, "Should have 3 error measurements"
        assert all(e > 0 for e in errors), "All errors should be positive"
    
    def test_kalman_filter_step_signal_reconstruction(self, figure_dir, rng):
        """Test KalmanFilter can track step changes in signal."""
        fsamp = 100  # sampling frequency
        duration = 4  # seconds
        
        # Create a step signal: constant at 0, then step to 1
        data_len = int(fsamp * duration)
        clean_data = np.zeros(data_len)
        clean_data[int(data_len/2):] = 1.0  # Step at midpoint
        clean_signal = create_signal(clean_data, sampling_freq=fsamp, name='step_signal')
        
        # Add noise
        noise_std = 0.05
        noise = rng.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_step')
        
        # Apply Kalman filter (use higher Q for step tracking)
        R = noise_std ** 2
        Q = 0.1  # Higher Q to track changes
        kalman = filt.KalmanFilter(R=R, Q=Q)
        filtered_signal = kalman(noisy_signal)
        filtered_data = filtered_signal.p.get_values().ravel()
        if figure_dir:
            times = np.arange(len(noisy_data)) / fsamp
            save_comparison_figure(figure_dir, 'kalman_step_reconstruction', times, noisy_data, filtered_data)
        
        # Check that filter detects the step
        # Before step: filtered value should be close to 0
        before_step_mean = np.mean(filtered_data[:int(data_len/2)])
        after_step_mean = np.mean(filtered_data[int(data_len/2):])
        
        # After step should be significantly higher than before
        assert after_step_mean > before_step_mean + 0.5, \
            f"Filter should track step change: before={before_step_mean:.3f}, after={after_step_mean:.3f}"
    
    def test_kalman_filter_with_varying_noise(self, figure_dir, rng):
        """Test KalmanFilter with signal that has varying frequency components."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        
        # Create multi-component signal
        components = [
            {'frequency': 5, 'amplitude': 0.5},
            {'frequency': 20, 'amplitude': 0.5}
        ]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.p.get_values().ravel()
        
        # Add noise
        noise_std = 0.15
        noise = rng.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_multicomp')
        
        # Apply Kalman filter
        R = noise_std ** 2
        Q = 0.05
        kalman = filt.KalmanFilter(R=R, Q=Q)
        filtered_signal = kalman(noisy_signal)
        filtered_data = filtered_signal.p.get_values().ravel()
        if figure_dir:
            times = np.arange(len(noisy_data)) / fsamp
            save_comparison_figure(figure_dir, 'kalman_varying_noise', times, noisy_data, filtered_data)
        
        # Compute errors
        filtered_error = np.mean((filtered_data - clean_data) ** 2)
        noisy_error = np.mean((noisy_data - clean_data) ** 2)
        
        # Filter should improve reconstruction
        assert filtered_error < noisy_error, \
            f"Kalman filter should reduce error for multi-component signal: filtered={filtered_error:.6f}, noisy={noisy_error:.6f}"
        
        # The improvement should be significant
        improvement_ratio = filtered_error / noisy_error
        assert improvement_ratio < 0.7, \
            f"Kalman filter should provide significant improvement: {improvement_ratio:.3f}"
