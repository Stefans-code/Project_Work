import pytest
import numpy as np
from pyphysio.signal import create_signal
import pyphysio.filters as filt
from pyphysio.generators.fundamental import SinusoidalGenerator

# Make tests deterministic
np.random.seed(0)

# TODO-AI [PRIORITY: HIGH]: Use accessor for signal arrays (`signal.p.get_values()`)
# and seed any calls to `np.random.*` to ensure determinism in tests.
# TODO-AI [PRIORITY: MEDIUM]: Parametrize window types and window lengths.


class TestConvolutionalFilter:
    """Tests for ConvolutionalFilter to verify smoothing behavior across different windows and lengths."""
    
    def test_convolutional_filter_smoothing_effect(self):
        """Test that convolutional filters smooth noisy signals."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz
        
        # Create clean sinusoidal signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.p.get_values().ravel()
        
        # Add noise
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
        
        # Test all filter window types (use small window to avoid over-smoothing)
        window_types = ['gauss', 'triang']
        
        for win_type in window_types:
            # Apply convolutional filter with small window
            conv_filter = filt.ConvolutionalFilter(win_type, win_len=0.02)
            filtered_signal = conv_filter(noisy_signal)
            filtered_data = filtered_signal.p.get_values().ravel()
            
            # Compute errors
            noisy_error = np.mean((noisy_data - clean_data) ** 2)
            filtered_error = np.mean((filtered_data - clean_data) ** 2)
            
            # Filter should reduce noise
            assert filtered_error < noisy_error, \
                f"{win_type} filter should reduce noise: filtered={filtered_error:.6f}, noisy={noisy_error:.6f}"
    
    def test_convolutional_filter_window_length_effect(self):
        """Test that window length relative to signal period affects filtering quality.
        
        When window length << signal period, filtering is ineffective.
        When window length is comparable to signal period, filtering works well.
        """
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz, period = 0.1 seconds
        signal_period = 1.0 / signal_freq
        
        # Create clean signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.data.flatten()
        
        # Add sinusoidal noise (high frequency - not in the signal)
        noise_freq = 50  # Hz - much higher frequency than signal
        noise_components = [{'frequency': noise_freq, 'amplitude': 0.3}]
        noise_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, noise_components)
        noise_data = noise_signal.data.flatten()
        
        noisy_data = clean_data + noise_data
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
        
        # Test different window lengths relative to signal period
        # Much smaller than period: should not filter effectively
        # Comparable to period: should filter effectively
        window_lengths = [
            0.02,   # << signal period (0.02 << 0.1)
            0.08,   # ~ signal period (0.08 ~ 0.1)
            0.15    # > signal period (0.15 > 0.1)
        ]
        errors = []
        noisy_error = np.mean((noisy_data - clean_data) ** 2)
        
        for win_len in window_lengths:
            conv_filter = filt.ConvolutionalFilter('gauss', win_len=win_len)
            filtered_signal = conv_filter(noisy_signal)
            filtered_data = np.asarray(filtered_signal).ravel()
            error = np.mean((filtered_data - clean_data) ** 2)
            errors.append(error)
        
        # Window length comparable to signal period should work best
        # Much smaller window (0.02s) should not work as well
        assert errors[1] < errors[0], \
            f"Window {window_lengths[1]}s (~ period) should work better than {window_lengths[0]}s (<< period): {errors}"
        
        # Very large window (0.15s) should over-smooth and perform worse than optimal
        assert errors[1] < errors[2], \
            f"Window {window_lengths[1]}s (~ period) should work better than {window_lengths[2]}s (> period): {errors}"
    
    def test_convolutional_filter_window_types_comparison(self):
        """Test different window types on noisy signal."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz
        
        # Create clean signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.data.flatten()
        
        # Add noise
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
        
        # Apply filters with same small window length
        window_types = ['gauss', 'triang']
        win_len = 0.02
        results = {}
        
        for win_type in window_types:
            conv_filter = filt.ConvolutionalFilter(win_type, win_len=win_len)
            filtered_signal = conv_filter(noisy_signal)
            filtered_data = np.asarray(filtered_signal).ravel()
            error = np.mean((filtered_data - clean_data) ** 2)
            results[win_type] = error
        
        # All window types should reduce noise
        noisy_error = np.mean((noisy_data - clean_data) ** 2)
        for win_type, error in results.items():
            assert error < noisy_error, \
                f"{win_type} should reduce noise: {error:.6f} < {noisy_error:.6f}"
    
    def test_convolutional_filter_preserves_signal_shape(self):
        """Test that convolutional filter preserves signal length and shape."""
        fsamp = 500  # sampling frequency
        duration = 3  # seconds
        
        # Create multi-component signal
        components = [
            {'frequency': 5, 'amplitude': 0.5},
            {'frequency': 15, 'amplitude': 0.5}
        ]
        signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        signal_obj = create_signal(signal.data, sampling_freq=fsamp, name='test_signal')
        
        # Test all window types
        window_types = ['gauss', 'rect', 'triang', 'dgauss']
        
        for win_type in window_types:
            conv_filter = filt.ConvolutionalFilter(win_type, win_len=0.1)
            filtered_signal = conv_filter(signal_obj)
            
            # Check shape preservation
            assert filtered_signal.shape == signal_obj.values.ravel().shape, \
                f"{win_type} filter should preserve signal shape"
            assert len(filtered_signal) == len(signal.data.flatten()), \
                f"{win_type} filter should preserve signal length"
    
    def test_convolutional_filter_dgauss_derivative_effect(self):
        """Test that derivative of Gaussian filter highlights changes in signal."""
        fsamp = 1000  # sampling frequency
        
        # Create step signal
        data_len = 5000
        data = np.zeros(data_len)
        data[int(data_len/2):] = 1.0  # Step at midpoint
        signal = create_signal(data, sampling_freq=fsamp, name='step_signal')
        
        # Apply dgauss filter (derivative of Gaussian)
        dgauss_filter = filt.ConvolutionalFilter('dgauss', win_len=0.1)
        filtered_signal = dgauss_filter(signal)
        filtered_data = np.asarray(filtered_signal).ravel()
        
        # Derivative should show peak around the step location
        # Find location of maximum absolute value
        max_idx = np.argmax(np.abs(filtered_data))
        step_idx = int(data_len/2)
        
        # Maximum should be near the step (within 500 samples = 0.5 seconds)
        assert abs(max_idx - step_idx) < 500, \
            f"dgauss filter should highlight step at idx {step_idx}, found peak at {max_idx}"
    
    def test_convolutional_filter_normalization_effect(self):
        """Test that normalization affects filter output scale."""
        fsamp = 1000  # sampling frequency
        duration = 2  # seconds
        
        # Create constant signal
        data = np.ones(int(fsamp * duration))
        signal = create_signal(data, sampling_freq=fsamp, name='constant_signal')
        
        # Apply normalized filter
        conv_norm = filt.ConvolutionalFilter('rect', win_len=0.1, normalize=True)
        filtered_norm = conv_norm(signal)
        
        # Apply non-normalized filter
        conv_no_norm = filt.ConvolutionalFilter('rect', win_len=0.1, normalize=False)
        filtered_no_norm = conv_no_norm(signal)
        
        # Constant input should produce approximately constant output
        # Normalized version should be closer to original value
        norm_mean = np.mean(filtered_norm)
        no_norm_mean = np.mean(filtered_no_norm)
        
        # Normalized should be closer to 1.0 (original value)
        assert abs(norm_mean - 1.0) < abs(no_norm_mean - 1.0), \
            f"Normalized filter should preserve constant signal better: {norm_mean:.4f} vs {no_norm_mean:.4f}"
    
    def test_convolutional_filter_custom_irf(self):
        """Test convolutional filter with custom impulse response."""
        fsamp = 1000  # sampling frequency
        duration = 2  # seconds
        
        # Create constant signal
        data = np.ones(int(fsamp * duration))
        signal = create_signal(data, sampling_freq=fsamp, name='constant_signal')
        
        # Define custom impulse response (simple moving average)
        custom_irf = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Apply custom filter
        conv_custom = filt.ConvolutionalFilter('custom', irf=custom_irf)
        filtered_signal = conv_custom(signal)
        
        # For constant input with normalized irf, output should be close to input
        assert np.allclose(filtered_signal, 1.0, atol=0.1), \
            f"Custom filter with constant input should produce constant output close to 1.0"
    
    def test_convolutional_filter_noise_reduction_vs_smoothing(self):
        """Test trade-off between noise reduction and signal preservation."""
        fsamp = 1000  # sampling frequency
        duration = 5  # seconds
        signal_freq = 10  # Hz
        
        # Create clean signal
        components = [{'frequency': signal_freq, 'amplitude': 1.0}]
        clean_signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        clean_data = clean_signal.data.flatten()
        
        # Add noise
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, len(clean_data))
        noisy_data = clean_data + noise
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='noisy_signal')
        
        # Test with different window lengths (small windows preserve signal better)
        window_lengths = [0.01, 0.02, 0.05]
        
        for win_len in window_lengths:
            conv_filter = filt.ConvolutionalFilter('gauss', win_len=win_len)
            filtered_signal = conv_filter(noisy_signal)
            filtered_data = np.asarray(filtered_signal).ravel()
            
            # Compute errors relative to clean signal
            error = np.mean((filtered_data - clean_data) ** 2)
            
            # Error should be positive and reasonable
            assert error > 0, "Error should be positive"
            # Even large windows should improve over pure noise
            assert error < np.mean((noisy_data - clean_data) ** 2), \
                f"Filter should improve upon noisy signal for window {win_len}"
