"""
Fundamental Signal Generators

This module provides functions to generate basic, well-defined synthetic signals
with known properties. These signals are useful for testing algorithms, validating
signal processing operations, and understanding signal behavior.

Classes and Functions:
    - SinusoidalGenerator: Generate sinusoidal signals with specified frequency/amplitude
    - NoiseGenerator: Generate various types of noise
    - CompositeGenerator: Combine multiple signals
    - FundamentalGenerator: Generate standard test signals (zeros, ones, deltas, ramps)
    - WindowGenerator: Generate window functions
"""

import numpy as np
from scipy.signal import get_window, chirp, sawtooth
from ..signal import create_signal


class FundamentalSignalGenerator:
    """
    Generate fundamental signal patterns with well-known properties.
    
    These are basic building blocks used to create more complex synthetic signals.
    """
    
    @staticmethod
    def zeros(duration, sampling_freq, start_time=0):
        """
        Generate a signal of all zeros.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Zero signal
        """
        n_timepoints = int(duration * sampling_freq)
        data = np.zeros(n_timepoints)
        return create_signal(data, sampling_freq=sampling_freq, 
                            start_time=start_time, name='zeros')
    
    @staticmethod
    def ones(duration, sampling_freq, start_time=0):
        """
        Generate a signal of all ones (DC component).
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Constant signal of ones
        """
        n_timepoints = int(duration * sampling_freq)
        data = np.ones(n_timepoints)
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='ones')
    
    @staticmethod
    def constant(duration, sampling_freq, value=1.0, start_time=0):
        """
        Generate a constant-valued signal.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        value : float, optional
            Constant value (default: 1.0)
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Constant signal
        """
        n_timepoints = int(duration * sampling_freq)
        data = np.ones(n_timepoints) * value
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name=f'constant_{value}')
    
    @staticmethod
    def ramp(duration, sampling_freq, start_value=0, end_value=1, start_time=0):
        """
        Generate a linear ramp signal.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        start_value : float, optional
            Value at start (default: 0)
        end_value : float, optional
            Value at end (default: 1)
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Linear ramp signal
        """
        n_timepoints = int(duration * sampling_freq)
        data = np.linspace(start_value, end_value, n_timepoints)
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='ramp')
    
    @staticmethod
    def delta(duration, sampling_freq, delta_times, delta_values, start_time=0):
        """
        Generate a signal with impulses (Dirac deltas) at specific times.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        delta_times : array-like
            Times (in seconds) where impulses occur
        delta_values : array-like
            Values (amplitudes) of impulses
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Impulse signal
            
        Raises
        ------
        AssertionError
            If delta_times and delta_values have different lengths
        """
        n_timepoints = int(duration * sampling_freq)
        assert len(delta_times) == len(delta_values), \
            "delta_times and delta_values must have same length"
        
        data = np.zeros(n_timepoints)
        delta_times_nostart = np.asarray(delta_times) - start_time
        idx_delta_times = (delta_times_nostart * sampling_freq).astype(int)
        
        # Filter indices that are within valid range
        valid_idx = (idx_delta_times >= 0) & (idx_delta_times < n_timepoints)
        data[idx_delta_times[valid_idx]] = np.asarray(delta_values)[valid_idx]
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='delta')
    
    @staticmethod
    def boxcar(duration, sampling_freq, start_time=0, 
               high_value=1, low_value=0, transitions=None):
        """
        Generate a signal with rectangular pulses (box car signal).
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        start_time : float, optional
            Start time of signal
        high_value : float, optional
            Value during high periods (default: 1)
        low_value : float, optional
            Value during low periods (default: 0)
        transitions : list of tuples, optional
            List of (time_on, time_off) tuples in seconds
            Example: [(0, 1), (2, 3)] = on [0-1s], off [1-2s], on [2-3s], off [3-end]
            
        Returns
        -------
        signal : xarray.DataArray
            Box car signal
        """
        n_timepoints = int(duration * sampling_freq)
        data = np.ones(n_timepoints) * low_value
        
        if transitions is not None:
            for on_time, off_time in transitions:
                on_idx = int((on_time - start_time) * sampling_freq)
                off_idx = int((off_time - start_time) * sampling_freq)
                on_idx = max(0, on_idx)
                off_idx = min(n_timepoints, off_idx)
                data[on_idx:off_idx] = high_value
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='boxcar')


class SinusoidalGenerator:
    """
    Generate sinusoidal signals with various characteristics.
    """
    
    @staticmethod
    def simple_sine(duration, sampling_freq, amplitude=1.0, frequency=1.0, 
                   phase=0, start_time=0):
        """
        Generate a simple sinusoidal signal.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        amplitude : float, optional
            Amplitude of sinusoid (default: 1.0)
        frequency : float, optional
            Frequency in Hz (default: 1.0)
        phase : float, optional
            Phase in radians (default: 0)
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Sinusoidal signal with frequency information in attributes
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        data = amplitude * np.sin(2 * np.pi * frequency * times + phase)
        signal = create_signal(data, sampling_freq=sampling_freq,
                              start_time=start_time, 
                              name=f'sine_{frequency}Hz')
        signal.attrs['frequency'] = frequency
        signal.attrs['amplitude'] = amplitude
        signal.attrs['phase'] = phase
        signal.attrs['frequencies'] = [frequency]  # For PSD validation
        return signal
    
    @staticmethod
    def multi_component_sine(duration, sampling_freq, components, start_time=0):
        """
        Generate a signal with multiple sinusoidal components.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        components : list of dict
            Each dict should have keys:
            - 'frequency': frequency in Hz
            - 'amplitude': amplitude (default: 1.0)
            - 'phase': phase in radians (default: 0)
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Multi-component sinusoidal signal with frequency information in attributes
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        data = np.zeros(n_timepoints)
        
        frequencies = []
        for component in components:
            freq = component['frequency']
            amp = component.get('amplitude', 1.0)
            phase = component.get('phase', 0)
            data += amp * np.sin(2 * np.pi * freq * times + phase)
            frequencies.append(freq)
        
        freqs = [c['frequency'] for c in components]
        signal = create_signal(data, sampling_freq=sampling_freq,
                              start_time=start_time,
                              name=f'multi_sine_{freqs}')
        signal.attrs['frequencies'] = frequencies
        signal.attrs['n_components'] = len(components)
        signal.attrs['components'] = components
        return signal
    
    @staticmethod
    def frequency_sweep(duration, sampling_freq, f_start, f_end, 
                       amplitude=1.0, method='linear', start_time=0):
        """
        Generate a chirp signal (frequency sweep).
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        f_start : float
            Starting frequency in Hz
        f_end : float
            Ending frequency in Hz
        amplitude : float, optional
            Amplitude (default: 1.0)
        method : {'linear', 'quadratic', 'logarithmic'}, optional
            Sweep method (default: 'linear')
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Chirp signal with sweeping frequency
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        data = amplitude * chirp(times, f_start, times[-1], f_end, method=method)
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time,
                            name=f'chirp_{f_start}_{f_end}_Hz')
    
    @staticmethod
    def amplitude_modulated(duration, sampling_freq, carrier_freq, 
                           modulation_freq, amplitude=1.0, start_time=0):
        """
        Generate an amplitude-modulated sinusoid.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        carrier_freq : float
            Carrier frequency in Hz
        modulation_freq : float
            Modulation frequency in Hz
        amplitude : float, optional
            Amplitude (default: 1.0)
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Amplitude-modulated signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        carrier = np.sin(2 * np.pi * carrier_freq * times)
        modulation = 0.5 * (1 + np.sin(2 * np.pi * modulation_freq * times))
        data = amplitude * carrier * modulation
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time,
                            name=f'AM_{carrier_freq}_{modulation_freq}Hz')


class NoiseGenerator:
    """
    Generate various types of noise for testing algorithm robustness.
    """
    
    @staticmethod
    def white_noise(duration, sampling_freq, std=1.0, mean=0, 
                   seed=None, start_time=0):
        """
        Generate white (Gaussian) noise.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        std : float, optional
            Standard deviation (default: 1.0)
        mean : float, optional
            Mean (default: 0)
        seed : int, optional
            Random seed for reproducibility
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            White noise signal
        """
        n_timepoints = int(duration * sampling_freq)
        if seed is not None:
            np.random.seed(seed)
        data = np.random.normal(mean, std, n_timepoints)
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='white_noise')
    
    @staticmethod
    def pink_noise(duration, sampling_freq, std=1.0, mean=0, 
                  seed=None, start_time=0):
        """
        Generate pink noise (1/f noise) using spectral method.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        std : float, optional
            Standard deviation (default: 1.0)
        mean : float, optional
            Mean (default: 0)
        seed : int, optional
            Random seed for reproducibility
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Pink noise signal
        """
        n_timepoints = int(duration * sampling_freq)
        if seed is not None:
            np.random.seed(seed)
        
        # Generate white noise in frequency domain
        white = np.random.normal(0, 1, n_timepoints)
        fft = np.fft.fft(white)
        
        # Apply 1/f filter
        freqs = np.abs(np.fft.fftfreq(n_timepoints))
        freqs[0] = 1  # Avoid division by zero
        fft = fft / np.sqrt(freqs)
        
        # Convert back to time domain
        data = np.real(np.fft.ifft(fft))
        data = (data - np.mean(data)) / np.std(data) * std + mean
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='pink_noise')
    
    @staticmethod
    def uniform_noise(duration, sampling_freq, low=0, high=1,
                     seed=None, start_time=0):
        """
        Generate uniform random noise.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        low : float, optional
            Lower bound (default: 0)
        high : float, optional
            Upper bound (default: 1)
        seed : int, optional
            Random seed for reproducibility
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Uniform noise signal
        """
        n_timepoints = int(duration * sampling_freq)
        if seed is not None:
            np.random.seed(seed)
        data = np.random.uniform(low, high, n_timepoints)
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='uniform_noise')


class CompositeSignalGenerator:
    """
    Generate composite signals by combining multiple generators.
    """
    
    @staticmethod
    def sum_signals(signals):
        """
        Sum multiple signals together.
        
        Parameters
        ----------
        signals : list of xarray.DataArray
            Signals to sum (must have same shape)
            
        Returns
        -------
        signal : xarray.DataArray
            Sum of input signals
        """
        result = signals[0].copy()
        for sig in signals[1:]:
            result.values = result.values + sig.values
        result.name = 'composite_sum'
        return result
    
    @staticmethod
    def concatenate_signals(signals):
        """
        Concatenate signals in time.
        
        Parameters
        ----------
        signals : list of xarray.DataArray
            Signals to concatenate
            
        Returns
        -------
        signal : xarray.DataArray
            Concatenated signal
        """
        import xarray as xr
        result = xr.concat(signals, dim='time')
        result.name = 'concatenated'
        return result

#TODO: concatenate crossfade? other concatenation methods?
    
    @staticmethod
    def signal_with_noise(signal, snr_db):
        """
        Add Gaussian noise to a signal at specified SNR.
        
        Parameters
        ----------
        signal : xarray.DataArray
            Original signal
        snr_db : float
            Signal-to-noise ratio in dB
            
        Returns
        -------
        noisy_signal : xarray.DataArray
            Signal with added noise
        """
        signal_power = np.mean(signal.values ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        
        result = signal.copy()
        result.values = result.values + noise
        result.name = f'{signal.name}_SNR{snr_db}dB'
        return result

    @staticmethod
    def signal_with_artifacts(signal, artifact_times, artifact_type='spike',
                            artifact_amplitude=None):
        """
        Add artifacts to a signal at specific times.
        
        Parameters
        ----------
        signal : xarray.DataArray
            Original signal
        artifact_times : list of float
            Times (in seconds) where artifacts occur
        artifact_type : {'spike', 'gaussian', 'impulse'}, optional
            Type of artifact (default: 'spike')

#TODO: Baseline shift artefact

        artifact_amplitude : float, optional
            Amplitude of artifacts (default: 5x signal std)
            
        Returns
        -------
        signal_with_artifacts : xarray.DataArray
            Signal with added artifacts
        """
        result = signal.copy()
        sampling_freq = signal.attrs['sampling_freq']
        start_time = signal.attrs.get('start_time', 0)
        
        if artifact_amplitude is None:
            artifact_amplitude = 5 * np.std(signal.values)
        
        for art_time in artifact_times:
            idx = int((art_time - start_time) * sampling_freq)
            if 0 <= idx < signal.shape[0]:
                if artifact_type == 'spike':
                    result.values[idx] = artifact_amplitude
                elif artifact_type == 'impulse':
                    if idx + 1 < signal.shape[0]:
                        result.values[idx:idx+2] = artifact_amplitude
                elif artifact_type == 'gaussian':
                    # Add Gaussian-shaped artifact
                    width = int(sampling_freq * 0.05)  # 50ms width
                    x = np.arange(-width, width)
                    gaussian = artifact_amplitude * np.exp(-(x**2) / (2*width**2))
                    start_idx = max(0, idx - width)
                    end_idx = min(signal.shape[0], idx + width)
                    result.values[start_idx:end_idx] += gaussian[
                        max(0, width - idx):
                        width + (end_idx - idx)
                    ]
        
        result.name = f'{signal.name}_artifacts'
        return result


class WindowGenerator:
    """
    Generate window functions for signal windowing and analysis.
    """
    
    @staticmethod
    def apply_window(signal, window_type='hann'):
        """
        Apply a window function to a signal.
        
        Parameters
        ----------
        signal : xarray.DataArray
            Signal to window
        window_type : str, optional
            Type of window: 'hann', 'hamming', 'blackman', 'bartlett', etc.
            (default: 'hann')
            
        Returns
        -------
        windowed_signal : xarray.DataArray
            Windowed signal
        """
        window = get_window(window_type, signal.shape[0])
        result = signal.copy()
        result.values = result.values * window[:, np.newaxis] if result.ndim > 1 else result.values * window
        result.name = f'{signal.name}_{window_type}_windowed'
        return result
    
    @staticmethod
    def generate_window(duration, window_type='hann', return_signal=False,
                       sampling_freq=1.0, start_time=0):
        """
        Generate a window function signal.
        
        Parameters
        ----------
        n_timepoints : int
            Number of samples
        window_type : str, optional
            Type of window (default: 'hann')
        return_signal : bool, optional
            If True, return as signal object (default: False)
        sampling_freq : float, optional
            Sampling frequency (used if return_signal=True)
        start_time : float, optional
            Start time (used if return_signal=True)
            
        Returns
        -------
        window : ndarray or xarray.DataArray
            Window function
        """
        n_timepoints = int(duration * sampling_freq)
        window = get_window(window_type, n_timepoints)
        
        if return_signal:
            return create_signal(window, sampling_freq=sampling_freq,
                               start_time=start_time, name=f'{window_type}_window')
        return window


class SawtoothGenerator:
    """
    Generate sawtooth waves with adjustable periods and amplitudes.
    """
    
    @staticmethod
    def sawtooth(duration, sampling_freq, frequency=1.0, amplitude=1.0,
                width=1.0, start_time=0):
        """
        Generate a sawtooth wave.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        frequency : float, optional
            Frequency of sawtooth in Hz (default: 1.0)
        amplitude : float, optional
            Amplitude of sawtooth (default: 1.0)
        width : float, optional
            Ratio of rise time to period (0-1) (default: 1.0)
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Sawtooth wave signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        data = amplitude * sawtooth(2 * np.pi * frequency * times, width)
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name=f'sawtooth_{frequency}Hz')


class GaussianComponentGenerator:
    """
    Generate signals using mixed Gaussian components.
    """
    
    @staticmethod
    def mixed_gaussians(duration, sampling_freq, 
                       means, stds, amplitudes, start_time=0):
        """
        Generate a signal as sum of Gaussian components.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        means : list of float
            Mean positions of Gaussians in time (seconds)
        stds : list of float
            Standard deviations of Gaussians (seconds)
        amplitudes : list of float
            Amplitudes of each Gaussian
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Composite Gaussian signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        data = np.zeros(n_timepoints)
        
        for mean, std, amp in zip(means, stds, amplitudes):
            gaussian = amp * np.exp(-((times - mean) ** 2) / (2 * std ** 2))
            data += gaussian
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='mixed_gaussians')


class PolynomialGenerator:
    """
    Generate polynomial curve signals.
    """
    
    @staticmethod
    def polynomial(duration, sampling_freq, coefficients, start_time=0):
        """
        Generate a polynomial signal.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        coefficients : list of float
            Polynomial coefficients from highest to lowest degree
            e.g., [1, 2, 3] for x^2 + 2*x + 3
        start_time : float, optional
            Start time of signal
            
        Returns
        -------
        signal : xarray.DataArray
            Polynomial signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        
        # Normalize times to [-1, 1] for stability
        times_norm = 2 * (times - times[0]) / (times[-1] - times[0] + 1e-10) - 1
        
        # Evaluate polynomial
        data = np.polyval(coefficients, times_norm)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='polynomial')