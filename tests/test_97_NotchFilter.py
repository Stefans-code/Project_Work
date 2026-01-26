import pytest
import numpy as np
from pyphysio.signal import create_signal
import pyphysio.filters as filt
from pyphysio.generators.fundamental import SinusoidalGenerator

# TODO-AI [PRIORITY: HIGH]: Use `signal.p.get_values()` when computing FFT/power
# and seed any random components used to build test signals.
# TODO-AI [PRIORITY: MEDIUM]: Use `pytest.approx` for power ratio tolerances.


class TestNotchFilter:
    """Tests for NotchFilter to verify frequency attenuation behavior."""
    
    def _compute_power_at_frequency(self, signal, freq_hz, fsamp, bandwidth_hz=5):
        """Compute power in a frequency band using FFT.
        
        Args:
            signal: Input signal
            freq_hz: Center frequency in Hz
            fsamp: Sampling frequency in Hz
            bandwidth_hz: Bandwidth around target frequency in Hz
            
        Returns:
            Power in the frequency band
        """
        data = signal.data.flatten() if hasattr(signal.data, 'flatten') else signal.data
        fft_vals = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/fsamp)
        power = np.abs(fft_vals) ** 2 / len(data)  # Normalize by length to avoid overflow
        
        # Select frequency band
        mask = (np.abs(freqs) >= freq_hz - bandwidth_hz/2) & (np.abs(freqs) <= freq_hz + bandwidth_hz/2)
        band_power = np.sum(power[mask])
        return band_power
    
    def test_notch_filter_frequency_attenuation(self):
        """Test NotchFilter correctly attenuates target frequency while preserving others."""
        fsamp = 1000  # sampling frequency
        
        # Define test cases with different notch target frequencies and Q factors
        test_cases = [
            {'notch_freq': 50, 'Q': 30, 'nearby_freqs': [40, 50, 60]},
            {'notch_freq': 100, 'Q': 30, 'nearby_freqs': [80, 100, 120]},
            {'notch_freq': 150, 'Q': 30, 'nearby_freqs': [130, 150, 170]},
            {'notch_freq': 50, 'Q': 60, 'nearby_freqs': [40, 50, 60]},  # Higher Q = narrower notch
            {'notch_freq': 100, 'Q': 15, 'nearby_freqs': [80, 100, 120]},  # Lower Q = wider notch
        ]
        
        for case in test_cases:
            notch_freq = case['notch_freq']
            Q = case['Q']
            nearby_freqs = case['nearby_freqs']
            
            # Create multi-component sinusoidal signal with frequencies around the notch target
            duration = 10  # seconds
            
            components = [{'frequency': freq, 'amplitude': 1.0} for freq in nearby_freqs]
            signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
            
            # Compute original power at each frequency
            orig_power = {freq: self._compute_power_at_frequency(signal, freq, fsamp) for freq in nearby_freqs}
            
            # Apply notch filter
            filter_obj = filt.NotchFilter(f=notch_freq, Q=Q)
            filtered = filter_obj(signal)
            
            # Compute filtered power at each frequency
            filt_power = {freq: self._compute_power_at_frequency(filtered, freq, fsamp) for freq in nearby_freqs}
            
            # Verify that the notch frequency is attenuated more than nearby frequencies
            notch_attenuation_ratio = filt_power[notch_freq] / orig_power[notch_freq] if orig_power[notch_freq] > 0 else 0
            
            # Verify nearby frequencies are less attenuated than the notch frequency
            for freq in nearby_freqs:
                if freq != notch_freq:
                    nearby_attenuation_ratio = filt_power[freq] / orig_power[freq] if orig_power[freq] > 0 else 0
                    assert nearby_attenuation_ratio > notch_attenuation_ratio, \
                        f"Notch {notch_freq}Hz (Q={Q}): nearby {freq}Hz should be attenuated less than notch"
            
            # Verify notch frequency is significantly attenuated (less than 20% of original)
            assert notch_attenuation_ratio < 0.2, \
                f"Notch {notch_freq}Hz (Q={Q}): notch frequency not attenuated enough (ratio={notch_attenuation_ratio:.4f})"
    
    def test_notch_filter_preserves_distant_frequencies(self):
        """Test NotchFilter preserves frequencies far from the notch target."""
        fsamp = 1000  # sampling frequency
        
        # Define test cases with notch target and distant frequencies
        test_cases = [
            {'notch_freq': 100, 'Q': 30, 'target_freq': 100, 'distant_freqs': [10, 250]},
            {'notch_freq': 200, 'Q': 30, 'target_freq': 200, 'distant_freqs': [30, 350]},
            {'notch_freq': 150, 'Q': 60, 'target_freq': 150, 'distant_freqs': [50, 300]},
        ]
        
        for case in test_cases:
            notch_freq = case['notch_freq']
            Q = case['Q']
            target_freq = case['target_freq']
            distant_freqs = case['distant_freqs']
            
            # Create multi-component signal with target frequency and distant frequencies
            duration = 10  # seconds
            all_freqs = [target_freq] + distant_freqs
            components = [{'frequency': freq, 'amplitude': 1.0} for freq in all_freqs]
            signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
            
            # Compute original power at each frequency
            orig_power = {freq: self._compute_power_at_frequency(signal, freq, fsamp) for freq in all_freqs}
            
            # Apply notch filter
            filter_obj = filt.NotchFilter(f=notch_freq, Q=Q)
            filtered = filter_obj(signal)
            
            # Compute filtered power at each frequency
            filt_power = {freq: self._compute_power_at_frequency(filtered, freq, fsamp) for freq in all_freqs}
            
            # Verify distant frequencies are preserved (more than 50% of original power)
            for freq in distant_freqs:
                attenuation_ratio = filt_power[freq] / orig_power[freq] if orig_power[freq] > 0 else 0
                assert attenuation_ratio > 0.5, \
                    f"Notch {notch_freq}Hz (Q={Q}): distant {freq}Hz should be preserved (ratio={attenuation_ratio:.4f})"
    
    def test_notch_filter_Q_factor_effect(self):
        """Test that higher Q factor results in narrower notch (less attenuation at nearby frequencies)."""
        fsamp = 1000  # sampling frequency
        notch_freq = 100
        Q_low = 15
        Q_high = 60
        
        # Create signal with notch frequency and nearby frequency
        nearby_freq = 110  # 10 Hz away from notch
        duration = 10  # seconds
        components = [
            {'frequency': notch_freq, 'amplitude': 1.0},
            {'frequency': nearby_freq, 'amplitude': 1.0}
        ]
        signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
        
        # Get original power at nearby frequency
        orig_power_nearby = self._compute_power_at_frequency(signal, nearby_freq, fsamp)
        
        # Apply notch filter with low Q
        filter_low_Q = filt.NotchFilter(f=notch_freq, Q=Q_low)
        filtered_low_Q = filter_low_Q(signal)
        filt_power_nearby_low_Q = self._compute_power_at_frequency(filtered_low_Q, nearby_freq, fsamp)
        attenuation_low_Q = filt_power_nearby_low_Q / orig_power_nearby if orig_power_nearby > 0 else 0
        
        # Apply notch filter with high Q
        filter_high_Q = filt.NotchFilter(f=notch_freq, Q=Q_high)
        filtered_high_Q = filter_high_Q(signal)
        filt_power_nearby_high_Q = self._compute_power_at_frequency(filtered_high_Q, nearby_freq, fsamp)
        attenuation_high_Q = filt_power_nearby_high_Q / orig_power_nearby if orig_power_nearby > 0 else 0
        
        # Higher Q should result in less attenuation at nearby frequencies (narrower notch)
        assert attenuation_high_Q > attenuation_low_Q, \
            f"Higher Q ({Q_high}) should result in less attenuation at nearby frequency than lower Q ({Q_low})"
