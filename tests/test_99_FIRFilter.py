import pytest
import numpy as np
from pyphysio.signal import create_signal
import pyphysio.filters as filt
from pyphysio.generators.fundamental import SinusoidalGenerator
from pathlib import Path
import matplotlib.pyplot as plt
from _helpers import save_comparison_figure


class TestFIRFilter:
    """Tests for FIRFilter to verify frequency filtering behavior."""
    
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
        data = signal.p.get_values().ravel()
        fft_vals = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/fsamp)
        power = np.abs(fft_vals) ** 2 / len(data)  # Normalize by length to avoid overflow
        
        # Select frequency band
        mask = (np.abs(freqs) >= freq_hz - bandwidth_hz/2) & (np.abs(freqs) <= freq_hz + bandwidth_hz/2)
        band_power = np.sum(power[mask])
        return band_power
    
    def test_fir_filter_frequency_response(self, figure_dir):
        """Test FIRFilter correctly filters frequencies for all btype combinations."""
        fsamp = 1000  # sampling frequency
        
        # Define test cases for different filter types using Hz frequencies
        # For FIRFilter: fp is the passband edge, fs is the stopband edge
        test_cases = [
            {'btype': 'lowpass', 'fp': 100, 'fs': 150, 'low_freq': 50, 'high_freq': 250, 'order': 50},
            {'btype': 'highpass', 'fp': 200, 'fs': 150, 'low_freq': 50, 'high_freq': 250, 'order': 50},
            {'btype': 'bandpass', 'fp': [100, 200], 'fs': [50, 250], 'low_freq': 50, 'mid_freq': 150, 'high_freq': 250, 'order': 50},
            {'btype': 'bandstop', 'fp': [100, 200], 'fs': [140, 160], 'low_freq': 50, 'mid_freq': 150, 'high_freq': 250, 'order': 50},
        ]
        
        for case in test_cases:
            btype = case['btype']
            order = case['order']
            
            # Create multi-component sinusoidal signal
            duration = 10  # seconds
            
            # Use Hz frequencies directly for signal generation
            if btype in ['lowpass', 'highpass']:
                components = [
                    {'frequency': case['low_freq'], 'amplitude': 1.0},
                    {'frequency': case['high_freq'], 'amplitude': 1.0}
                ]
            else:
                components = [
                    {'frequency': case['low_freq'], 'amplitude': 1.0},
                    {'frequency': case['mid_freq'], 'amplitude': 1.0},
                    {'frequency': case['high_freq'], 'amplitude': 1.0}
                ]
            
            signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)
            
            # Compute original power in relevant bands
            if btype == 'lowpass':
                power_pass = self._compute_power_at_frequency(signal, case['low_freq'], fsamp)
                power_stop = self._compute_power_at_frequency(signal, case['high_freq'], fsamp)
            elif btype == 'highpass':
                power_pass = self._compute_power_at_frequency(signal, case['high_freq'], fsamp)
                power_stop = self._compute_power_at_frequency(signal, case['low_freq'], fsamp)
            elif btype == 'bandpass':
                power_pass = self._compute_power_at_frequency(signal, case['mid_freq'], fsamp)
                power_stop_low = self._compute_power_at_frequency(signal, case['low_freq'], fsamp)
                power_stop_high = self._compute_power_at_frequency(signal, case['high_freq'], fsamp)
            elif btype == 'bandstop':
                power_pass_low = self._compute_power_at_frequency(signal, case['low_freq'], fsamp)
                power_pass_high = self._compute_power_at_frequency(signal, case['high_freq'], fsamp)
                power_stop = self._compute_power_at_frequency(signal, case['mid_freq'], fsamp)
            
            # Apply FIR filter with Hz frequencies
            filter_obj = filt.FIRFilter(fp=case['fp'], fs=case['fs'], btype=btype, order=order, att=40)
            filtered = filter_obj(signal)
            if figure_dir:
                times = np.arange(len(signal.p.get_values().ravel())) / fsamp
                save_comparison_figure(figure_dir, f'fir_{btype}_order{order}', times, signal.p.get_values().ravel(), filtered.p.get_values().ravel())
            
            # Compute filtered power and assert correct filtering
            if btype == 'lowpass':
                power_pass_filt = self._compute_power_at_frequency(filtered, case['low_freq'], fsamp)
                power_stop_filt = self._compute_power_at_frequency(filtered, case['high_freq'], fsamp)
                assert power_pass_filt > 0.5 * power_pass, f"Lowpass: pass band power too low"
                assert power_stop_filt < 0.1 * power_stop, f"Lowpass: stop band not attenuated enough"
            elif btype == 'highpass':
                power_pass_filt = self._compute_power_at_frequency(filtered, case['high_freq'], fsamp)
                power_stop_filt = self._compute_power_at_frequency(filtered, case['low_freq'], fsamp)
                assert power_pass_filt > 0.5 * power_pass, f"Highpass: pass band power too low"
                assert power_stop_filt < 0.1 * power_stop, f"Highpass: stop band not attenuated enough"
            elif btype == 'bandpass':
                power_pass_filt = self._compute_power_at_frequency(filtered, case['mid_freq'], fsamp)
                power_stop_low_filt = self._compute_power_at_frequency(filtered, case['low_freq'], fsamp)
                power_stop_high_filt = self._compute_power_at_frequency(filtered, case['high_freq'], fsamp)
                assert power_pass_filt > 0.5 * power_pass, f"Bandpass: pass band power too low"
                assert power_stop_low_filt < 0.1 * power_stop_low, f"Bandpass: low stop band not attenuated"
                assert power_stop_high_filt < 0.1 * power_stop_high, f"Bandpass: high stop band not attenuated"
            elif btype == 'bandstop':
                power_pass_low_filt = self._compute_power_at_frequency(filtered, case['low_freq'], fsamp)
                power_pass_high_filt = self._compute_power_at_frequency(filtered, case['high_freq'], fsamp)
                power_stop_filt = self._compute_power_at_frequency(filtered, case['mid_freq'], fsamp)
                assert power_pass_low_filt > 0.5 * power_pass_low, f"Bandstop: low pass band power too low"
                assert power_pass_high_filt > 0.5 * power_pass_high, f"Bandstop: high pass band power too low"
                assert power_stop_filt < 0.1 * power_stop, f"Bandstop: stop band not attenuated enough"
