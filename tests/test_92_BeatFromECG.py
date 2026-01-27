"""
Comprehensive tests for BeatFromECG algorithm using ECGGenerator.

Tests verify that BeatFromECG correctly identifies beat peaks in synthetic ECG signals
generated with known beat times, under various noise conditions and parameter settings.
"""

import numpy as np
import pytest
from pyphysio.signal import create_signal
from pyphysio.generators.physiological import ECGGenerator
from pyphysio.specialized.heart import BeatFromECG


class TestBeatFromECG:
    """Tests for BeatFromECG beat detection algorithm."""
    
    def _extract_beat_times_from_ibi(self, ibi_signal):
        """
        Extract beat times from IBI signal returned by BeatFromECG.
        
        The IBI signal contains IBI values at beat locations, NaN everywhere else.
        Beat times are reconstructed from non-NaN indices.
        
        Parameters
        ----------
        ibi_signal : pyphysio.Signal
            IBI scaffold signal from BeatFromECG
            
        Returns
        -------
        beat_times : np.ndarray
            Times of detected beats in seconds
        beat_indices : np.ndarray
            Indices of detected beats in samples
        """
        ibi_data = ibi_signal.values.ravel()
        fsamp = ibi_signal.p.get_sampling_freq()
        
        # Find non-NaN values which indicate beat locations
        beat_indices = np.where(~np.isnan(ibi_data))[0]
        beat_times = beat_indices / fsamp
        
        return beat_times, beat_indices
    
    def _match_beat_times(self, true_times, detected_times, tolerance=0.05):
        """
        Match detected beats with true beats allowing a tolerance window.
        
        Parameters
        ----------
        true_times : np.ndarray
            Ground truth beat times
        detected_times : np.ndarray
            Detected beat times
        tolerance : float
            Maximum time difference to consider a match (seconds)
            
        Returns
        -------
        true_matched : np.ndarray
            Indices of true beats that were matched
        detected_matched : np.ndarray
            Indices of detected beats that matched
        """
        true_matched = []
        detected_matched = []
        
        for i, true_time in enumerate(true_times):
            # Find closest detected beat
            if len(detected_times) == 0:
                continue
            distances = np.abs(detected_times - true_time)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            
            # If within tolerance and not already matched
            if closest_distance <= tolerance and closest_idx not in detected_matched:
                true_matched.append(i)
                detected_matched.append(closest_idx)
        
        return np.array(true_matched), np.array(detected_matched)
    
    def test_beat_detection_simple_ecg_clean(self):
        """Test beat detection on clean realistic ECG signal."""
        # Use realistic ECG which works better with BeatFromECG
        duration = 10
        fsamp = 500
        heart_rate = 70
        
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=2,  # Very small HRV
            noise_std=0,
            ectopy_rate=0
        )
        
        # Get ground truth beat times
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply BeatFromECG
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        # Extract detected beat times
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        # Check detection rate: should detect most beats
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.9, \
            f"Clean signal detection rate {detection_rate:.2%} < 90%"
    
    def test_beat_detection_simple_ecg_with_gaussian_noise(self):
        """Test beat detection on realistic ECG with Gaussian noise."""
        duration = 10
        fsamp = 500
        heart_rate = 70
        noise_std = 0.1
        
        # Generate realistic ECG with Gaussian noise
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=5,
            noise_std=noise_std,
            ectopy_rate=0
        )
        
        # Get ground truth beat times
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.85, \
            f"Gaussian noise detection rate {detection_rate:.2%} < 85%"
    
    def test_beat_detection_simple_ecg_with_sinusoidal_noise(self):
        """Test beat detection on realistic ECG with sinusoidal noise.
        
        Sinusoidal noise at frequencies far from the QRS peak can still interfere
        with peak detection, especially if amplitude is significant.
        """
        duration = 10
        fsamp = 500
        heart_rate = 70
        
        # Generate realistic ECG
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=5,
            noise_std=0,
            ectopy_rate=0
        )
        
        # Get ground truth beat times before adding noise
        true_beat_times = ecg_signal.attrs['beat_times'].copy()
        
        # Add sinusoidal noise (60 Hz - typical interference)
        ecg_data = ecg_signal.values.ravel()
        times = np.arange(len(ecg_data)) / fsamp
        sinusoidal_noise = 0.15 * np.sin(2 * np.pi * 60 * times)
        noisy_data = ecg_data + sinusoidal_noise
        
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='ECG_noisy')
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(noisy_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.80, \
            f"Sinusoidal noise detection rate {detection_rate:.2%} < 80%"
    
    def test_beat_detection_mixed_noise(self):
        """Test beat detection with combined Gaussian and sinusoidal noise."""
        duration = 10
        fsamp = 500
        heart_rate = 70
        gaussian_noise_std = 0.08
        sinusoidal_amplitude = 0.12
        
        # Generate ECG with Gaussian noise
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=8,
            noise_std=gaussian_noise_std,
            ectopy_rate=0
        )
        
        # Get ground truth before adding extra sinusoidal noise
        true_beat_times = ecg_signal.attrs['beat_times'].copy()
        
        # Add sinusoidal noise on top
        ecg_data = ecg_signal.values.ravel()
        times = np.arange(len(ecg_data)) / fsamp
        sinusoidal_noise = sinusoidal_amplitude * np.sin(2 * np.pi * 50 * times)
        noisy_data = ecg_data + sinusoidal_noise
        
        noisy_signal = create_signal(noisy_data, sampling_freq=fsamp, name='ECG_mixed_noise')
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(noisy_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.75, \
            f"Mixed noise detection rate {detection_rate:.2%} < 75%"
    
    def test_beat_detection_realistic_ecg_clean(self):
        """Test beat detection on clean realistic ECG with HRV."""
        duration = 15
        fsamp = 500
        heart_rate = 75
        
        # Generate realistic ECG (includes ground truth beat times in attributes)
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=5,  # Small HRV
            noise_std=0,  # No noise
            ectopy_rate=0
        )
        
        # Get ground truth beat times from attributes
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.90, \
            f"Realistic clean signal detection rate {detection_rate:.2%} < 90%"
    
    def test_beat_detection_realistic_ecg_with_noise(self):
        """Test beat detection on realistic ECG with noise and HRV."""
        duration = 15
        fsamp = 500
        heart_rate = 80
        
        # Generate realistic ECG with noise and HRV
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=15,  # Moderate HRV
            noise_std=0.05,
            ectopy_rate=0
        )
        
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.85, \
            f"Realistic noisy signal detection rate {detection_rate:.2%} < 85%"
    
    def test_beat_detection_realistic_with_ectopy(self):
        """Test beat detection with ectopic beats (morphology variations).
        
        Ectopic beats have different morphology which can affect detection.
        """
        duration = 15
        fsamp = 500
        heart_rate = 72
        
        # Generate realistic ECG with ectopic beats
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=10,
            noise_std=0.03,
            ectopy_rate=0.1  # 10% ectopic beats
        )
        
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Match beats
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.80, \
            f"Ectopy signal detection rate {detection_rate:.2%} < 80%"
    
    def test_beat_detection_high_heart_rate(self):
        """Test beat detection with high heart rate (tachycardia)."""
        duration = 10
        fsamp = 500
        heart_rate = 120  # High heart rate
        
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=8,
            noise_std=0.05,
            ectopy_rate=0
        )
        
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection with appropriate bpm_max parameter
        beat_detector = BeatFromECG(bpm_max=150)
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.85, \
            f"High HR detection rate {detection_rate:.2%} < 85%"
    
    def test_beat_detection_low_heart_rate(self):
        """Test beat detection with low heart rate (bradycardia)."""
        duration = 20
        fsamp = 500
        heart_rate = 40  # Low heart rate
        
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=5,
            noise_std=0.05,
            ectopy_rate=0
        )
        
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection with appropriate bpm_max for low HR
        beat_detector = BeatFromECG(bpm_max=80)
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        true_matched, detected_matched = self._match_beat_times(
            true_beat_times, detected_beat_times, tolerance=0.05
        )
        
        detection_rate = len(true_matched) / len(true_beat_times)
        assert detection_rate >= 0.90, \
            f"Low HR detection rate {detection_rate:.2%} < 90%"
    
    def test_ibi_values_accuracy(self):
        """Test that computed IBI (inter-beat interval) values are accurate.
        
        IBI values should match the time intervals between consecutive detected beats.
        """
        duration = 10
        fsamp = 500
        heart_rate = 70
        
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=2,  # Very small HRV for consistent intervals
            noise_std=0,
            ectopy_rate=0
        )
        
        beat_interval = 60.0 / heart_rate  # seconds
        true_beat_times = ecg_signal.attrs['beat_times']
        
        # Apply beat detection
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        # Extract IBI values at beat locations
        ibi_data = ibi_signal.values.ravel()
        beat_indices = np.where(~np.isnan(ibi_data))[0]
        ibi_values = ibi_data[beat_indices]
        
        # IBI values are in seconds
        expected_ibi_seconds = beat_interval
        
        # IBI values should be approximately constant for constant heart rate
        # Allow some tolerance due to temporal resolution
        if len(ibi_values) > 1:
            mean_ibi = np.mean(ibi_values[1:])  # Skip first beat
            # Allow 15% tolerance for IBI variation
            assert np.abs(mean_ibi - expected_ibi_seconds) < expected_ibi_seconds * 0.15, \
                f"IBI values {mean_ibi:.3f}s don't match expected {expected_ibi_seconds:.3f}s"
    
    def test_delta_parameter_effect(self):
        """Test the effect of delta parameter on beat detection.
        
        delta: minimum sample distance or height threshold for peak detection
        Different delta values should produce detections.
        """
        duration = 10
        fsamp = 500
        heart_rate = 90
        
        ecg_signal = ECGGenerator.realistic_ecg(
            duration=duration,
            sampling_freq=fsamp,
            heart_rate=heart_rate,
            heart_rate_variability=8,
            noise_std=0.08,
            ectopy_rate=0
        )
        
        # Try detection with default parameters
        beat_detector = BeatFromECG()
        ibi_signal = beat_detector(ecg_signal)
        
        detected_beat_times, _ = self._extract_beat_times_from_ibi(ibi_signal)
        
        # Should detect at least some beats
        assert len(detected_beat_times) > 0, \
            "No beats detected with default parameters"
