"""
Physiological Signal Generators

This module provides functions to generate synthetic physiological signals
with realistic characteristics based on published research and models.

Supported signal types:
    - ECG (Electrocardiogram)
    - Respiration
    - EDA (Electrodermal Activity)
    - fNIRS (Functional Near-Infrared Spectroscopy)
    - EEG (Electroencephalography)
    - EMG (Electromyography)

References:
    - McSharry et al. (2003) - ECG synthesis from dynamical models
    - Moody & Mark (2001) - The impact of the MIT-BIH Arrhythmia Database
    - Pereira et al. (2015) - REMOS: A Research Tool for EDA Signal Analysis
"""

import numpy as np
from scipy.signal import sawtooth, fftconvolve
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d
from ..signal import create_signal


class ECGGenerator:
    """
    Generate synthetic ECG signals using dynamical models.
    
    Methods based on:
    McSharry, P. E., Clifford, G. D., Tarassenko, L., & Smith, L. A. (2003).
    "A dynamical model for generating synthetic electrocardiogram signals."
    IEEE Transactions on Biomedical Engineering, 50(3), 289-294.
    """
    
    @staticmethod
    def simple_ecg(duration, sampling_freq, heart_rate=70, 
                  start_time=0, noise_std=0):
        """
        Generate a simple periodic ECG-like signal.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        heart_rate : float, optional
            Heart rate in beats per minute (default: 70)
        start_time : float, optional
            Start time of signal (default: 0)
        noise_std : float, optional
            Standard deviation of additive Gaussian noise (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Synthetic ECG signal with heart_rate and heart_rate_variability attributes
        """
        n_timepoints = int(duration * sampling_freq)
        # Convert heart rate to frequency
        hr_hz = heart_rate / 60.0
        times = np.arange(n_timepoints) / sampling_freq
        
        # Generate QRS-like complex using Gaussian pulses
        # P wave
        p_wave = 0.15 * np.exp(-((times % (1/hr_hz) - 0.05)**2) / 0.001)
        
        # QRS complex (main spike)
        qrs_complex = 1.0 * np.exp(-((times % (1/hr_hz) - 0.12)**2) / 0.0008)
        
        # T wave
        t_wave = 0.3 * np.exp(-((times % (1/hr_hz) - 0.25)**2) / 0.003)
        
        # Combine components
        data = p_wave + qrs_complex + t_wave - 0.5
        
        # Add baseline wander (slow oscillation)
        baseline_wander = 0.3 * np.sin(2 * np.pi * 0.1 * times)
        data = data + baseline_wander
        
        # Add noise
        if noise_std > 0:
            data = data + np.random.normal(0, noise_std, n_timepoints)
        
        signal = create_signal(data, sampling_freq=sampling_freq,
                              start_time=start_time, name='ECG_synthetic')
        # Store generation parameters as attributes
        signal.attrs['heart_rate'] = heart_rate
        signal.attrs['heart_rate_variability'] = 0  # No variability in simple ECG
        signal.attrs['noise_std'] = noise_std
        signal.attrs['signal_type'] = 'ECG'
        return signal
    
    @staticmethod
    def realistic_ecg(duration, sampling_freq, heart_rate=70,
                     heart_rate_variability=10, noise_std=0.05,
                     start_time=0, ectopy_rate=0):
        """
        Generate more realistic ECG with heart rate variability and arrhythmias.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        heart_rate : float, optional
            Mean heart rate in bpm (default: 70)
        heart_rate_variability : float, optional
            Standard deviation of RR intervals in ms (default: 10)
        noise_std : float, optional
            Standard deviation of noise (default: 0.05)
        start_time : float, optional
            Start time of signal (default: 0)
        ectopy_rate : float, optional
            Fraction of beats that are ectopic (0-1) (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Realistic synthetic ECG signal with generation parameters stored as attributes
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        duration = n_timepoints / sampling_freq
        hr_hz = heart_rate / 60.0
        
        # Generate RR intervals with variability
        n_beats = int(duration * hr_hz) + 2
        rr_intervals = np.random.normal(1/hr_hz, 
                                       heart_rate_variability/1000.0,
                                       n_beats)
        rr_intervals = np.abs(rr_intervals)  # Ensure positive
        
        # Generate beat times
        beat_times = np.cumsum(rr_intervals)
        beat_times = beat_times[beat_times < duration]
        
        # Generate ECG at beat times
        data = np.zeros(n_timepoints)
        
        for i, beat_time in enumerate(beat_times):
            if np.random.random() < ectopy_rate:
                # Ectopic beat - different morphology
                amplitude = 0.7
                width_factor = 0.8
            else:
                amplitude = 1.0
                width_factor = 1.0
            
            # QRS complex
            qrs_idx = int(beat_time * sampling_freq)
            if 0 <= qrs_idx < n_timepoints:
                # QRS duration ~100ms
                qrs_width = int(0.1 * width_factor * sampling_freq)
                qrs_range = slice(max(0, qrs_idx - qrs_width//2),
                                 min(n_timepoints, qrs_idx + qrs_width//2))
                data[qrs_range] += amplitude * np.sin(np.linspace(0, np.pi, 
                                                                  len(data[qrs_range])))
        
        # Add baseline wander
        baseline_wander = 0.3 * np.sin(2 * np.pi * 0.1 * times)
        baseline_wander += 0.1 * np.sin(2 * np.pi * 0.05 * times)
        data = data + baseline_wander
        
        # Add noise
        data = data + np.random.normal(0, noise_std, n_timepoints)
        
        signal = create_signal(data, sampling_freq=sampling_freq,
                              start_time=start_time, name='ECG_realistic')
        # Store generation parameters as attributes
        signal.attrs['heart_rate'] = heart_rate
        signal.attrs['heart_rate_variability'] = heart_rate_variability
        signal.attrs['noise_std'] = noise_std
        signal.attrs['ectopy_rate'] = ectopy_rate
        signal.attrs['signal_type'] = 'ECG'
        signal.attrs['beat_times'] = beat_times  # Ground truth beat locations
        return signal
    
    @staticmethod
    def load_real_signal(return_signal=True):
        """
        Load real ECG data from test dataset.
        
        Parameters
        ----------
        return_signal : bool, optional
            If True, return as Signal object; if False, return raw values (default: True)
            
        Returns
        -------
        signal : xarray.DataArray or ndarray
            Real ECG signal sampled at 2048 Hz, ~120 seconds duration from medical test dataset
        """
        from .. import TestData
        return TestData.ecg(return_signal=return_signal)


class RespirationGenerator:
    """
    Generate synthetic respiration signals based on physiological models.
    """
    
    @staticmethod
    def sinusoidal_respiration(duration, sampling_freq, 
                              respiration_rate=15, amplitude=1.0,
                              start_time=0, noise_std=0):
        """
        Generate simple sinusoidal respiration signal.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        respiration_rate : float, optional
            Respiration rate in breaths per minute (default: 15)
        amplitude : float, optional
            Amplitude of respiration (default: 1.0)
        start_time : float, optional
            Start time of signal (default: 0)
        noise_std : float, optional
            Standard deviation of noise (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Synthetic respiration signal with generation parameters as attributes
        """
        n_timepoints = int(duration * sampling_freq)
        rr_hz = respiration_rate / 60.0
        times = np.arange(n_timepoints) / sampling_freq
        
        # Simple sine wave
        data = amplitude * np.sin(2 * np.pi * rr_hz * times)
        
        # Add noise
        if noise_std > 0:
            data = data + np.random.normal(0, noise_std, n_timepoints)
        
        signal = create_signal(data, sampling_freq=sampling_freq,
                              start_time=start_time, name='respiration_sine')
        signal.attrs['respiration_rate'] = respiration_rate
        signal.attrs['amplitude'] = amplitude
        signal.attrs['noise_std'] = noise_std
        signal.attrs['signal_type'] = 'respiration'
        signal.attrs['fundamental_frequency'] = rr_hz
        return signal
    
    @staticmethod
    def realistic_respiration(duration, sampling_freq,
                             respiration_rate=15, amplitude=1.0,
                             inspiration_expiration_ratio=0.4,
                             respiratory_sinus_arrhythmia=False,
                             start_time=0, noise_std=0):
        """
        Generate realistic respiration with asymmetric inspiration/expiration.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        respiration_rate : float, optional
            Respiration rate in breaths per minute (default: 15)
        amplitude : float, optional
            Amplitude of respiration (default: 1.0)
        inspiration_expiration_ratio : float, optional
            Ratio of inspiration to total cycle (default: 0.4, i.e., 40% inspiration)
        respiratory_sinus_arrhythmia : bool, optional
            Add respiratory sinus arrhythmia modulation (default: False)
        start_time : float, optional
            Start time of signal (default: 0)
        noise_std : float, optional
            Standard deviation of noise (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Realistic respiration signal
        """
        n_timepoints = int(duration * sampling_freq)
        rr_hz = respiration_rate / 60.0
        times = np.arange(n_timepoints) / sampling_freq
        
        # Create sawtooth wave with asymmetric inspiration/expiration
        cycle_duration = 1 / rr_hz
        phase = (times % cycle_duration) / cycle_duration
        
        # Asymmetric rise and fall
        data = np.where(phase < inspiration_expiration_ratio,
                       amplitude * (phase / inspiration_expiration_ratio),
                       amplitude * ((1 - phase) / (1 - inspiration_expiration_ratio)))
        
        # Add respiratory sinus arrhythmia if requested
        if respiratory_sinus_arrhythmia:
            # Heart rate modulation by respiration
            rsa_modulation = 0.1 * amplitude * np.sin(2 * np.pi * rr_hz * times)
            data = data + rsa_modulation
        
        # Add noise
        if noise_std > 0:
            data = data + np.random.normal(0, noise_std, n_timepoints)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='respiration_realistic')
    
    @staticmethod
    def load_real_signal(return_signal=True):
        """
        Load real respiration data from test dataset.
        
        Parameters
        ----------
        return_signal : bool, optional
            If True, return as Signal object; if False, return raw values (default: True)
            
        Returns
        -------
        signal : xarray.DataArray or ndarray
            Real respiration signal sampled at 2048 Hz, ~120 seconds duration
        """
        from .. import TestData
        return TestData.resp(return_signal=return_signal)

class EDAGenerator:
    """
    Generate synthetic Electrodermal Activity (skin conductance) signals.
    
    EDA consists of two components:
    - Skin Conductance Level (SCL): Slow tonic component
    - Skin Conductance Response (SCR): Fast phasic component
    """
    
    @staticmethod
    def eda_phasic_component(duration, sampling_freq, stimulus_times,
                            response_magnitude=0.5, t1=1.0, t2=10.0,
                            start_time=0):
        """
        Generate EDA phasic component using Bateman function convolution.
        
        Uses a delta impulse train convolved with a Bateman (biexponential) function.
        The Bateman function models the pharmacokinetic response commonly used in EDA analysis.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        stimulus_times : list of float
            Times (in seconds) when stimuli occur
        response_magnitude : float or list, optional
            Magnitude of phasic response in microSiemens (default: 0.5)
        t1 : float, optional
            Rise time constant in seconds (default: 1.0). Controls how quickly the response rises.
        t2 : float, optional
            Recovery time constant in seconds (default: 10.0). Controls how slowly the response decays.
            Must be greater than t1 for realistic response shape.
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Phasic EDA component
            
        Notes
        -----
        The Bateman function is defined as:
            f(t) = (k / (t2 - t1)) * (exp(-t1*t) - exp(-t2*t))
        where k is a scaling constant related to response_magnitude.
        """
        from scipy.signal import convolve
        
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        
        # Ensure response_magnitude is a list
        if np.isscalar(response_magnitude):
            response_magnitude = [response_magnitude] * len(stimulus_times)
        
        # Generate impulse train (delta function at stimulus times with magnitudes)
        impulse_signal = np.zeros(n_timepoints)
        for stim_time, magnitude in zip(stimulus_times, response_magnitude):
            idx = int((stim_time - start_time) * sampling_freq)
            if 0 <= idx < n_timepoints:
                impulse_signal[idx] = magnitude
        
        # Ensure t2 > t1 for realistic response shape
        if t2 <= t1:
            t2 = t1 + 1.0
        
        # Generate Bateman function (biexponential impulse response)
        # Duration of impulse response: cover up to ~5 time constants of recovery
        ir_duration = max(5 * t2, 50)  # At least 50 seconds
        ir_samples = int(ir_duration * sampling_freq)
        ir_times = np.arange(ir_samples) / sampling_freq
        
        # Bateman function: biexponential response
        # Form: f(t) = (k / (t2 - t1)) * (exp(-t/t1) - exp(-t/t2))
        # This produces a characteristic rise-and-fall curve
        # For t1 < t2: exponential with fast rise, slow decay
        bateman = (1.0 / (t2 - t1)) * (np.exp(-ir_times / t2) - np.exp(-ir_times / t1))
        
        # Take absolute value to ensure positive response
        bateman = np.abs(bateman)
        
        # Normalize to unit peak to ensure consistent scaling
        peak_val = np.max(bateman)
        if peak_val > 0:
            bateman = bateman / peak_val
        else:
            # Fallback: if somehow we get all zeros, create a simple exponential decay
            bateman = np.exp(-ir_times / ((t1 + t2) / 2.0))
            bateman = bateman / np.max(bateman)
        
        # Convolve impulse train with Bateman function
        # Use 'same' mode to keep output same size as input
        phasic = convolve(impulse_signal, bateman, mode='same')
        
        # Truncate to exact duration (in case of edge effects)
        phasic = phasic[:n_timepoints]
        
        return create_signal(phasic, sampling_freq=sampling_freq,
                            start_time=start_time, name='EDA_phasic')
    
    @staticmethod
    def realistic_eda(duration, sampling_freq, 
                     base_scl=2.0, stimulus_times=None,
                     stimulus_magnitudes=None, noise_std=0.01,
                     t1=1.0, t2=10.0,
                     start_time=0):
        """
        Generate realistic EDA signal (SCL + SCR).
        
        Uses Bateman function (biexponential) to model the phasic component response.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        base_scl : float, optional
            Base skin conductance level (default: 2.0)
        stimulus_times : list of float, optional
            Times when stimuli occur (default: None)
        stimulus_magnitudes : list of float, optional
            Magnitudes of phasic responses (default: None)
        noise_std : float, optional
            Standard deviation of noise (default: 0.01)
        t1 : float, optional
            Rise time constant in seconds (default: 1.0)
        t2 : float, optional
            Recovery time constant in seconds (default: 10.0)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Realistic EDA signal (baseline + phasic component)
        """
        n_timepoints = int(duration * sampling_freq)
        
        # Generate baseline level
        baseline = np.ones(n_timepoints) * base_scl
        
        # Generate phasic component
        if stimulus_times is None:
            stimulus_times = []
        if stimulus_magnitudes is None:
            stimulus_magnitudes = [0.5] * len(stimulus_times)
        
        phasic = EDAGenerator.eda_phasic_component(duration, sampling_freq,
                                                  stimulus_times,
                                                  response_magnitude=stimulus_magnitudes,
                                                  t1=t1, t2=t2,
                                                  start_time=start_time)
        
        # Combine components
        data = baseline + phasic.values
        
        # Add noise
        n_timepoints = int(duration * sampling_freq)
        data = data + np.random.normal(0, noise_std, n_timepoints)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='EDA_realistic')
    
    @staticmethod
    def load_real_signal(return_signal=True):
        """
        Load real EDA data from test dataset.
        
        Parameters
        ----------
        return_signal : bool, optional
            If True, return as Signal object; if False, return raw values (default: True)
            
        Returns
        -------
        signal : xarray.DataArray or ndarray
            Real EDA signal sampled at 2048 Hz, ~120 seconds duration
        """
        from .. import TestData
        return TestData.eda(return_signal=return_signal)


class fNIRSGenerator:
    """
    Generate synthetic functional Near-Infrared Spectroscopy signals.
    
    fNIRS measures hemoglobin concentration changes (HbO and HbR)
    in response to neural activity.
    """
    
    @staticmethod
    def hbo_hbr_response(duration, sampling_freq, 
                        stimulus_times, stimulus_duration=5.0,
                        hbo_magnitude=2.0, hbr_magnitude=-1.0,
                        rise_time=1.0, peak_time=5.0, recovery_time=10.0,
                        start_time=0, noise_std=0.1):
        """
        Generate realistic HbO and HbR responses to stimuli.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        stimulus_times : list of float
            Onset times of stimuli in seconds
        stimulus_duration : float, optional
            Duration of stimuli in seconds (default: 5.0)
        hbo_magnitude : float, optional
            Amplitude of HbO response in µM (default: 2.0)
        hbr_magnitude : float, optional
            Amplitude of HbR response in µM (negative) (default: -1.0)
        rise_time : float, optional
            Time to rise in seconds (default: 1.0)
        peak_time : float, optional
            Time to peak in seconds (default: 5.0)
        recovery_time : float, optional
            Recovery time constant in seconds (default: 10.0)
        start_time : float, optional
            Start time of signal (default: 0)
        noise_std : float, optional
            Standard deviation of noise (default: 0.1)
            
        Returns
        -------
        hbo : xarray.DataArray
            HbO signal
        hbr : xarray.DataArray
            HbR signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        hbo_data = np.zeros(n_timepoints)
        hbr_data = np.zeros(n_timepoints)
        
        for stim_time in stimulus_times:
            stim_idx = int((stim_time - start_time) * sampling_freq)
            
            for idx in range(stim_idx, n_timepoints):
                time_since_stim = (idx - stim_idx) / sampling_freq
                
                # Hemodynamic response function (HRF) approximation
                # Rise phase
                if time_since_stim < peak_time:
                    phase = time_since_stim / peak_time
                    hrf = phase ** 2 * np.exp(-phase)
                else:
                    # Decay phase
                    hrf = np.exp(-(time_since_stim - peak_time) / recovery_time)
                
                hbo_data[idx] += hbo_magnitude * hrf
                hbr_data[idx] += hbr_magnitude * hrf
        
        # Add noise
        hbo_data = hbo_data + np.random.normal(0, noise_std, n_timepoints)
        hbr_data = hbr_data + np.random.normal(0, noise_std, n_timepoints)
        
        # Add baseline
        hbo_baseline = 100.0
        hbr_baseline = 50.0
        hbo_data = hbo_data + hbo_baseline
        hbr_data = hbr_data + hbr_baseline
        
        hbo_signal = create_signal(hbo_data, sampling_freq=sampling_freq,
                                  start_time=start_time, name='fNIRS_HbO')
        hbr_signal = create_signal(hbr_data, sampling_freq=sampling_freq,
                                  start_time=start_time, name='fNIRS_HbR')
        
        return hbo_signal, hbr_signal
    
    @staticmethod
    def load_real_signal(task='rest', return_signal=True):
        """
        Load real fNIRS data from test dataset.
        
        Parameters
        ----------
        task : str, optional
            Task type - 'rest' for resting state or 'tapping' for motor tapping task (default: 'rest')
        return_signal : bool, optional
            If True, return as Signal object; if False, return raw values (default: True)
            
        Returns
        -------
        signal : xarray.DataArray or ndarray
            Real fNIRS signal with multiple optodes/wavelengths
        """
        from .. import TestData
        if task == 'tapping':
            return TestData.tapping(return_signal=return_signal)
        else:
            return TestData.fnirs(return_signal=return_signal)


class EEGGenerator:
    """
    Generate synthetic EEG signals with characteristic frequency bands.
    """
    
    @staticmethod
    def eeg_with_bands(duration, sampling_freq, 
                      delta_power=0.5, theta_power=1.0,
                      alpha_power=2.0, beta_power=1.0,
                      gamma_power=0.5, noise_std=0.5,
                      start_time=0):
        """
        Generate EEG signal with specified power in different frequency bands.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        delta_power : float, optional
            Power in delta band (0.5-4 Hz) (default: 0.5)
        theta_power : float, optional
            Power in theta band (4-8 Hz) (default: 1.0)
        alpha_power : float, optional
            Power in alpha band (8-12 Hz) (default: 2.0)
        beta_power : float, optional
            Power in beta band (12-30 Hz) (default: 1.0)
        gamma_power : float, optional
            Power in gamma band (30-100 Hz) (default: 0.5)
        noise_std : float, optional
            Standard deviation of background noise (default: 0.5)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Synthetic EEG signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        data = np.zeros(n_timepoints)
        
        # Define frequency bands
        bands = [
            (0.5, 4, delta_power, "delta"),
            (4, 8, theta_power, "theta"),
            (8, 12, alpha_power, "alpha"),
            (12, 30, beta_power, "beta"),
            (30, 100, gamma_power, "gamma"),
        ]
        
        # Generate each band
        for f_low, f_high, power, band_name in bands:
            # Random frequency within band
            freq = np.random.uniform(f_low, f_high)
            amplitude = np.sqrt(power)
            
            # Generate component
            component = amplitude * np.sin(2 * np.pi * freq * times)
            
            # Apply envelope (slow amplitude modulation)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * times)
            data = data + component * envelope
        
        # Add noise
        data = data + np.random.normal(0, noise_std, n_timepoints)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='EEG_synthetic')
    
    @staticmethod
    def eeg_alpha_burst(duration, sampling_freq,
                       burst_times=None, burst_duration=2.0,
                       alpha_freq=10.0, alpha_amplitude=1.0,
                       background_power=0.5, noise_std=0.5,
                       start_time=0):
        """
        Generate EEG with alpha bursts (typical in relaxation).
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        burst_times : list of float, optional
            Times when alpha bursts occur (default: None)
        burst_duration : float, optional
            Duration of each burst in seconds (default: 2.0)
        alpha_freq : float, optional
            Alpha band frequency in Hz (default: 10.0)
        alpha_amplitude : float, optional
            Amplitude of alpha activity (default: 1.0)
        background_power : float, optional
            Power of background activity (default: 0.5)
        noise_std : float, optional
            Standard deviation of noise (default: 0.5)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            EEG signal with alpha bursts
        """
        n_timepoints = int(duration * sampling_freq)
        if burst_times is None:
            burst_times = []
        
        times = np.arange(n_timepoints) / sampling_freq
        data = np.zeros(n_timepoints)
        
        # Background activity (mixed frequencies)
        background = EEGGenerator.eeg_with_bands(
            duration=duration, 
            sampling_freq=sampling_freq,
            delta_power=background_power,
            theta_power=background_power,
            alpha_power=0.1,  # Low alpha in background
            beta_power=background_power,
            gamma_power=0,
            start_time=start_time
        )
        data = data + background.values
        
        # Add alpha bursts
        for burst_time in burst_times:
            burst_idx = int((burst_time - start_time) * sampling_freq)
            burst_end = burst_idx + int(burst_duration * sampling_freq)
            burst_end = min(burst_end, n_timepoints)
            
            burst_times_arr = np.arange(burst_end - burst_idx) / sampling_freq
            alpha_component = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * burst_times_arr)
            
            # Smooth onset and offset
            envelope = np.ones(burst_end - burst_idx)
            fade_samples = int(0.5 * sampling_freq)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            data[burst_idx:burst_end] += alpha_component * envelope
        
        # Add noise
        data = data + np.random.normal(0, noise_std, n_timepoints)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='EEG_alpha_bursts')


class EMGGenerator:
    """
    Generate synthetic Electromyography signals.
    """
    
    @staticmethod
    def emg_at_rest(duration, sampling_freq, 
                   baseline_activity=0.05, noise_std=0.1,
                   start_time=0):
        """
        Generate resting EMG with minimal activity.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        baseline_activity : float, optional
            Baseline motor unit activity level (default: 0.05)
        noise_std : float, optional
            Standard deviation of noise (default: 0.1)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Resting EMG signal
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        
        # Random motor unit action potentials
        n_muaps = 5
        data = np.zeros(n_timepoints)
        
        for _ in range(n_muaps):
            muap_freq = np.random.uniform(8, 12)  # 8-12 Hz motor unit
            muap_amplitude = baseline_activity / n_muaps
            data += muap_amplitude * np.sin(2 * np.pi * muap_freq * times)
        
        # Add noise
        data = data + np.random.normal(0, noise_std, n_timepoints)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='EMG_rest')
    
    @staticmethod
    def emg_during_contraction(duration, sampling_freq,
                             contraction_times=None,
                             contraction_force=0.5,
                             start_time=0, noise_std=0.1):
        """
        Generate EMG during muscle contraction.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        contraction_times : list of tuples, optional
            List of (start_time, end_time) for contractions (default: None)
        contraction_force : float, optional
            Normalized force (0-1) (default: 0.5)
        start_time : float, optional
            Start time of signal (default: 0)
        noise_std : float, optional
            Standard deviation of noise (default: 0.1)
            
        Returns
        -------
        signal : xarray.DataArray
            EMG signal during contraction
        """
        n_timepoints = int(duration * sampling_freq)
        if contraction_times is None:
            contraction_times = []
        
        times = np.arange(n_timepoints) / sampling_freq
        data = np.zeros(n_timepoints)
        
        # Get contraction intervals
        contraction_mask = np.zeros(n_timepoints, dtype=bool)
        for cont_start, cont_end in contraction_times:
            start_idx = int((cont_start - start_time) * sampling_freq)
            end_idx = int((cont_end - start_time) * sampling_freq)
            contraction_mask[max(0, start_idx):min(n_timepoints, end_idx)] = True
        
        # Generate motor units with rate coding
        n_muaps = 10
        for i in range(n_muaps):
            # MU frequency increases with force
            muap_freq_min = 8 + i
            muap_freq_max = 15 + i * 2
            
            for idx in range(n_timepoints):
                if contraction_mask[idx]:
                    muap_freq = muap_freq_min + (muap_freq_max - muap_freq_min) * contraction_force
                else:
                    muap_freq = muap_freq_min
                
                amplitude = contraction_force * 0.1 / n_muaps
                data[idx] += amplitude * np.sin(2 * np.pi * muap_freq * times[idx])
        
        # Add noise
        data = data + np.random.normal(0, noise_std, n_timepoints)
        
        return create_signal(data, sampling_freq=sampling_freq,
                            start_time=start_time, name='EMG_contraction')


class BloodVolumePulseGenerator:
    """
    Generate synthetic Blood Volume Pulse (BVP) and Pulse Oximetry signals.
    
    BVP signals measure changes in blood volume in tissue (typically from fingers/wrist).
    Pulse Oximetry measures oxygen saturation (SpO2) through light absorption.
    """
    
    @staticmethod
    def bvp_signal(duration, sampling_freq, heart_rate=70, 
                  amplitude=1.0, heart_rate_variability=5,
                  noise_std=0.0, start_time=0):
        """
        Generate a realistic Blood Volume Pulse signal using dual Gaussian model.
        
        The signal is generated as a sum of two Gaussian functions per beat:
        - First Gaussian: Sharp systolic peak (main pulse)
        - Second Gaussian: Broader diastolic peak (reflected wave)
        
        This creates the characteristic morphology with systolic peak, dicrotic notch,
        and diastolic peak observed in real photoplethysmography (PPG) signals.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        heart_rate : float, optional
            Heart rate in beats per minute (default: 70)
        amplitude : float, optional
            Pulse amplitude controlling systolic peak height (default: 1.0)
        heart_rate_variability : float, optional
            Heart rate variability in bpm (default: 5). Creates natural variation in beat intervals.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise (default: 0.0)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Realistic BVP signal with dual-Gaussian pulse morphology
            
        Notes
        -----
        The implementation uses two Gaussian functions to model each cardiac pulse:
        - Systolic Gaussian: peaks around 35% of beat cycle, narrow (sharp rise)
        - Diastolic Gaussian: peaks around 75% of beat cycle, wider (smooth decline with reflection)
        
        References:
            Biswas et al. (2020). "PhysioNet/Computing in Cardiology Challenge 2015: 
            Reducing False Arrhythmias Alarms in the ICU". Scientific Reports, 10, 8274.
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        
        # Convert heart rate to base frequency
        hr_hz = heart_rate / 60.0
        base_period = 1 / hr_hz
        
        # Generate beat intervals with heart rate variability
        beat_times = []
        current_time = 0
        while current_time < duration:
            beat_times.append(current_time)
            # Add HRV: normal distribution with mean=base_period, std based on HRV parameter
            ibi = np.random.normal(base_period, heart_rate_variability / 60.0)
            ibi = np.clip(ibi, base_period * 0.5, base_period * 1.5)  # Clip to reasonable range
            current_time += ibi
        
        beat_times = np.array(beat_times)
        beat_times = beat_times[beat_times < duration]
        
        # Initialize BVP signal
        bvp = np.zeros(n_timepoints)
        
        # Generate each individual pulse using dual Gaussian model
        for beat_idx, beat_time in enumerate(beat_times):
            # Get duration of this beat cycle
            if beat_idx < len(beat_times) - 1:
                cycle_duration = beat_times[beat_idx + 1] - beat_time
            else:
                cycle_duration = base_period
            
            # Find indices for this beat cycle
            beat_start_idx = int(beat_time * sampling_freq)
            beat_end_idx = int((beat_time + cycle_duration) * sampling_freq)
            beat_end_idx = min(beat_end_idx, n_timepoints)
            
            cycle_length = beat_end_idx - beat_start_idx
            if cycle_length <= 0:
                continue
            
            # Time within beat cycle [0, cycle_duration]
            cycle_times = np.arange(cycle_length) / sampling_freq
            
            # === FIRST GAUSSIAN: Systolic Peak ===
            # Main systolic surge with wider width
            # Peaks around 35% of the cardiac cycle
            systolic_mean = 0.35 * cycle_duration
            systolic_std = 0.085 * cycle_duration  # Larger standard deviation
            systolic_amplitude = amplitude  # Full amplitude
            
            systolic_peak = (systolic_amplitude * 
                           np.exp(-0.5 * ((cycle_times - systolic_mean) / systolic_std) ** 2))
            
            # === SECOND GAUSSIAN: Diastolic Peak ===
            # Lower amplitude peak representing reflected wave
            # Peaks closer to systolic peak, around 55% of the cardiac cycle
            diastolic_mean = 0.55 * cycle_duration
            diastolic_std = 0.08 * cycle_duration  # Similar width to systolic
            diastolic_amplitude = 0.35 * amplitude  # About 35% of systolic peak
            
            diastolic_peak = (diastolic_amplitude * 
                            np.exp(-0.5 * ((cycle_times - diastolic_mean) / diastolic_std) ** 2))
            
            # Combine the two Gaussians
            pulse = systolic_peak + diastolic_peak
            
            # Add pulse to BVP signal
            bvp[beat_start_idx:beat_end_idx] += pulse
        
        # Add baseline component (DC offset with slow respiratory modulation)
        baseline = 10.0 + 0.3 * np.sin(2 * np.pi * 0.15 * times)  # ~9 breaths/min
        bvp = baseline + bvp
        
        # Add noise if specified
        if noise_std > 0:
            bvp = bvp + np.random.normal(0, noise_std, n_timepoints)
        
        signal = create_signal(bvp, sampling_freq=sampling_freq,
                              start_time=start_time, name='BVP')
        signal.attrs['heart_rate'] = heart_rate
        signal.attrs['heart_rate_variability'] = heart_rate_variability
        signal.attrs['amplitude'] = amplitude
        signal.attrs['noise_std'] = noise_std
        signal.attrs['signal_type'] = 'BVP'
        signal.attrs['beat_times'] = beat_times
        return signal
    
    @staticmethod
    def pulse_oximetry_signal(duration, sampling_freq, heart_rate=70,
                            spo2=98.0, noise_std=0.3, start_time=0):
        """
        Generate a Pulse Oximetry signal (SpO2).
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        heart_rate : float, optional
            Heart rate in beats per minute (default: 70)
        spo2 : float, optional
            Oxygen saturation percentage (default: 98.0)
        noise_std : float, optional
            Standard deviation of noise (default: 0.3)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            SpO2 signal with cardiac pulsation
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        
        # Convert heart rate to frequency
        hr_hz = heart_rate / 60.0
        
        # Base SpO2 with slow drift
        spo2_base = spo2 + 0.5 * np.sin(2 * np.pi * 0.01 * times)
        
        # Add cardiac pulsation (small modulation at heart rate frequency)
        cardiac_modulation = 1.5 * np.sin(2 * np.pi * hr_hz * times)
        
        # Combine
        spo2_signal = spo2_base + cardiac_modulation
        
        # Add noise
        if noise_std > 0:
            spo2_signal = spo2_signal + np.random.normal(0, noise_std, n_timepoints)
        
        # Clip to valid range [0, 100]
        spo2_signal = np.clip(spo2_signal, 0, 100)
        
        signal = create_signal(spo2_signal, sampling_freq=sampling_freq,
                              start_time=start_time, name='SpO2')
        signal.attrs['heart_rate'] = heart_rate
        signal.attrs['spo2_baseline'] = spo2
        signal.attrs['noise_std'] = noise_std
        signal.attrs['signal_type'] = 'Pulse_Oximetry'
        return signal
    
    @staticmethod
    def realistic_pulse_oximetry(duration, sampling_freq, heart_rate=70,
                                spo2_events=None, event_severity=0.05,
                                noise_std=0.3, start_time=0):
        """
        Generate realistic pulse oximetry with potential desaturation events.
        
        Parameters
        ----------
        duration : float
            Duration of signal in seconds
        sampling_freq : float
            Sampling frequency in Hz
        heart_rate : float, optional
            Heart rate in beats per minute (default: 70)
        spo2_events : list of tuples, optional
            List of (start_time, end_time, depth) for desaturation events
            depth is the SpO2 drop (e.g., 0.05 for 5% drop) (default: None)
        event_severity : float, optional
            Default severity of desaturation events (default: 0.05)
        noise_std : float, optional
            Standard deviation of noise (default: 0.3)
        start_time : float, optional
            Start time of signal (default: 0)
            
        Returns
        -------
        signal : xarray.DataArray
            Realistic SpO2 signal with potential desaturation events
        """
        n_timepoints = int(duration * sampling_freq)
        times = np.arange(n_timepoints) / sampling_freq
        
        # Generate baseline SpO2
        hr_hz = heart_rate / 60.0
        spo2_base = 98.0 + 0.5 * np.sin(2 * np.pi * 0.01 * times)
        cardiac_modulation = 1.5 * np.sin(2 * np.pi * hr_hz * times)
        spo2_signal = spo2_base + cardiac_modulation
        
        # Add desaturation events
        if spo2_events is None:
            spo2_events = []
        
        for event_start, event_end, depth in spo2_events:
            start_idx = int((event_start - start_time) * sampling_freq)
            end_idx = int((event_end - start_time) * sampling_freq)
            
            if 0 <= start_idx < n_timepoints:
                # Create smooth desaturation curve
                event_duration = end_idx - start_idx
                event_times = np.arange(event_duration) / event_duration
                
                # Smooth envelope for event
                event_envelope = np.sin(np.pi * event_times) ** 2
                spo2_drop = depth * 100 * event_envelope
                
                spo2_signal[start_idx:min(end_idx, n_timepoints)] -= spo2_drop[:min(event_duration, n_timepoints - start_idx)]
        
        # Add noise
        if noise_std > 0:
            spo2_signal = spo2_signal + np.random.normal(0, noise_std, n_timepoints)
        
        # Clip to valid range
        spo2_signal = np.clip(spo2_signal, 0, 100)
        
        signal = create_signal(spo2_signal, sampling_freq=sampling_freq,
                              start_time=start_time, name='SpO2_realistic')
        signal.attrs['heart_rate'] = heart_rate
        signal.attrs['noise_std'] = noise_std
        signal.attrs['n_desaturation_events'] = len(spo2_events)
        signal.attrs['signal_type'] = 'Pulse_Oximetry_Realistic'
        return signal
    
    @staticmethod
    def load_real_signal(return_signal=True):
        """
        Load real BVP data from test dataset.
        
        Parameters
        ----------
        return_signal : bool, optional
            If True, return as Signal object; if False, return raw values (default: True)
            
        Returns
        -------
        signal : xarray.DataArray or ndarray
            Real BVP signal sampled at 2048 Hz, ~120 seconds duration
        """
        from .. import TestData
        return TestData.bvp(return_signal=return_signal)


class RealTestDataGenerator:
    """
    Convenience class for loading multimodal real physiological test data.
    
    Individual signal types can be loaded directly from their respective generators:
    - ECGGenerator.load_real_signal()
    - EDAGenerator.load_real_signal()
    - RespirationGenerator.load_real_signal()
    - BloodVolumePulseGenerator.load_real_signal()
    - fNIRSGenerator.load_real_signal(task='rest' or 'tapping')
    
    This class provides a convenience method to load all available multimodal signals at once.
    """
    
    @staticmethod
    def load_multimodal(return_signals=True):
        """
        Load all available multimodal signals from medical dataset.
        
        Parameters
        ----------
        return_signals : bool, optional
            If True, return as Signal objects; if False, return raw values (default: True)
            
        Returns
        -------
        ecg : xarray.DataArray or ndarray
            ECG signal
        eda : xarray.DataArray or ndarray
            EDA signal
        bvp : xarray.DataArray or ndarray
            BVP signal
        resp : xarray.DataArray or ndarray
            Respiration signal
        """
        ecg = ECGGenerator.load_real_signal(return_signal=return_signals)
        eda = EDAGenerator.load_real_signal(return_signal=return_signals)
        bvp = BloodVolumePulseGenerator.load_real_signal(return_signal=return_signals)
        resp = RespirationGenerator.load_real_signal(return_signal=return_signals)
        
        return ecg, eda, bvp, resp