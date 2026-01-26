"""
Visualize generated BVP signals to verify waveform characteristics.

This script generates blood pulse signals with various noise conditions
and plots them to verify the waveforms match expected physiological characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyphysio.generators.physiological import BloodVolumePulseGenerator


def generate_bvp_with_ground_truth(duration, sampling_freq, heart_rate=70, 
                                    amplitude=1.0, heart_rate_variability=5, 
                                    noise_std=0, start_time=0):
    """Generate blood pulse signal with known beat times."""
    bvp_signal = BloodVolumePulseGenerator.bvp_signal(
        duration=duration,
        sampling_freq=sampling_freq,
        heart_rate=heart_rate,
        amplitude=amplitude,
        noise_std=noise_std,
        start_time=start_time
    )
    
    # Generate ground truth beat times
    mean_ibi = 60.0 / heart_rate
    beat_times = []
    current_time = 0
    
    while current_time < duration:
        beat_times.append(current_time)
        ibi = np.random.normal(mean_ibi, heart_rate_variability / 60.0)
        ibi = np.clip(ibi, mean_ibi * 0.5, mean_ibi * 1.5)
        current_time += ibi
    
    beat_times = np.array(beat_times)
    beat_times = beat_times[beat_times < duration]
    
    bvp_signal.attrs['beat_times'] = beat_times
    return bvp_signal


# Set up parameters
duration = 10
fsamp = 500
heart_rate = 70

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle('Blood Volume Pulse (BVP) Signal Waveforms - Different Noise Conditions', 
             fontsize=14, fontweight='bold')

# Time vector for plotting
times = np.arange(0, duration, 1/fsamp)

# 1. Clean signal
print("Generating clean BVP signal...")
bvp_clean = generate_bvp_with_ground_truth(
    duration=duration,
    sampling_freq=fsamp,
    heart_rate=heart_rate,
    heart_rate_variability=2,
    noise_std=0
)
axes[0].plot(times, bvp_clean.values, 'b-', linewidth=1)
axes[0].scatter(bvp_clean.attrs['beat_times'], 
               np.interp(bvp_clean.attrs['beat_times'], times, bvp_clean.values),
               color='red', s=30, zorder=5, label='Detected beats')
axes[0].set_ylabel('Amplitude', fontsize=10)
axes[0].set_title('Clean BVP Signal (No Noise)', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 2. With low-frequency noise
print("Generating BVP with low-frequency noise...")
bvp_lowfreq = generate_bvp_with_ground_truth(
    duration=duration,
    sampling_freq=fsamp,
    heart_rate=heart_rate,
    heart_rate_variability=2,
    noise_std=0
)
# Add respiratory noise (~0.25 Hz)
lowfreq_noise = 0.4 * np.sin(2 * np.pi * 0.25 * times)
bvp_lowfreq.values[:] = bvp_lowfreq.values + lowfreq_noise

axes[1].plot(times, bvp_lowfreq.values, 'g-', linewidth=1)
axes[1].scatter(bvp_lowfreq.attrs['beat_times'],
               np.interp(bvp_lowfreq.attrs['beat_times'], times, bvp_lowfreq.values),
               color='red', s=30, zorder=5, label='Ground truth beats')
axes[1].set_ylabel('Amplitude', fontsize=10)
axes[1].set_title('BVP with Low-Frequency Noise (~0.25 Hz - Respiratory)', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 3. With Gaussian noise
print("Generating BVP with Gaussian noise...")
bvp_gaussian = generate_bvp_with_ground_truth(
    duration=duration,
    sampling_freq=fsamp,
    heart_rate=heart_rate,
    heart_rate_variability=2,
    noise_std=0.3  # Gaussian noise
)

axes[2].plot(times, bvp_gaussian.values, 'orange', linewidth=1)
axes[2].scatter(bvp_gaussian.attrs['beat_times'],
               np.interp(bvp_gaussian.attrs['beat_times'], times, bvp_gaussian.values),
               color='red', s=30, zorder=5, label='Ground truth beats')
axes[2].set_ylabel('Amplitude', fontsize=10)
axes[2].set_title('BVP with Gaussian Noise (σ=0.3)', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# 4. With mixed noise (low-freq + Gaussian)
print("Generating BVP with mixed noise...")
bvp_mixed = generate_bvp_with_ground_truth(
    duration=duration,
    sampling_freq=fsamp,
    heart_rate=heart_rate,
    heart_rate_variability=2,
    noise_std=0.15
)
# Add low-frequency noise
lowfreq_noise = 0.3 * np.sin(2 * np.pi * 0.25 * times)
bvp_mixed.values[:] = bvp_mixed.values + lowfreq_noise

axes[3].plot(times, bvp_mixed.values, 'purple', linewidth=1)
axes[3].scatter(bvp_mixed.attrs['beat_times'],
               np.interp(bvp_mixed.attrs['beat_times'], times, bvp_mixed.values),
               color='red', s=30, zorder=5, label='Ground truth beats')
axes[3].set_xlabel('Time (seconds)', fontsize=10)
axes[3].set_ylabel('Amplitude', fontsize=10)
axes[3].set_title('BVP with Mixed Noise (Gaussian σ=0.15 + Low-Freq 0.25 Hz)', 
                  fontsize=11, fontweight='bold')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('/home/bizzego/UniTn/software/pyphysio/bvp_waveforms.png', dpi=150, bbox_inches='tight')
print("Figure saved to: /home/bizzego/UniTn/software/pyphysio/bvp_waveforms.png")

# Display figure
plt.show()

# Print signal statistics
print("\n" + "="*60)
print("BVP Signal Statistics")
print("="*60)

signals = {
    'Clean': bvp_clean,
    'Low-Freq Noise': bvp_lowfreq,
    'Gaussian Noise': bvp_gaussian,
    'Mixed Noise': bvp_mixed
}

for name, signal in signals.items():
    data = signal.values.ravel()
    beat_times = signal.attrs['beat_times']
    print(f"\n{name}:")
    print(f"  Mean:           {np.mean(data):.4f}")
    print(f"  Std Dev:        {np.std(data):.4f}")
    print(f"  Min:            {np.min(data):.4f}")
    print(f"  Max:            {np.max(data):.4f}")
    print(f"  Number of beats: {len(beat_times)}")
    if len(beat_times) > 1:
        ibis = np.diff(beat_times)
        print(f"  Mean IBI:       {np.mean(ibis):.4f}s ({60/np.mean(ibis):.1f} bpm)")
        print(f"  IBI Std Dev:    {np.std(ibis):.4f}s")
