import numpy as np
from pyphysio.generators.fundamental import SinusoidalGenerator
import pyphysio.filters as filt

# Test bandstop filter
fsamp = 1000
duration = 10

# Create signal with 50, 150, and 250 Hz components
components = [
    {'frequency': 50, 'amplitude': 1.0},   # 50 Hz - should pass
    {'frequency': 150, 'amplitude': 1.0},  # 150 Hz - should be stopped
    {'frequency': 250, 'amplitude': 1.0}   # 250 Hz - should pass
]

signal = SinusoidalGenerator.multi_component_sine(duration, fsamp, components)

print(f"Original signal shape: {signal.data.shape}")
print(f"Original signal stats: min={np.min(signal.data)}, max={np.max(signal.data)}")

# Test bandstop: fp=[140, 160] means pass band edges, fs=[100, 200] means stop band edges
# This should stop frequencies between 100-200 Hz and pass others
filter_obj = filt.IIRFilter(fp=[140, 160], fs=[100, 200], btype='bandstop',
                            ftype='butter', order=5, loss=0.1, att=40)
try:
    filtered = filter_obj(signal)
    print(f"\nBandstop filter [fp=[140,160], fs=[100,200]]:")
    print(f"Filtered signal stats: min={np.min(filtered.data)}, max={np.max(filtered.data)}")
    print(f"Filtered signal has NaN: {np.any(np.isnan(filtered.data))}")
    
    # Check power at each frequency
    def compute_power(sig, freq_hz, fsamp, bandwidth_hz=5):
        data = sig.data.flatten() if hasattr(sig.data, 'flatten') else sig.data
        fft_vals = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/fsamp)
        power = np.abs(fft_vals) ** 2 / len(data)
        mask = (np.abs(freqs) >= freq_hz - bandwidth_hz/2) & (np.abs(freqs) <= freq_hz + bandwidth_hz/2)
        return np.sum(power[mask])
    
    for freq in [50, 150, 250]:
        orig_power = compute_power(signal, freq, fsamp)
        filt_power = compute_power(filtered, freq, fsamp)
        ratio = filt_power / orig_power if orig_power > 0 else 0
        print(f"  {freq} Hz: orig={orig_power:.2f}, filt={filt_power:.2f}, ratio={ratio:.4f}")
        
except Exception as e:
    print(f"Error applying filter: {e}")
    import traceback
    traceback.print_exc()

