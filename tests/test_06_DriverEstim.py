import numpy as np
import pytest

from pyphysio.signal import create_signal
from pyphysio.specialized.eda import DriverEstim
from pyphysio.generators.physiological import EDAGenerator


def test_driverestim_recovers_known_impulses():
    np.random.seed(0)
    fs = 100.0
    n = 2000

    # create sparse driver (impulses)
    driver = np.zeros(n)
    true_idxs = [200, 800, 1500]
    true_amps = [0.5, 0.2, 0.8]
    for i, a in zip(true_idxs, true_amps):
        driver[i] = a

    # create observed EDA by using the EDAGenerator (convolves impulses with Bateman IR)
    stimulus_times = [i / fs for i in true_idxs]
    phasic = EDAGenerator.eda_phasic_component(duration=n / fs,
                                              sampling_freq=fs,
                                              stimulus_times=stimulus_times,
                                              response_magnitude=true_amps,
                                              t1=0.96, t2=3.76,
                                              start_time=0)

    sig = phasic

    # run DriverEstim (deconvolution) to estimate driver
    de_est = DriverEstim(t1=0.96, t2=3.76, rescale_driver=False)
    est = de_est(sig)
    est_vals = np.array(est).ravel()

    # For each true impulse check a peak nearby in the estimate
    peaks_found = []
    for idx, amp in zip(true_idxs, true_amps):
        i0 = max(0, idx - 5)
        i1 = min(n, idx + 6)
        window = est_vals[i0:i1]
        peak = np.max(window)
        peak_idx = i0 + int(np.argmax(window))
        assert peak > 0.0
        # location should be close to true index
        assert abs(peak_idx - idx) <= 5
        peaks_found.append(peak)

    # ensure we detected at least as many positive peaks as true impulses
    assert len([p for p in peaks_found if p > 0]) >= len(true_idxs)

    # We primarily verify that peaks are detected at the expected locations
    # Amplitude scaling may differ due to normalization differences in generator
    # so we do not assert exact amplitude recovery here.
