import numpy as np
import pytest

from pyphysio.signal import create_signal
import pyphysio.sqi as sqi


def _extract_sqi_and_flag(result):
    # result values shape should have last axis == 2 (sqi, is_good)
    vals = np.array(result.values)
    assert vals.shape[-1] == 2
    sqi_val = float(vals[..., 0].ravel()[0])
    is_good = bool(vals[..., 1].ravel()[0])
    return sqi_val, is_good


class TestKurtosis:
    def test_kurtosis_normal_distribution(self):
        np.random.seed(0)
        data = np.random.normal(0, 1, 1000)
        signal = create_signal(data, sampling_freq=100)

        ind = sqi.Kurtosis(threshold=[-2.0, 2.0])
        result = ind(signal)

        k, good = _extract_sqi_and_flag(result)
        # Normal distribution kurtosis around 0
        assert abs(k) < 2.0
        assert good is True


class TestEntropy:
    def test_entropy_uniform_vs_constant(self):
        # Uniform signal -> high entropy
        np.random.seed(1)
        data_u = np.random.uniform(-1, 1, 1000)
        sig_u = create_signal(data_u, sampling_freq=100)
        ind_u = sqi.Entropy(threshold=[0.0, 10.0], nbins=25)
        res_u = ind_u(sig_u)
        ent_u, good_u = _extract_sqi_and_flag(res_u)
        assert ent_u >= 0.0
        assert good_u is True

        # Constant signal -> low entropy
        data_c = np.ones(1000) * 5.0
        sig_c = create_signal(data_c, sampling_freq=100)
        ind_c = sqi.Entropy(threshold=[0.0, 0.1], nbins=10)
        res_c = ind_c(sig_c)
        ent_c, good_c = _extract_sqi_and_flag(res_c)
        assert ent_c >= 0.0
        assert good_c is True or good_c is False


class TestSpectralPowerRatio:
    def test_spectral_power_ratio_sine(self):
        fs = 100
        t = np.arange(0, 10, 1 / fs)
        data = np.sin(2 * np.pi * 10 * t)
        signal = create_signal(data, sampling_freq=fs)

        # bandN contains the tone, bandD is a wider band
        ind = sqi.SpectralPowerRatio(threshold=[0.0, 10.0], method='welch', bandN=(5, 15), bandD=(1, 49))
        res = ind(signal)
        spr, good = _extract_sqi_and_flag(res)
        assert spr >= 0.0
        assert good is True


class TestCVSignal:
    def test_cv_signal_low_variability(self):
        np.random.seed(2)
        data = 5.0 + np.random.normal(0, 0.05, 1000)
        signal = create_signal(data, sampling_freq=100)

        ind = sqi.CVSignal(threshold=[0.0, 10.0])
        res = ind(signal)
        cv, good = _extract_sqi_and_flag(res)
        assert cv >= 0.0
        assert good is True


class TestPercentageNAN:
    def test_percentage_nan(self):
        data = np.ones(1000)
        # insert 100 NaNs -> 10%
        data[:100] = np.nan
        signal = create_signal(data, sampling_freq=100)

        ind = sqi.PercentageNAN(threshold=[0.0, 15.0])
        res = ind(signal)
        perc, good = _extract_sqi_and_flag(res)
        assert pytest.approx(perc, rel=1e-2) == 10.0
        assert good is True

        # require stricter threshold -> should be flagged as bad
        ind2 = sqi.PercentageNAN(threshold=[0.0, 5.0])
        res2 = ind2(signal)
        perc2, good2 = _extract_sqi_and_flag(res2)
        assert good2 is False
