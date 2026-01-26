#CHECKED

"""
Comprehensive tests for signal creation and manipulation.
"""
import numpy as np
import pytest
from pyphysio.signal import create_signal, load
from pyphysio import TestData
import xarray as xr

# TODO-AI [PRIORITY: HIGH]: Standardize signal access via accessor.
# - Replace uses of `signal.values` with `signal.p.get_values()`.
# TODO-AI [PRIORITY: HIGH]: Make random data deterministic in tests.
# - Add `np.random.seed(0)` at module start or use a seeded `rng` fixture.
# TODO-AI [PRIORITY: MEDIUM]: Replace loose duration/threshold checks with
# exact or `pytest.approx` comparisons derived from sampling_freq and length.


class TestSignalCreationBasic:
    """Test basic signal creation functionality."""

    def test_create_signal_with_sampling_freq(self):
        """Test creating a signal with sampling frequency."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        assert signal is not None
        assert signal.shape[0] == 1000
        assert signal.attrs['sampling_freq'] == 100

    def test_create_signal_with_times(self):
        """Test creating a signal with explicit times."""
        data = np.random.uniform(size=100)
        times = np.linspace(0, 10, 100)
        signal = create_signal(data, times=times)
        assert signal is not None
        assert signal.shape[0] == 100
        assert signal.attrs['sampling_freq'] == 'unevenly'

    def test_create_signal_with_start_time(self):
        """Test creating a signal with a start time."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10, start_time=5.0)
        assert signal is not None
        assert signal.attrs['start_time'] == 5.0

    def test_create_signal_1d(self):
        """Test creating a 1D signal."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        assert signal.values.ndim >= 1

    def test_create_signal_2d(self):
        """Test creating a 2D signal."""
        data = np.random.uniform(size=(1000, 5))
        signal = create_signal(data, sampling_freq=100)
        assert signal.values.ndim >= 2

    def test_create_signal_3d(self):
        """Test creating a 3D signal."""
        data = np.random.uniform(size=(1000, 5, 2))
        signal = create_signal(data, sampling_freq=100)
        assert signal.values.ndim >= 3

    def test_create_signal_with_name(self):
        """Test creating a signal with a custom name."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100, name='test_signal')
        assert signal.name == 'test_signal'

    def test_create_signal_with_info(self):
        """Test creating a signal with info metadata."""
        data = np.random.uniform(size=1000)
        info = {'subject_id': '001', 'session': 'baseline'}
        signal = create_signal(data, sampling_freq=100, info=info)
        assert signal.attrs['subject_id'] == '001'
        assert signal.attrs['session'] == 'baseline'

    def test_create_signal_invalid_both_times_and_freq(self):
        """Test that providing both times and sampling_freq raises error."""
        data = np.random.uniform(size=100)
        times = np.linspace(0, 10, 100)
        with pytest.raises(AssertionError):
            create_signal(data, times=times, sampling_freq=100)

    def test_create_signal_neither_times_nor_freq(self):
        """Test that providing neither times nor sampling_freq raises error."""
        data = np.random.uniform(size=100)
        with pytest.raises(AssertionError):
            create_signal(data)

    def test_create_signal_mismatched_times_length(self):
        """Test that mismatched times length raises error."""
        data = np.random.uniform(size=100)
        times = np.linspace(0, 10, 50)  # Wrong length
        with pytest.raises(AssertionError):
            create_signal(data, times=times)

    def test_create_signal_negative_sampling_freq(self):
        """Test that negative sampling frequency raises error."""
        data = np.random.uniform(size=100)
        with pytest.raises(AssertionError):
            create_signal(data, sampling_freq=-100)

    def test_create_signal_zero_sampling_freq(self):
        """Test that zero sampling frequency raises error."""
        data = np.random.uniform(size=100)
        with pytest.raises(AssertionError):
            create_signal(data, sampling_freq=0)


class TestSignalAttributes:
    """Test signal attributes and metadata."""

    def test_signal_has_xarray_structure(self):
        """Test that created signal is an xarray DataArray."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        assert isinstance(signal, xr.DataArray)

    def test_signal_preserves_data_values(self):
        """Test that signal preserves original data values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=1)
        np.testing.assert_array_almost_equal(signal.values.ravel()[:5], data)

    def test_signal_time_dimension(self):
        """Test that signal has correct time dimension."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=10)
        assert 'time' in signal.dims

    def test_signal_coordinates(self):
        """Test that signal has time coordinates."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        assert 'time' in signal.coords
        assert len(signal.coords['time']) == 100

    def test_signal_with_different_sampling_freqs(self):
        """Test creating signals with various sampling frequencies."""
        data = np.random.uniform(size=1000)
        freqs = [1, 10, 100, 1000, 0.1, 0.01]
        for freq in freqs:
            signal = create_signal(data, sampling_freq=freq)
            assert signal.attrs['sampling_freq'] == freq

class TestSignalOperations:
    """Test signal manipulation operations via PyphysioDataArray accessor (.p)."""

    # ===== Basic Data Access Methods =====
    
    def test_get_values(self):
        """Test retrieving signal values using get_values()."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=10)
        retrieved_values = signal.p.get_values()
        np.testing.assert_array_almost_equal(retrieved_values.ravel()[:5], data)

    def test_get_times(self):
        """Test retrieving time values using get_times()."""
        data = np.arange(100)
        times = np.linspace(0, 10, 100)
        signal = create_signal(data, times=times)
        retrieved_times = signal.p.get_times()
        np.testing.assert_array_almost_equal(retrieved_times, times)

    def test_clone_signal(self):
        """Test cloning a signal with new values."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10, name='original')
        new_data = np.random.uniform(size=100)
        cloned = signal.p.clone(new_data, name='cloned')
        
        assert cloned.name == 'cloned'
        np.testing.assert_array_almost_equal(cloned.p.get_values().ravel()[:100], new_data)
        # Check that metadata is preserved (or copied)
        assert 'sampling_freq' in cloned.attrs

    def test_clone_signal_shape_mismatch(self):
        """Test that cloning with mismatched shape raises error."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        wrong_data = np.random.uniform(size=50)
        
        with pytest.raises(AssertionError):
            signal.p.clone(wrong_data)

    # ===== Time-related Methods =====

    def test_get_start_time(self):
        """Test retrieving signal start time."""
        data = np.random.uniform(size=100)
        start = 5.0
        signal = create_signal(data, sampling_freq=10, start_time=start)
        assert signal.p.get_start_time() == start

    def test_get_end_time(self):
        """Test retrieving signal end time."""
        data = np.arange(100)
        signal = create_signal(data, sampling_freq=10, start_time=0)
        # With 100 samples at 10 Hz, duration is ~9.9 seconds (99/10)
        end_time = signal.p.get_end_time()
        assert end_time >= 9.9  # Allow for small floating point errors

    def test_reset_times(self):
        """Test resetting time coordinates."""
        data = np.arange(100)
        signal = create_signal(data, sampling_freq=10, start_time=0)
        new_start = 50.0
        reset_signal = signal.p.reset_times(new_start)
        
        assert reset_signal.p.get_start_time() == new_start
        assert reset_signal.attrs['start_time'] == new_start

    def test_get_duration(self):
        """Test retrieving signal duration."""
        data = np.arange(100)
        signal = create_signal(data, sampling_freq=10)
        duration = signal.p.get_duration()
        assert duration >= 9.9  # Approximately 9.9 seconds for 100 samples at 10 Hz

    # ===== Sampling Frequency Methods =====

    def test_get_sampling_freq(self):
        """Test retrieving sampling frequency."""
        data = np.random.uniform(size=1000)
        freq = 50
        signal = create_signal(data, sampling_freq=freq)
        assert signal.p.get_sampling_freq() == freq

    def test_resample_signal(self):
        """Test resampling a signal to different frequency."""
        data = np.sin(np.linspace(0, 10*np.pi, 1000))
        signal = create_signal(data, sampling_freq=100)
        
        new_freq = 50
        resampled = signal.p.resample(new_freq)
        
        assert resampled.attrs['sampling_freq'] == new_freq
        # New signal should have roughly half the samples
        assert resampled.shape[0] < signal.shape[0]

    def test_resample_same_frequency(self):
        """Test resampling to the same frequency."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        resampled = signal.p.resample(10)
        
        # Should have similar number of samples
        assert abs(resampled.shape[0] - signal.shape[0]) <= 1

    def test_resample_higher_frequency(self):
        """Test upsampling to higher frequency."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        resampled = signal.p.resample(100)
        
        assert resampled.attrs['sampling_freq'] == 100
        assert resampled.shape[0] > signal.shape[0]

    # ===== Segmentation Methods =====

    def test_segment_time_with_stop(self):
        """Test segmenting signal with both start and stop times."""
        data = np.arange(1000)
        signal = create_signal(data, sampling_freq=10, start_time=0)
        
        t_start = 2.0
        t_stop = 5.0
        segment = signal.p.segment_time(t_start, t_stop)
        
        assert segment.p.get_start_time() >= t_start
        assert segment.p.get_end_time() <= t_stop
        assert segment.shape[0] > 0

    def test_segment_time_without_stop(self):
        """Test segmenting signal with only start time."""
        data = np.arange(1000)
        signal = create_signal(data, sampling_freq=10, start_time=0)
        
        t_start = 5.0
        segment = signal.p.segment_time(t_start)
        
        assert segment.p.get_start_time() >= t_start
        assert segment.p.get_end_time() == signal.p.get_end_time()

    def test_segment_time_edge_cases(self):
        """Test segmenting at signal boundaries."""
        data = np.arange(100)
        signal = create_signal(data, sampling_freq=10, start_time=0)
        
        # Segment from start
        segment_start = signal.p.segment_time(signal.p.get_start_time())
        assert segment_start.shape[0] == signal.shape[0]
        
        # Segment to end
        segment_end = signal.p.segment_time(0, signal.p.get_end_time())
        assert segment_end.shape[0] == signal.shape[0]

    # ===== Channel/Component Methods =====

    def test_has_multi_channels_false(self):
        """Test has_multi_channels() for 1D signal."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        assert not signal.p.has_multi_channels()

    def test_has_multi_channels_true(self):
        """Test has_multi_channels() for multi-channel signal."""
        data = np.random.uniform(size=(100, 5))
        signal = create_signal(data, sampling_freq=10)
        assert signal.p.has_multi_channels()

    def test_get_nchannels_single(self):
        """Test get_nchannels() for single channel."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        assert signal.p.get_nchannels() is None

    def test_get_nchannels_multiple(self):
        """Test get_nchannels() for multiple channels."""
        n_channels = 8
        data = np.random.uniform(size=(100, n_channels))
        signal = create_signal(data, sampling_freq=10)
        assert signal.p.get_nchannels() == n_channels

    def test_has_multi_components_false(self):
        """Test has_multi_components() for signal without components."""
        data = np.random.uniform(size=(100, 5))
        signal = create_signal(data, sampling_freq=10)
        assert not signal.p.has_multi_components()

    def test_has_multi_components_true(self):
        """Test has_multi_components() for multi-component signal."""
        data = np.random.uniform(size=(100, 5, 3))
        signal = create_signal(data, sampling_freq=10)
        assert signal.p.has_multi_components()

    def test_get_ncomponents_none(self):
        """Test get_ncomponents() for single component."""
        data = np.random.uniform(size=(100, 5))
        signal = create_signal(data, sampling_freq=10)
        assert signal.p.get_ncomponents() is None

    def test_get_ncomponents_multiple(self):
        """Test get_ncomponents() for multiple components."""
        n_components = 4
        data = np.random.uniform(size=(100, 5, n_components))
        signal = create_signal(data, sampling_freq=10)
        assert signal.p.get_ncomponents() == n_components

    def test_select_channels_single(self):
        """Test selecting a single channel from multi-channel signal."""
        data = np.random.uniform(size=(100, 5))
        signal = create_signal(data, sampling_freq=10)
        
        selected = signal.p.select_channels(0, keep_dims=True)
        assert selected.shape[0] == 100

    def test_select_channels_multiple(self):
        """Test selecting multiple channels from multi-channel signal."""
        data = np.random.uniform(size=(100, 5))
        signal = create_signal(data, sampling_freq=10)
        
        selected = signal.p.select_channels([0, 2, 4], keep_dims=True)
        assert selected.shape[0] == 100

    # ===== Metadata Methods =====

    def test_get_info(self):
        """Test retrieving signal metadata."""
        data = np.random.uniform(size=100)
        info = {'subject_id': '001', 'session': 'baseline'}
        signal = create_signal(data, sampling_freq=10, info=info)
        
        retrieved_info = signal.p.get_info()
        assert retrieved_info['subject_id'] == '001'
        assert retrieved_info['session'] == 'baseline'
        assert retrieved_info['sampling_freq'] == 10

    def test_main_signal_property(self):
        """Test the main_signal property of PyphysioDataArray accessor."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        
        main_sig = signal.p.main_signal
        assert isinstance(main_sig, xr.DataArray)
        np.testing.assert_array_equal(main_sig.values, signal.values)

    # ===== NaN Processing Methods =====

    def test_process_na_keep(self):
        """Test process_na with 'keep' action."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=10)
        
        result = signal.p.process_na(na_action='keep')
        assert np.isnan(result.values[2])
        assert result.shape[0] == signal.shape[0]

    def test_process_na_remove(self):
        """Test process_na with 'remove' action."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=10)
        
        result = signal.p.process_na(na_action='remove')
        assert not np.any(np.isnan(result.values))
        assert result.shape[0] < signal.shape[0]
        assert result.attrs['sampling_freq'] == 'unevenly'

    def test_process_na_impute(self):
        """Test process_na with 'impute' action."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=10)
        
        result = signal.p.process_na(na_action='impute')
        assert not np.any(np.isnan(result.values))
        assert result.shape[0] == signal.shape[0]

    def test_process_na_no_nans(self):
        """Test process_na on signal without NaNs."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        signal = create_signal(data, sampling_freq=10)
        
        result = signal.p.process_na(na_action='keep')
        np.testing.assert_array_equal(result.values, signal.values)

    def test_process_na_invalid_action(self):
        """Test process_na with invalid action raises error."""
        data = np.array([1.0, 2.0, 3.0])
        signal = create_signal(data, sampling_freq=10)
        
        with pytest.raises(AssertionError):
            signal.p.process_na(na_action='invalid')

    # ===== Plotting Methods =====

    def test_plot_1d_signal(self):
        """Test plotting a 1D signal."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=10)
        # Should not raise an error
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        signal.p.plot()
        plt.close('all')

    def test_plot_1d_with_marker(self):
        """Test plotting 1D signal with marker."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        import matplotlib.pyplot as plt
        plt.ioff()
        signal.p.plot(marker='.')
        plt.close('all')

    def test_plot_1d_with_marker_vertical(self):
        """Test plotting 1D signal with vertical line marker."""
        data = np.random.uniform(size=100)
        signal = create_signal(data, sampling_freq=10)
        import matplotlib.pyplot as plt
        plt.ioff()
        signal.p.plot(marker='|')
        plt.close('all')

    def test_plot_2d_signal(self):
        """Test plotting a 2D multi-channel signal."""
        data = np.random.uniform(size=(1000, 5))
        signal = create_signal(data, sampling_freq=10)
        import matplotlib.pyplot as plt
        plt.ioff()
        signal.p.plot()
        plt.close('all')

    def test_plot_3d_signal(self):
        """Test plotting a 3D signal with channels and components."""
        data = np.random.uniform(size=(1000, 2, 3))
        signal = create_signal(data, sampling_freq=10)
        import matplotlib.pyplot as plt
        plt.ioff()
        signal.p.plot()
        plt.close('all')

    def test_plot_varying_sampling_freqs(self):
        """Test plotting signals with various sampling frequencies."""
        import matplotlib.pyplot as plt
        plt.ioff()
        data = np.random.uniform(size=1000)
        freqs = [0.01, 1, 10, 100]
        for freq in freqs:
            signal = create_signal(data, sampling_freq=freq)
            signal.p.plot()
            plt.close('all')

    def test_plot_unevenly_sampled(self):
        """Test plotting unevenly sampled signal."""
        import matplotlib.pyplot as plt
        plt.ioff()
        data = np.array([10, 20, 30])
        times = np.array([10, 20, 30])
        signal = create_signal(data, times=times)
        signal.p.plot(marker='|', color='r')
        plt.close('all')

    # ===== XArray Operations =====

    def test_signal_indexing(self):
        """Test signal indexing operations via isel."""
        data = np.arange(1000)
        signal = create_signal(data, sampling_freq=10)
        # Should be able to index the signal
        subset = signal.isel(time=slice(0, 100))
        assert subset.shape[0] == 100

    def test_signal_slicing(self):
        """Test signal time slicing via sel."""
        data = np.arange(1000)
        signal = create_signal(data, sampling_freq=10)
        # Should be able to slice by time
        subset = signal.sel(time=slice(0, 5))
        assert len(subset.time) > 0

    def test_signal_concatenation(self):
        """Test concatenating signals."""
        data1 = np.random.uniform(size=500)
        data2 = np.random.uniform(size=500)
        signal1 = create_signal(data1, sampling_freq=10)
        signal2 = create_signal(data2, sampling_freq=10, start_time=50)
        # Should be able to concatenate
        result = xr.concat([signal1, signal2], dim='time')
        assert result.shape[0] == 1000

    def test_signal_values_access(self):
        """Test accessing signal values directly."""
        data = np.random.uniform(size=(100, 5))
        signal = create_signal(data, sampling_freq=10)
        values = signal.values
        assert values.shape[0] == 100


class TestTestDataInterface:
    """Test the TestData class interface."""

    # ===== Test data loading as arrays =====

    def test_testdata_ecg_as_array(self):
        """Test loading ECG test data as raw array."""
        ecg = TestData.ecg(return_signal=False)
        assert ecg is not None
        assert len(ecg) > 0
        assert isinstance(ecg, np.ndarray)

    def test_testdata_eda_as_array(self):
        """Test loading EDA test data as raw array."""
        eda = TestData.eda(return_signal=False)
        assert eda is not None
        assert len(eda) > 0
        assert isinstance(eda, np.ndarray)

    def test_testdata_bvp_as_array(self):
        """Test loading BVP test data as raw array."""
        bvp = TestData.bvp(return_signal=False)
        assert bvp is not None
        assert len(bvp) > 0
        assert isinstance(bvp, np.ndarray)

    def test_testdata_resp_as_array(self):
        """Test loading RESP test data as raw array."""
        resp = TestData.resp(return_signal=False)
        assert resp is not None
        assert len(resp) > 0
        assert isinstance(resp, np.ndarray)

    def test_testdata_consistency(self):
        """Test that repeated loads of TestData return consistent data."""
        ecg1 = TestData.ecg(return_signal=False)
        ecg2 = TestData.ecg(return_signal=False)
        np.testing.assert_array_equal(ecg1, ecg2)

    def test_testdata_shapes_consistent(self):
        """Test that all medical signals have the same length."""
        ecg = TestData.ecg(return_signal=False)
        eda = TestData.eda(return_signal=False)
        bvp = TestData.bvp(return_signal=False)
        resp = TestData.resp(return_signal=False)
        
        # All should have same first dimension
        assert ecg.shape[0] == eda.shape[0]
        assert eda.shape[0] == bvp.shape[0]
        assert bvp.shape[0] == resp.shape[0]

    # ===== Test data loading as signals =====

    def test_testdata_ecg(self):
        """Test loading ECG test data."""
        ecg = TestData.ecg()
        assert ecg is not None
        assert len(ecg) > 0

    def test_testdata_ecg_as_signal(self):
        """Test loading ECG test data as signal."""
        signal = TestData.ecg(return_signal=True)
        assert isinstance(signal, xr.DataArray)

    def test_testdata_eda(self):
        """Test loading EDA test data."""
        eda = TestData.eda()
        assert eda is not None
        assert len(eda) > 0

    def test_testdata_eda_as_signal(self):
        """Test loading EDA test data as signal."""
        signal = TestData.eda(return_signal=True)
        assert isinstance(signal, xr.DataArray)

    def test_testdata_bvp(self):
        """Test loading BVP test data."""
        bvp = TestData.bvp()
        assert bvp is not None
        assert len(bvp) > 0

    def test_testdata_bvp_as_signal(self):
        """Test loading BVP test data as signal."""
        signal = TestData.bvp(return_signal=True)
        assert isinstance(signal, xr.DataArray)

    def test_testdata_resp(self):
        """Test loading RESP test data."""
        resp = TestData.resp()
        assert resp is not None
        assert len(resp) > 0

    def test_testdata_resp_as_signal(self):
        """Test loading RESP test data as signal."""
        signal = TestData.resp(return_signal=True)
        assert isinstance(signal, xr.DataArray)

    def test_testdata_fnirs(self):
        """Test loading fNIRS test data."""
        fnirs = TestData.fnirs()
        assert fnirs is not None

    def test_testdata_fnirs_as_signal(self):
        """Test loading fNIRS test data as signal."""
        signal = TestData.fnirs(return_signal=True)
        assert isinstance(signal, xr.DataArray)

    def test_testdata_tapping(self):
        """Test loading tapping test data."""
        tapping = TestData.tapping()
        assert tapping is not None

    def test_testdata_tapping_as_signal(self):
        """Test loading tapping test data as signal."""
        signal = TestData.tapping(return_signal=True)
        assert isinstance(signal, xr.DataArray)

    def test_testdata_consistent_dimensions(self):
        """Test that all test data have consistent dimensions when loaded as signals."""
        medical_signals = [
            TestData.ecg(return_signal=True),
            TestData.eda(return_signal=True),
            TestData.bvp(return_signal=True),
            TestData.resp(return_signal=True)
        ]
        # All should have time dimension
        for signal in medical_signals:
            assert 'time' in signal.dims


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
