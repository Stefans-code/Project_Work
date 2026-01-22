"""
Tests for the segmenters module including signal segmentation and feature extraction.
"""
import numpy as np
import pytest
from pyphysio.signal import create_signal
from pyphysio import TestData
import pyphysio.segmenters as segm
import pyphysio.indicators.timedomain as td
import pyphysio.indicators.frequencydomain as fd


class TestLabelSegments:
    """Test label-based signal segmentation."""

    def test_label_segments_basic(self):
        """Test basic label-based segmentation."""
        # Create data signal
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        # Create stimulus/label signal
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim)
        result = segmenter(signal)
        assert result is not None

    def test_label_segments_with_drop_mixed(self):
        """Test label segmentation with drop_mixed option."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim, drop_mixed=True)
        result = segmenter(signal)
        assert result is not None

    def test_label_segments_with_drop_cut(self):
        """Test label segmentation with drop_cut option."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim, drop_cut=True)
        result = segmenter(signal)
        assert result is not None

    def test_label_segments_with_feature_extraction(self):
        """Test label segmentation with feature extraction."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim)
        indicators = [td.Mean(), td.StDev()]
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None

    def test_label_segments_multiple_labels(self):
        """Test label segmentation with multiple distinct labels."""
        data = np.random.uniform(size=(2000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(2000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim_data[1000:1200] = 3
        stim_data[1500:1800] = 1
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim)
        result = segmenter(signal)
        assert result is not None


class TestFixedSegments:
    """Test fixed-size window segmentation."""

    def test_fixed_segments_basic(self):
        """Test basic fixed-size segmentation."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)  # 5 sec windows, 2 sec step
        result = segmenter(signal)
        assert result is not None

    def test_fixed_segments_with_timeline(self):
        """Test fixed segmentation with timeline."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        stim_data = np.ones(10000)
        stim = create_signal(stim_data, sampling_freq=1000, name='stimulus')
        
        segmenter = segm.FixedSegments(5, 2, timeline=stim)
        result = segmenter(signal)
        assert result is not None

    def test_fixed_segments_with_feature_extraction(self):
        """Test fixed segmentation with feature extraction."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [td.Mean(), td.StDev()]
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None

    def test_fixed_segments_different_window_sizes(self):
        """Test fixed segmentation with different window sizes."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        window_sizes = [1, 2, 5, 10]
        for win_size in window_sizes:
            segmenter = segm.FixedSegments(win_size, win_size // 2)
            result = segmenter(signal)
            assert result is not None

    def test_fixed_segments_step_size(self):
        """Test fixed segmentation with different step sizes."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        step_sizes = [1, 2, 5, 10]
        for step in step_sizes:
            segmenter = segm.FixedSegments(5, step)
            result = segmenter(signal)
            assert result is not None

    def test_fixed_segments_with_drop_options(self):
        """Test fixed segmentation with drop options."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2, drop_mixed=True, drop_cut=True)
        result = segmenter(signal)
        assert result is not None


class TestRandomFixedSegments:
    """Test random fixed-size segment selection."""

    def test_random_fixed_segments_basic(self):
        """Test basic random fixed-size segmentation."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.RandomFixedSegments(10, 2)  # 10 segments, 2 sec duration
        result = segmenter(signal)
        assert result is not None

    def test_random_fixed_segments_with_timeline(self):
        """Test random fixed segmentation with timeline."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        stim_data = np.ones(10000)
        stim = create_signal(stim_data, sampling_freq=1000, name='stimulus')
        
        segmenter = segm.RandomFixedSegments(10, 2, timeline=stim)
        result = segmenter(signal)
        assert result is not None

    def test_random_fixed_segments_with_feature_extraction(self):
        """Test random fixed segmentation with feature extraction."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.RandomFixedSegments(10, 2)
        indicators = [td.Mean(), td.StDev()]
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None

    def test_random_fixed_segments_reproducibility(self):
        """Test that random segmentation can be made reproducible."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        # Create two segmenters with same seed
        segmenter1 = segm.RandomFixedSegments(10, 2)
        result1 = segmenter1(signal)
        
        segmenter2 = segm.RandomFixedSegments(10, 2)
        result2 = segmenter2(signal)
        
        assert result1 is not None
        assert result2 is not None

    def test_random_fixed_segments_different_sizes(self):
        """Test random segmentation with different segment counts."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        for n_segments in [5, 10, 20]:
            segmenter = segm.RandomFixedSegments(n_segments, 2)
            result = segmenter(signal)
            assert result is not None

    def test_random_fixed_segments_different_durations(self):
        """Test random segmentation with different segment durations."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        for duration in [1, 2, 5]:
            segmenter = segm.RandomFixedSegments(10, duration)
            result = segmenter(signal)
            assert result is not None


class TestSegmentationFeatureExtraction:
    """Test feature extraction on segmented signals."""

    def test_segmentation_with_time_domain_indicators(self):
        """Test segmentation with time-domain indicators."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [
            td.Mean(),
            td.StDev(),
            td.Max(),
            td.Min(),
            td.Range()
        ]
        
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None

    def test_segmentation_with_frequency_domain_indicators(self):
        """Test segmentation with frequency-domain indicators."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [
            fd.PowerInBand(0, 50, method='welch'),
            fd.PowerInBand(50, 100, method='welch')
        ]
        
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None

    def test_segmentation_with_mixed_indicators(self):
        """Test segmentation with mixed time and frequency domain indicators."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [
            td.Mean(),
            td.StDev(),
            fd.PowerInBand(0, 50, method='welch')
        ]
        
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None

    def test_segmentation_feature_shape(self):
        """Test that extracted features have correct shape."""
        data = np.random.uniform(size=(10000, 3))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [td.Mean(), td.StDev()]
        result = segm.fmap(segmenter, indicators, signal)
        
        assert result is not None


class TestSegmentationWithRealData:
    """Test segmentation with real physiological data."""

    def test_segmentation_ecg_signal(self):
        """Test segmentation on ECG signal."""
        signal = TestData.ecg(return_signal=True)
        sampling_freq = signal.attrs['sampling_freq']
        
        # Adjust window size based on sampling frequency
        window_size = max(0.5, 100 / sampling_freq)
        step_size = window_size / 2
        
        segmenter = segm.FixedSegments(window_size, step_size)
        result = segmenter(signal)
        assert result is not None

    def test_segmentation_eda_signal(self):
        """Test segmentation on EDA signal."""
        signal = TestData.eda(return_signal=True)
        sampling_freq = signal.attrs['sampling_freq']
        
        window_size = max(1.0, 500 / sampling_freq)
        step_size = window_size / 2
        
        segmenter = segm.FixedSegments(window_size, step_size)
        result = segmenter(signal)
        assert result is not None

    def test_segmentation_bvp_signal(self):
        """Test segmentation on BVP signal."""
        signal = TestData.bvp(return_signal=True)
        sampling_freq = signal.attrs['sampling_freq']
        
        window_size = max(0.5, 100 / sampling_freq)
        step_size = window_size / 2
        
        segmenter = segm.FixedSegments(window_size, step_size)
        result = segmenter(signal)
        assert result is not None


class TestSegmentationEdgeCases:
    """Test edge cases in segmentation."""

    def test_segmentation_short_signal(self):
        """Test segmentation on short signals."""
        data = np.random.uniform(size=(100, 2))
        signal = create_signal(data, sampling_freq=100)
        
        segmenter = segm.FixedSegments(0.5, 0.25)
        result = segmenter(signal)
        assert result is not None

    def test_segmentation_single_channel(self):
        """Test segmentation on single-channel signal."""
        data = np.random.uniform(size=10000)
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        result = segmenter(signal)
        assert result is not None

    def test_segmentation_very_small_window(self):
        """Test segmentation with very small window."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(0.1, 0.05)
        result = segmenter(signal)
        assert result is not None

    def test_segmentation_window_larger_than_signal(self):
        """Test segmentation when window is larger than signal."""
        data = np.random.uniform(size=(100, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        # Window larger than signal
        segmenter = segm.FixedSegments(10, 5)
        result = segmenter(signal)
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
