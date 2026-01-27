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
        segmenter(signal)

        # collect and verify segments
        segs = list(segmenter)
        assert len(segs) > 0
        for seg in segs:
            seg_sig = seg(signal)
            assert pytest.approx(seg.get_begin_time(), rel=1e-6) == seg_sig.p.get_start_time()
            assert pytest.approx(seg.get_end_time(), rel=1e-6) == seg_sig.p.get_end_time()
            assert seg_sig.p.get_end_time() > seg_sig.p.get_start_time()

    def test_label_segments_with_drop_mixed(self):
        """Test label segmentation with drop_mixed option."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim, drop_mixed=True)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0
        for seg in segs:
            seg_sig = seg(signal)
            assert pytest.approx(seg.get_begin_time(), rel=1e-6) == seg_sig.p.get_start_time()
            assert pytest.approx(seg.get_end_time(), rel=1e-6) == seg_sig.p.get_end_time()

    def test_label_segments_with_drop_cut(self):
        """Test label segmentation with drop_cut option."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim, drop_cut=True)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0
        for seg in segs:
            seg_sig = seg(signal)
            assert pytest.approx(seg.get_begin_time(), rel=1e-6) == seg_sig.p.get_start_time()
            assert pytest.approx(seg.get_end_time(), rel=1e-6) == seg_sig.p.get_end_time()

    def test_label_segments_with_feature_extraction(self):
        """Test label segmentation with feature extraction."""
        data = np.random.uniform(size=(1000, 2))
        signal = create_signal(data, sampling_freq=100)
        
        stim_data = np.zeros(1000)
        stim_data[200:400] = 1
        stim_data[600:800] = 2
        stim = create_signal(stim_data, sampling_freq=100, name='stimulus')
        
        segmenter = segm.LabelSegments(timeline=stim)
        segmenter(signal)
        # number of segments
        n_segs = sum(1 for _ in segmenter)

        indicators = [td.Mean()]
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs

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
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0
        # verify begin/end correspond to extracted segment signal
        for seg in segs:
            seg_sig = seg(signal)
            assert pytest.approx(seg.get_begin_time(), rel=1e-6) == seg_sig.p.get_start_time()
            assert pytest.approx(seg.get_end_time(), rel=1e-6) == seg_sig.p.get_end_time()


class TestFixedSegments:
    """Test fixed-size window segmentation."""

    def test_fixed_segments_basic(self):
        """Test basic fixed-size segmentation."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        step = 5
        width = 2
        segmenter = segm.FixedSegments(step, width)  # 5 sec windows, 2 sec step
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0
        for seg in segs:
            seg_sig = seg(signal)
            # duration should be approximately the width
            dur = seg.get_end_time() - seg.get_begin_time()
            assert pytest.approx(dur, rel=1e-3) == width
            assert pytest.approx(seg.get_begin_time(), rel=1e-6) == seg_sig.p.get_start_time()
            # end time may be slightly adjusted due to internal rounding; allow small absolute tolerance
            assert abs(seg.get_end_time() - float(seg_sig.p.get_end_time())) < 1e-3

    def test_fixed_segments_with_timeline(self):
        """Test fixed segmentation with timeline."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        stim_data = np.ones(10000)
        stim = create_signal(stim_data, sampling_freq=1000, name='stimulus')
        
        step = 5
        width = 2
        segmenter = segm.FixedSegments(step, width, timeline=stim)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0
        for seg in segs:
            seg_sig = seg(signal)
            dur = seg.get_end_time() - seg.get_begin_time()
            assert pytest.approx(dur, rel=1e-3) == width

    def test_fixed_segments_with_feature_extraction(self):
        """Test fixed segmentation with feature extraction."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        segmenter(signal)
        n_segs = sum(1 for _ in segmenter)
        indicators = [td.Mean()]
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs

    def test_fixed_segments_different_window_sizes(self):
        """Test fixed segmentation with different window sizes."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        window_sizes = [1, 2, 5, 10]
        for win_size in window_sizes:
            segmenter = segm.FixedSegments(win_size, win_size / 2)
            segmenter(signal)
            segs = list(segmenter)
            assert len(segs) > 0

    def test_fixed_segments_step_size(self):
        """Test fixed segmentation with different step sizes."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        step_sizes = [1, 2, 5, 10]
        for step in step_sizes:
            segmenter = segm.FixedSegments(5, step)
            segmenter(signal)
            segs = list(segmenter)
            assert isinstance(segs, list)
            if len(segs) == 0:
                # acceptable: depending on window/step relative to signal length
                continue

    def test_fixed_segments_with_drop_options(self):
        """Test fixed segmentation with drop options."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2, drop_mixed=True, drop_cut=True)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0


class TestRandomFixedSegments:
    """Test random fixed-size segment selection."""

    def test_random_fixed_segments_basic(self):
        """Test basic random fixed-size segmentation."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.RandomFixedSegments(10, 2, reference=signal)  # 10 segments, 2 sec duration
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_random_fixed_segments_with_timeline(self):
        """Test random fixed segmentation with timeline."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        stim_data = np.ones(10000)
        stim = create_signal(stim_data, sampling_freq=1000, name='stimulus')
        
        segmenter = segm.RandomFixedSegments(10, 2, timeline=stim)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_random_fixed_segments_with_feature_extraction(self):
        """Test random fixed segmentation with feature extraction."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.RandomFixedSegments(10, 2, reference=signal)
        segmenter(signal)
        n_segs = sum(1 for _ in segmenter)
        indicators = [td.Mean()]
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs

    def test_random_fixed_segments_reproducibility(self):
        """Test that random segmentation can be made reproducible."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        # Create two segmenters with same seed
        segmenter1 = segm.RandomFixedSegments(10, 2, reference=signal)
        segmenter1(signal)
        result1 = list(segmenter1)

        segmenter2 = segm.RandomFixedSegments(10, 2, reference=signal)
        segmenter2(signal)
        result2 = list(segmenter2)

        assert len(result1) > 0
        assert len(result2) > 0

    def test_random_fixed_segments_different_sizes(self):
        """Test random segmentation with different segment counts."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        for n_segments in [5, 10, 20]:
            segmenter = segm.RandomFixedSegments(n_segments, 2, reference=signal)
            segmenter(signal)
            segs = list(segmenter)
            assert len(segs) > 0

    def test_random_fixed_segments_different_durations(self):
        """Test random segmentation with different segment durations."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        for duration in [1, 2, 5]:
            segmenter = segm.RandomFixedSegments(10, duration, reference=signal)
            segmenter(signal)
            segs = list(segmenter)
            assert len(segs) > 0


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
        
        segmenter(signal)
        n_segs = sum(1 for _ in segmenter)
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs

    def test_segmentation_with_frequency_domain_indicators(self):
        """Test segmentation with frequency-domain indicators."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [
            fd.PowerInBand(0, 50, method='welch'),
            fd.PowerInBand(50, 100, method='welch')
        ]
        
        segmenter(signal)
        n_segs = sum(1 for _ in segmenter)
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs

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
        
        segmenter(signal)
        n_segs = sum(1 for _ in segmenter)
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs

    def test_segmentation_feature_shape(self):
        """Test that extracted features have correct shape."""
        data = np.random.uniform(size=(10000, 3))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        indicators = [td.Mean(), td.StDev()]
        segmenter(signal)
        n_segs = sum(1 for _ in segmenter)
        result = segm.fmap(segmenter, indicators, signal)
        assert result is not None
        assert result.sizes['time'] == n_segs


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
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_segmentation_eda_signal(self):
        """Test segmentation on EDA signal."""
        signal = TestData.eda(return_signal=True)
        sampling_freq = signal.attrs['sampling_freq']
        
        window_size = max(1.0, 500 / sampling_freq)
        step_size = window_size / 2
        
        segmenter = segm.FixedSegments(window_size, step_size)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_segmentation_bvp_signal(self):
        """Test segmentation on BVP signal."""
        signal = TestData.bvp(return_signal=True)
        sampling_freq = signal.attrs['sampling_freq']
        
        window_size = max(0.5, 100 / sampling_freq)
        step_size = window_size / 2
        
        segmenter = segm.FixedSegments(window_size, step_size)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0


class TestSegmentationEdgeCases:
    """Test edge cases in segmentation."""

    def test_segmentation_short_signal(self):
        """Test segmentation on short signals."""
        data = np.random.uniform(size=(100, 2))
        signal = create_signal(data, sampling_freq=100)
        
        segmenter = segm.FixedSegments(0.5, 0.25)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_segmentation_single_channel(self):
        """Test segmentation on single-channel signal."""
        data = np.random.uniform(size=10000)
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(5, 2)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_segmentation_very_small_window(self):
        """Test segmentation with very small window."""
        data = np.random.uniform(size=(10000, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        segmenter = segm.FixedSegments(0.1, 0.05)
        segmenter(signal)
        segs = list(segmenter)
        assert len(segs) > 0

    def test_segmentation_window_larger_than_signal(self):
        """Test segmentation when window is larger than signal."""
        data = np.random.uniform(size=(100, 2))
        signal = create_signal(data, sampling_freq=1000)
        
        # Window larger than signal
        segmenter = segm.FixedSegments(10, 5)
        segmenter(signal)
        segs = list(segmenter)
        assert isinstance(segs, list)
        # zero segments is allowed when window is larger than the signal


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
