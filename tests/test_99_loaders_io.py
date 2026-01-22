#CHECKED

"""
Tests for signal loading and I/O operations.
"""
import numpy as np
import pytest
import os
import tempfile
from pyphysio.signal import create_signal, load
from pyphysio import TestData


class TestSignalLoading:
    """Test signal loading functionality."""

    def test_load_saved_signal(self):
        """Test saving and loading a signal."""
        # Create a signal
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100, name='test')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save
            signal.to_netcdf(temp_file)
            
            # Load
            loaded = load(temp_file)
            
            assert loaded is not None
            assert np.allclose(loaded.values, signal.values, equal_nan=True)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_load_preserves_attributes(self):
        """Test that loading preserves signal attributes."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100, name='test_signal')
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            
            assert loaded.name == signal.name
            assert loaded.attrs['sampling_freq'] == signal.attrs['sampling_freq']
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_load_preserves_dimensions(self):
        """Test that loading preserves signal dimensions."""
        data = np.random.uniform(size=(1000, 5, 2))
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            
            assert loaded.shape == signal.shape
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestSignalFileFormats:
    """Test different signal file format handling."""

    def test_save_netcdf_format(self):
        """Test saving in NetCDF format."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_roundtrip_netcdf(self):
        """Test NetCDF save and load roundtrip."""
        data = np.random.uniform(size=(100, 5))
        original = create_signal(data, sampling_freq=100, name='roundtrip_test')
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save and load
            original.to_netcdf(temp_file)
            loaded = load(temp_file)
            
            # Check equality
            np.testing.assert_array_almost_equal(
                original.values, loaded.values
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestDataImportExport:
    """Test data import and export operations."""

    def test_export_to_numpy(self):
        """Test exporting signal to numpy array."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        
        numpy_array = signal.values
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape[0] == 1000

    def test_export_time_coordinates(self):
        """Test exporting time coordinates."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=10)
        
        times = signal.coords['time'].values
        assert isinstance(times, np.ndarray)
        assert len(times) == 1000

    def test_export_to_numpy_file(self):
        """Test exporting signal to numpy .npy file format."""
        data = np.random.uniform(size=1000)
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_file = f.name
        
        try:
            # Export to numpy file
            np.save(temp_file, signal.values)
            
            # Load from numpy file
            loaded_data = np.load(temp_file)
            
            assert isinstance(loaded_data, np.ndarray)
            np.testing.assert_array_almost_equal(loaded_data, signal.values)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_export_multi_channel_to_npy(self):
        """Test exporting multi-channel signal to .npy format."""
        data = np.random.uniform(size=(1000, 5))
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_file = f.name
        
        try:
            np.save(temp_file, signal.values)
            loaded_data = np.load(temp_file)
            
            assert loaded_data.shape == (1000, 5)
            np.testing.assert_array_almost_equal(loaded_data, signal.values)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestDataIntegrity:
    """Test data integrity during load/save operations."""

    def test_no_data_loss_float32(self):
        """Test no data loss with float32."""
        data = np.random.uniform(size=1000).astype(np.float32)
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            # Allow small precision loss due to format conversion
            np.testing.assert_array_almost_equal(
                signal.values, loaded.values, decimal=5
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_no_data_loss_float64(self):
        """Test no data loss with float64."""
        data = np.random.uniform(size=1000).astype(np.float64)
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            np.testing.assert_array_almost_equal(
                signal.values, loaded.values, decimal=10
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_preserve_nan_values(self):
        """Test that NaN values are preserved during save/load."""
        data = np.random.uniform(size=1000)
        data[100:110] = np.nan
        signal = create_signal(data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            
            # Check NaN positions match
            original_nan = np.isnan(signal.values)
            loaded_nan = np.isnan(loaded.values)
            np.testing.assert_array_equal(original_nan, loaded_nan)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestLargeFileHandling:
    """Test handling of large signal files."""

    def test_large_signal_save_load(self):
        """Test saving and loading large signals."""
        # Create large signal (1 hour at 1000 Hz)
        large_data = np.random.uniform(size=3600000)
        signal = create_signal(large_data, sampling_freq=1000)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            
            assert loaded.shape[0] == 3600000
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_multichannel_large_signal(self):
        """Test large multi-channel signal."""
        # 1 hour at 100 Hz, 32 channels
        large_data = np.random.uniform(size=(360000, 32))
        signal = create_signal(large_data, sampling_freq=100)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_file = f.name
        
        try:
            signal.to_netcdf(temp_file)
            loaded = load(temp_file)
            
            assert loaded.shape == (360000, 32)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
