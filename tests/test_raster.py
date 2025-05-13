"""
Tests for the raster data handling module.
"""

import os
import tempfile
import numpy as np
import xarray as xr
import pytest
import rasterio
from rasterio.transform import from_origin
from snowdroughtindex.core.raster import (
    read_raster, write_raster, raster_to_dataset, dataset_to_raster
)

def create_test_raster():
    """Create a test raster array with metadata."""
    # Create test data
    data = np.random.rand(100, 100)
    
    # Create transform
    transform = from_origin(0, 0, 1, 1)
    
    # Create CRS
    crs = 'EPSG:4326'
    
    return data, transform, crs

def test_read_write_raster():
    """Test reading and writing raster files."""
    # Create test data
    data, transform, crs = create_test_raster()
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        # Write using write_raster
        write_raster(
            data,
            tmp.name,
            crs=crs,
            transform=transform,
            nodata=-9999
        )
        
        # Read back
        da = read_raster(tmp.name)
        
        # Compare
        assert np.allclose(da.values, data)
        assert da.attrs['crs'] == crs
        assert da.attrs['transform'] == transform
        assert da.attrs['nodata'] == -9999
        
        # Clean up
        os.unlink(tmp.name)

def test_read_write_multiband_raster():
    """Test reading and writing multiband raster files."""
    # Create test data
    data = np.random.rand(3, 100, 100)  # 3 bands
    transform = from_origin(0, 0, 1, 1)
    crs = 'EPSG:4326'
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        # Write using write_raster
        write_raster(
            data,
            tmp.name,
            crs=crs,
            transform=transform,
            nodata=-9999
        )
        
        # Read back
        da = read_raster(tmp.name)
        
        # Compare
        assert np.allclose(da.values, data)
        assert da.attrs['crs'] == crs
        assert da.attrs['transform'] == transform
        assert da.attrs['nodata'] == -9999
        assert da.dims == ('band', 'y', 'x')
        
        # Clean up
        os.unlink(tmp.name)

def test_raster_dataset_conversion():
    """Test conversion between raster and dataset."""
    # Create test data
    data, transform, crs = create_test_raster()
    
    # Create DataArray
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        attrs={'crs': crs, 'transform': transform}
    )
    
    # Convert to Dataset
    ds = raster_to_dataset(da, name='temperature')
    
    # Check Dataset
    assert 'temperature' in ds
    assert ds.attrs['crs'] == crs
    assert ds.attrs['transform'] == transform
    
    # Convert back to DataArray
    da2 = dataset_to_raster(ds, 'temperature')
    
    # Check DataArray
    assert np.allclose(da2.values, data)
    assert da2.attrs['crs'] == crs
    assert da2.attrs['transform'] == transform

def test_read_specific_band():
    """Test reading a specific band from a multiband raster."""
    # Create test data
    data = np.random.rand(3, 100, 100)  # 3 bands
    transform = from_origin(0, 0, 1, 1)
    crs = 'EPSG:4326'
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        # Write using write_raster
        write_raster(
            data,
            tmp.name,
            crs=crs,
            transform=transform,
            nodata=-9999
        )
        
        # Read specific band
        da = read_raster(tmp.name, band=2)
        
        # Compare
        assert np.allclose(da.values, data[1])  # Note: bands are 1-indexed
        assert da.attrs['crs'] == crs
        assert da.attrs['transform'] == transform
        
        # Clean up
        os.unlink(tmp.name)

def test_masked_read():
    """Test reading raster with masked values."""
    # Create test data with nodata values
    data = np.random.rand(100, 100)
    data[0, 0] = -9999
    transform = from_origin(0, 0, 1, 1)
    crs = 'EPSG:4326'
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        # Write using write_raster
        write_raster(
            data,
            tmp.name,
            crs=crs,
            transform=transform,
            nodata=-9999
        )
        
        # Read with masking
        da = read_raster(tmp.name, masked=True)
        
        # Check that nodata value is masked
        assert np.isnan(da.values[0, 0])
        
        # Clean up
        os.unlink(tmp.name) 