"""
Tests for the IO module.
"""

import os
import tempfile
import numpy as np
import xarray as xr
import pytest
from snowdroughtindex.core.io import read_netcdf, write_netcdf, read_hdf5, write_hdf5

def test_netcdf_io():
    """Test NetCDF read and write functionality."""
    # Create a test dataset
    data = xr.Dataset(
        data_vars=dict(
            temperature=(["x", "y"], np.random.rand(4, 5)),
            precipitation=(["x", "y"], np.random.rand(4, 5))
        ),
        coords=dict(
            x=(["x"], np.arange(4)),
            y=(["y"], np.arange(5))
        )
    )
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        write_netcdf(data, tmp.name)
        
        # Read back
        read_data = read_netcdf(tmp.name)
        
        # Compare
        assert read_data.equals(data)
        
        # Clean up
        os.unlink(tmp.name)

def test_hdf5_io():
    """Test HDF5 read and write functionality."""
    # Create test data
    data = {
        'temperature': np.random.rand(4, 5),
        'precipitation': np.random.rand(4, 5)
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        write_hdf5(data, tmp.name)
        
        # Read back
        read_data = read_hdf5(tmp.name)
        
        # Compare
        for key in data:
            assert np.array_equal(read_data[key], data[key])
        
        # Clean up
        os.unlink(tmp.name)

def test_hdf5_group_io():
    """Test HDF5 read and write functionality with groups."""
    # Create test data
    data = {
        'temperature': np.random.rand(4, 5),
        'precipitation': np.random.rand(4, 5)
    }
    
    # Write to temporary file with group
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        write_hdf5(data, tmp.name, group='climate')
        
        # Read back
        read_data = read_hdf5(tmp.name, group='climate')
        
        # Compare
        for key in data:
            assert np.array_equal(read_data[key], data[key])
        
        # Clean up
        os.unlink(tmp.name) 