"""
Tests for the data validation and quality control module.
"""

import numpy as np
import xarray as xr
import pytest
from datetime import datetime, timedelta
from snowdroughtindex.core.validation import DataValidator, validate_time_series

def create_test_dataset():
    """Create a test dataset with various quality issues."""
    time = np.array([
        datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)
    ])
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    
    # Create data with some missing values and outliers
    data = np.random.rand(10, 5, 5)
    data[0, 0, 0] = np.nan  # Missing value
    data[1, 1, 1] = 1000.0  # Outlier
    
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["time", "lat", "lon"], data),
            precipitation=(["time", "lat", "lon"], data * 10)
        ),
        coords=dict(
            time=time,
            lat=lat,
            lon=lon
        )
    )
    return ds

def test_data_validator_initialization():
    """Test DataValidator initialization."""
    validator = DataValidator(
        missing_value_threshold=0.1,
        temporal_gap_threshold=3,
        spatial_coverage_threshold=0.9,
        value_range={"temperature": (-50, 50)}
    )
    
    assert validator.missing_value_threshold == 0.1
    assert validator.temporal_gap_threshold == 3
    assert validator.spatial_coverage_threshold == 0.9
    assert validator.value_range["temperature"] == (-50, 50)

def test_validate_dataset():
    """Test dataset validation."""
    ds = create_test_dataset()
    validator = DataValidator()
    issues = validator.validate_dataset(ds)
    
    # Check that issues are detected
    assert "missing_values" in issues
    assert "value_ranges" in issues

def test_quality_control():
    """Test quality control functionality."""
    ds = create_test_dataset()
    validator = DataValidator(value_range={"temperature": (0, 1)})
    qc_ds = validator.quality_control(ds)
    
    # Check that outliers are removed
    assert np.isnan(qc_ds.temperature[1, 1, 1])
    
    # Check that original data is not modified
    assert not np.isnan(ds.temperature[1, 1, 1])

def test_validate_time_series():
    """Test time series validation."""
    time = np.array([
        datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)
    ])
    data = np.random.rand(10)
    data[0] = np.nan  # Missing value
    data[1] = 1000.0  # Outlier
    
    issues = validate_time_series(data, time)
    
    # Check that issues are detected
    assert "missing_values" in issues
    assert "outliers" in issues

def test_temporal_gap_detection():
    """Test detection of temporal gaps."""
    # Create dataset with a gap
    time = np.array([
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 8)  # 6-day gap
    ])
    data = np.random.rand(3, 5, 5)
    
    ds = xr.Dataset(
        data_vars=dict(temperature=(["time", "lat", "lon"], data)),
        coords=dict(
            time=time,
            lat=np.linspace(-90, 90, 5),
            lon=np.linspace(-180, 180, 5)
        )
    )
    
    validator = DataValidator(temporal_gap_threshold=5)
    issues = validator.validate_dataset(ds)
    
    assert "temporal_gaps" in issues

def test_spatial_coverage():
    """Test spatial coverage validation."""
    ds = create_test_dataset()
    validator = DataValidator(spatial_coverage_threshold=0.99)
    issues = validator.validate_dataset(ds)
    
    assert "spatial_coverage" in issues 