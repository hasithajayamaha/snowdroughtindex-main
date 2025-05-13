"""
Tests for the drought indices module.
"""

import numpy as np
import xarray as xr
import pandas as pd
import pytest
from datetime import datetime
from snowdroughtindex.core.drought_indices import (
    calculate_swe_p_ratio,
    calculate_swe_p_drought_index,
    calculate_swe_p_anomaly,
    classify_drought_severity
)

def create_test_data():
    """Create test data for drought indices."""
    # Create time coordinates
    time = pd.date_range(start='2000-01-01', end='2000-12-31', freq='D')
    
    # Create spatial coordinates
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-120, -110, 10)
    
    # Create random SWE and precipitation data
    np.random.seed(42)
    swe = np.random.rand(len(time), len(lat), len(lon)) * 100
    precipitation = np.random.rand(len(time), len(lat), len(lon)) * 50
    
    # Create xarray datasets
    swe_da = xr.DataArray(
        swe,
        dims=['time', 'lat', 'lon'],
        coords={'time': time, 'lat': lat, 'lon': lon},
        name='swe'
    )
    
    precip_da = xr.DataArray(
        precipitation,
        dims=['time', 'lat', 'lon'],
        coords={'time': time, 'lat': lat, 'lon': lon},
        name='precipitation'
    )
    
    return swe_da, precip_da

def test_calculate_swe_p_ratio():
    """Test SWE/P ratio calculation."""
    swe, precip = create_test_data()
    
    # Test monthly ratio
    ratio_monthly = calculate_swe_p_ratio(swe, precip, time_period='monthly')
    assert isinstance(ratio_monthly, xr.DataArray)
    assert 'time' in ratio_monthly.dims  # Now checking for time dimension
    assert ratio_monthly.shape[1:] == (10, 10)  # Check spatial dimensions
    
    # Test seasonal ratio
    ratio_seasonal = calculate_swe_p_ratio(swe, precip, time_period='seasonal')
    assert isinstance(ratio_seasonal, xr.DataArray)
    assert 'time' in ratio_seasonal.dims  # Now checking for time dimension
    assert ratio_seasonal.shape[1:] == (10, 10)  # Check spatial dimensions
    
    # Test annual ratio
    ratio_annual = calculate_swe_p_ratio(swe, precip, time_period='annual')
    assert isinstance(ratio_annual, xr.DataArray)
    assert 'time' in ratio_annual.dims  # Now checking for time dimension
    assert ratio_annual.shape[1:] == (10, 10)  # Check spatial dimensions

def test_calculate_swe_p_drought_index():
    """Test the SWE/P drought index calculation."""
    # Create test data
    swe, precip = create_test_data()
    
    # Test with monthly time period and no climatology period
    drought_index = calculate_swe_p_drought_index(swe, precip, time_period='monthly')
    assert isinstance(drought_index, xr.DataArray)
    assert 'time' in drought_index.dims
    assert drought_index.shape == swe.shape
    assert not drought_index.isnull().values.any()  # Changed to access values before checking
    
    # Test with seasonal time period
    drought_index_seasonal = calculate_swe_p_drought_index(swe, precip, time_period='seasonal')
    assert isinstance(drought_index_seasonal, xr.DataArray)
    assert 'time' in drought_index_seasonal.dims
    assert drought_index_seasonal.shape[1:] == swe.shape[1:]  # Check spatial dimensions match
    
    # Test with annual time period
    drought_index_annual = calculate_swe_p_drought_index(swe, precip, time_period='annual')
    assert isinstance(drought_index_annual, xr.DataArray)
    assert 'time' in drought_index_annual.dims
    assert drought_index_annual.shape[1:] == swe.shape[1:]  # Check spatial dimensions match
    
    # Test with climatology period
    start_year = pd.Timestamp(swe.time.min().values).year
    end_year = pd.Timestamp(swe.time.max().values).year
    drought_index_clim = calculate_swe_p_drought_index(
        swe, precip, 
        time_period='monthly',
        climatology_start_year=start_year,
        climatology_end_year=end_year
    )
    assert isinstance(drought_index_clim, xr.DataArray)
    assert 'time' in drought_index_clim.dims
    assert drought_index_clim.shape == swe.shape
    
    # Test with numpy arrays
    swe_np = swe.values
    precip_np = precip.values
    drought_index_np = calculate_swe_p_drought_index(swe_np, precip_np)
    assert isinstance(drought_index_np, np.ndarray)
    assert drought_index_np.shape == swe_np.shape
    
    # Test standardization
    # The standardized values should have mean close to 0 and std close to 1
    if isinstance(drought_index, xr.DataArray):
        mean = float(drought_index.mean())  # Convert to float to avoid xarray comparison
        std = float(drought_index.std())    # Convert to float to avoid xarray comparison
        assert abs(mean) < 0.1  # Mean should be close to 0
        assert abs(std - 1) < 0.1  # Std should be close to 1
    
    # Test with different climatology periods
    mid_year = (start_year + end_year) // 2
    drought_index_half = calculate_swe_p_drought_index(
        swe, precip,
        time_period='monthly',
        climatology_start_year=start_year,
        climatology_end_year=mid_year
    )
    assert isinstance(drought_index_half, xr.DataArray)
    assert drought_index_half.shape == swe.shape
    
    # Test that different climatology periods produce different results
    assert not np.allclose(drought_index.values, drought_index_half.values)
    
    # Test with invalid climatology years
    with pytest.raises(ValueError):
        calculate_swe_p_drought_index(
            swe, precip,
            climatology_start_year=end_year + 1,
            climatology_end_year=end_year + 2
        )
    
    # Test with invalid time period
    with pytest.raises(ValueError):
        calculate_swe_p_drought_index(swe, precip, time_period='invalid')

def test_calculate_swe_p_anomaly():
    """Test anomaly calculation."""
    swe, precip = create_test_data()
    ratio = calculate_swe_p_ratio(swe, precip)
    
    # Test without climatology period
    anomaly = calculate_swe_p_anomaly(ratio)
    assert isinstance(anomaly, xr.DataArray)
    assert anomaly.shape == ratio.shape
    
    # Test with climatology period
    anomaly_clim = calculate_swe_p_anomaly(ratio, climatology_period=('2000-01-01', '2000-06-30'))
    assert isinstance(anomaly_clim, xr.DataArray)
    assert anomaly_clim.shape == ratio.shape

def test_classify_drought_severity():
    """Test drought severity classification."""
    swe, precip = create_test_data()
    ratio = calculate_swe_p_ratio(swe, precip)
    anomaly = calculate_swe_p_anomaly(ratio)
    
    severity = classify_drought_severity(anomaly)
    assert isinstance(severity, xr.DataArray)
    assert severity.shape == anomaly.shape
    assert np.all(np.isin(severity.values, range(-2, 7)))  # Values from -2 to 6
    
    # Test specific cases
    test_anomaly = xr.DataArray([-2.5, -1.8, -1.2, -0.8, 0.0, 0.8, 1.2, 1.8, 2.5])
    test_severity = classify_drought_severity(test_anomaly)
    expected_severity = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6])
    assert np.array_equal(test_severity.values, expected_severity) 