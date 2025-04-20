"""
Pytest configuration file for the Snow Drought Index package.

This file contains shared fixtures and configuration for the test suite.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Add the package root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_dates():
    """
    Generate a list of sample dates.
    
    Returns
    -------
    list
        List of datetime objects.
    """
    start_date = datetime(2010, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365 * 3)]  # 3 years of data
    return dates

@pytest.fixture
def sample_swe_dataframe(sample_dates):
    """
    Generate a sample SWE DataFrame.
    
    Parameters
    ----------
    sample_dates : list
        List of datetime objects.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample SWE data.
    """
    # Create a DataFrame with random SWE values
    np.random.seed(42)  # For reproducibility
    
    # Create station columns
    stations = [f'station_{i}' for i in range(1, 6)]
    
    # Create a DataFrame with random SWE values
    data = {}
    for station in stations:
        # Create seasonal pattern with random noise
        days = np.arange(len(sample_dates))
        seasonal = 100 * np.sin(2 * np.pi * days / 365.25 - np.pi/2) + 100
        seasonal[seasonal < 0] = 0  # No negative SWE values
        noise = np.random.normal(0, 10, len(sample_dates))
        values = seasonal + noise
        values[values < 0] = 0  # No negative SWE values
        data[station] = values
    
    df = pd.DataFrame(data, index=sample_dates)
    return df

@pytest.fixture
def sample_swe_dataset(sample_dates):
    """
    Generate a sample SWE xarray Dataset.
    
    Parameters
    ----------
    sample_dates : list
        List of datetime objects.
        
    Returns
    -------
    xarray.Dataset
        Dataset containing sample SWE data.
    """
    # Create a Dataset with random SWE values
    np.random.seed(42)  # For reproducibility
    
    # Create station IDs
    station_ids = [f'station_{i}' for i in range(1, 6)]
    
    # Create coordinates
    coords = {
        'time': sample_dates,
        'station_id': station_ids
    }
    
    # Create data variables
    data_vars = {
        'swe': (['time', 'station_id'], np.random.rand(len(sample_dates), len(station_ids)) * 100),
        'lat': ('station_id', np.random.uniform(40, 45, len(station_ids))),
        'lon': ('station_id', np.random.uniform(-120, -115, len(station_ids))),
        'elevation': ('station_id', np.random.uniform(1500, 3000, len(station_ids)))
    }
    
    # Create the Dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds

@pytest.fixture
def sample_stations():
    """
    Generate a sample DataFrame of station information.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample station information.
    """
    # Create station IDs
    station_ids = [f'station_{i}' for i in range(1, 6)]
    
    # Create a DataFrame with station information
    stations = pd.DataFrame({
        'station_id': station_ids,
        'lat': np.random.uniform(40, 45, len(station_ids)),
        'lon': np.random.uniform(-120, -115, len(station_ids)),
        'elevation': np.random.uniform(1500, 3000, len(station_ids))
    })
    
    return stations

@pytest.fixture
def sample_basins():
    """
    Generate a sample DataFrame of basin information.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample basin information.
    """
    # Create basin IDs
    basin_ids = [f'basin_{i}' for i in range(1, 4)]
    
    # Create a DataFrame with basin information
    basins = pd.DataFrame({
        'basin_id': basin_ids,
        'lat': np.random.uniform(40, 45, len(basin_ids)),
        'lon': np.random.uniform(-120, -115, len(basin_ids)),
        'area': np.random.uniform(1000, 5000, len(basin_ids))
    })
    
    return basins
