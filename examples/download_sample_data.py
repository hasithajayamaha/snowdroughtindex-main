"""
Script to download sample SWE and precipitation data from NSIDC.
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd

def create_sample_data():
    """Create sample data for testing."""
    # Create coordinates
    lat = np.linspace(30, 50, 100)  # Western US latitudes
    lon = np.linspace(-125, -100, 100)  # Western US longitudes
    time = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    
    # Create meshgrid for coordinates
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Create synthetic SWE data with realistic patterns
    # Higher values in mountains (approximated by latitude and elevation)
    elevation = 1000 + 2000 * np.exp(-(lon_grid + 115)**2/100) * np.exp(-(lat_grid - 40)**2/50)
    base_swe = np.maximum(0, elevation - 1000) / 1000  # Base SWE pattern
    
    # Create time-varying SWE data
    swe_data = np.zeros((len(time), len(lat), len(lon)))
    for i in range(len(time)):
        daily_variation = 0.2 * np.random.randn(len(lat), len(lon))
        swe_data[i] = base_swe + daily_variation
    swe_data = np.maximum(0, swe_data)  # Ensure non-negative values
    
    # Create synthetic precipitation data
    # Precipitation correlated with elevation but with more temporal variation
    precip_data = np.zeros((len(time), len(lat), len(lon)))
    for i in range(len(time)):
        daily_precip = 0.5 * base_swe + 0.5 * np.random.rand(len(lat), len(lon))
        precip_data[i] = daily_precip
    precip_data = np.maximum(0, precip_data)  # Ensure non-negative values
    
    # Create xarray DataArrays
    swe = xr.DataArray(
        swe_data * 1000,  # Convert to mm
        dims=['time', 'lat', 'lon'],
        coords={
            'time': time,
            'lat': lat,
            'lon': lon
        },
        name='swe',
        attrs={
            'units': 'mm',
            'long_name': 'Snow Water Equivalent'
        }
    )
    
    precip = xr.DataArray(
        precip_data * 1000,  # Convert to mm
        dims=['time', 'lat', 'lon'],
        coords={
            'time': time,
            'lat': lat,
            'lon': lon
        },
        name='precipitation',
        attrs={
            'units': 'mm',
            'long_name': 'Total Precipitation'
        }
    )
    
    return swe, precip

def main():
    """Create sample data for testing."""
    # Create output directory if it doesn't exist
    os.makedirs('sample_data', exist_ok=True)
    
    print("Creating synthetic sample data...")
    swe, precip = create_sample_data()
    
    # Save the data
    print("Saving data...")
    swe.to_netcdf('sample_data/processed_swe.nc')
    precip.to_netcdf('sample_data/processed_precip.nc')
    
    print("Sample data creation complete!")
    print("Files saved:")
    print("- sample_data/processed_swe.nc")
    print("- sample_data/processed_precip.nc")

if __name__ == "__main__":
    main() 