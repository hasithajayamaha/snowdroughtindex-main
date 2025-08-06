"""
Example script demonstrating drought indices with real data.
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
from snowdroughtindex.core.drought_indices import (
    calculate_swe_p_ratio,
    calculate_swe_p_drought_index,
    calculate_swe_p_anomaly,
    classify_drought_severity
)

def load_data():
    """Load example SWE and precipitation data."""
    # Use the processed sample data
    swe_path = "sample_data/processed_swe.nc"
    precip_path = "sample_data/processed_precip.nc"
    
    try:
        swe = xr.open_dataarray(swe_path)
        precip = xr.open_dataarray(precip_path)
        return swe, precip
    except FileNotFoundError:
        print("Data files not found. Please run download_sample_data.py first.")
        return None, None

def plot_drought_index(drought_index, title, output_path):
    """Plot drought index map."""
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot the data
    drought_index.plot(ax=ax, transform=ccrs.PlateCarree(),
                      cmap='RdBu', vmin=-1, vmax=1)
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines()
    
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def plot_severity(severity, title, output_path):
    """Plot drought severity map."""
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Define custom colormap for severity
    colors = ['#8B0000', '#FF0000', '#FFA500', '#FFD700',  # Drought colors
              '#FFFFFF',  # Normal
              '#90EE90', '#008000', '#006400', '#004000']  # Wet colors
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Plot the data
    severity.plot(ax=ax, transform=ccrs.PlateCarree(),
                 cmap=cmap, vmin=-2, vmax=6)
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines()
    
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function to demonstrate drought indices."""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Load data
    swe, precip = load_data()
    if swe is None or precip is None:
        return
    
    print("\nData dimensions:")
    print("SWE:", swe.dims)
    print("Precipitation:", precip.dims)
    
    # Calculate SWE/P ratio
    ratio = calculate_swe_p_ratio(swe, precip, time_period='monthly')
    
    print("\nRatio dimensions:", ratio.dims)
    if 'time' in ratio.dims:
        print("Time coordinates:", ratio.time.values)
    
    # Calculate drought index
    drought_index = calculate_swe_p_drought_index(ratio)
    
    # Calculate anomaly
    anomaly = calculate_swe_p_anomaly(ratio)
    
    # Classify severity
    severity = classify_drought_severity(anomaly)
    
    # Plot results for the first time step
    plot_drought_index(drought_index.isel(time=0),
                      "Drought Index (First Time Step)",
                      "output/drought_index.png")
    
    plot_severity(severity.isel(time=0),
                 "Drought Severity (First Time Step)",
                 "output/drought_severity.png")
    
    # Print summary statistics
    print("\nDrought Index Summary:")
    print(f"Mean: {drought_index.mean().values:.2f}")
    print(f"Standard Deviation: {drought_index.std().values:.2f}")
    print(f"Minimum: {drought_index.min().values:.2f}")
    print(f"Maximum: {drought_index.max().values:.2f}")
    
    print("\nSeverity Classification Counts:")
    for i in range(-2, 7):
        count = (severity == i).sum().values
        print(f"Severity {i}: {count} grid cells")

if __name__ == "__main__":
    main() 