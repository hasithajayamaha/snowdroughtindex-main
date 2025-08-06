#!/usr/bin/env python3
"""
Script to examine coordinate variables in the CaSR NetCDF file
"""

import xarray as xr
import numpy as np

# Load the dataset
file_path = r"data\input_data\CaSR_SWE\CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1980-1983.nc"
ds = xr.open_dataset(file_path)

print('COORDINATE VARIABLES IN THE NETCDF FILE:')
print('='*60)

for coord_name, coord_var in ds.coords.items():
    print(f'\nCoordinate: {coord_name}')
    print(f'  Dimensions: {coord_var.dims}')
    print(f'  Shape: {coord_var.shape}')
    print(f'  Data type: {coord_var.dtype}')
    
    # Show attributes
    if coord_var.attrs:
        print(f'  Attributes:')
        for attr, value in coord_var.attrs.items():
            print(f'    {attr}: {value}')
    
    # Show sample values
    if coord_var.size <= 10:
        print(f'  Values: {coord_var.values}')
    else:
        # Handle different data types
        if coord_var.dtype.kind in ['f', 'i']:  # float or integer
            print(f'  Value range: {coord_var.min().values:.6f} to {coord_var.max().values:.6f}')
        else:  # datetime or other types
            print(f'  Value range: {coord_var.min().values} to {coord_var.max().values}')

print('\n' + '='*60)
print('GEOGRAPHIC COORDINATES SUMMARY:')
print('='*60)

# Geographic coordinates
if 'lon' in ds.coords and 'lat' in ds.coords:
    lon = ds.lon
    lat = ds.lat
    
    print(f'\nGeographic Longitude (lon):')
    print(f'  Variable name: lon')
    print(f'  Dimensions: {lon.dims}')
    print(f'  Shape: {lon.shape}')
    print(f'  Range: {lon.min().values:.3f} to {lon.max().values:.3f} degrees')
    print(f'  Units: {lon.attrs.get("units", "Not specified")}')
    print(f'  Description: {lon.attrs.get("long_name", "Not specified")}')
    
    print(f'\nGeographic Latitude (lat):')
    print(f'  Variable name: lat')
    print(f'  Dimensions: {lat.dims}')
    print(f'  Shape: {lat.shape}')
    print(f'  Range: {lat.min().values:.3f} to {lat.max().values:.3f} degrees')
    print(f'  Units: {lat.attrs.get("units", "Not specified")}')
    print(f'  Description: {lat.attrs.get("long_name", "Not specified")}')
    
    print(f'\nIMPORTANT: These are 2D coordinate arrays that provide the geographic')
    print(f'coordinates (longitude and latitude) for each grid point in the rotated coordinate system.')
    print(f'Each grid cell at position [i,j] has its geographic coordinates at lon[i,j] and lat[i,j].')

# Rotated coordinates
if 'rlon' in ds.coords and 'rlat' in ds.coords:
    rlon = ds.rlon
    rlat = ds.rlat
    
    print(f'\nRotated Longitude (rlon):')
    print(f'  Variable name: rlon')
    print(f'  Dimensions: {rlon.dims}')
    print(f'  Shape: {rlon.shape}')
    print(f'  Range: {rlon.min().values:.3f} to {rlon.max().values:.3f} degrees')
    print(f'  Description: 1D array defining the rotated longitude grid')
    
    print(f'\nRotated Latitude (rlat):')
    print(f'  Variable name: rlat')
    print(f'  Dimensions: {rlat.dims}')
    print(f'  Shape: {rlat.shape}')
    print(f'  Range: {rlat.min().values:.3f} to {rlat.max().values:.3f} degrees')
    print(f'  Description: 1D array defining the rotated latitude grid')

print('\n' + '='*60)
print('COORDINATE SYSTEM EXPLANATION:')
print('='*60)
print('\nThe CaSR dataset uses a rotated pole coordinate system:')
print('\n1. ROTATED COORDINATES (rlon, rlat):')
print('   - These are 1D arrays that define the regular grid in rotated space')
print('   - rlon: 35 points from -16.497° to -13.437°')
print('   - rlat: 35 points from -9.450° to -6.390°')
print('   - Used as the primary grid indices for data storage')

print('\n2. GEOGRAPHIC COORDINATES (lon, lat):')
print('   - These are 2D arrays (35x35) that provide the actual geographic coordinates')
print('   - lon[i,j]: Geographic longitude for grid point at rlat[i], rlon[j]')
print('   - lat[i,j]: Geographic latitude for grid point at rlat[i], rlon[j]')
print('   - Range: Longitude 242.266° to 248.021°, Latitude 45.872° to 49.754°')
print('   - These represent the actual Earth coordinates (Western Canada)')

print('\n3. HOW TO ACCESS GEOGRAPHIC COORDINATES:')
print('   - For precipitation data at grid point [i,j]:')
print('     * Geographic longitude: dataset.lon[i,j]')
print('     * Geographic latitude: dataset.lat[i,j]')
print('     * Precipitation value: dataset.CaSR_v3.1_A_PR24_SFC[time,i,j]')

ds.close()
print('\n✓ Analysis complete')
