#!/usr/bin/env python3
"""
Improved CaSR SWE Data Analysis Script

This script provides enhanced analysis of Canadian Snow and Sea Ice Service Reanalysis (CaSR) 
precipitation data from NetCDF files, with better handling of sparse data and improved visualizations.

Author: Data Analysis Script (Improved)
Date: 2025-06-27
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class CaSRDataAnalyzer:
    """
    An improved class to analyze CaSR precipitation NetCDF data files.
    """
    
    def __init__(self, file_path):
        """
        Initialize the analyzer with a NetCDF file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the NetCDF file to analyze
        """
        self.file_path = file_path
        self.dataset = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load the NetCDF dataset."""
        try:
            print(f"Loading data from: {self.file_path}")
            self.dataset = xr.open_dataset(self.file_path)
            print("✓ Data loaded successfully!")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def display_basic_info(self):
        """Display basic information about the dataset."""
        if self.dataset is None:
            print("No dataset loaded. Please load data first.")
            return
        
        print("\n" + "="*70)
        print("DATASET BASIC INFORMATION")
        print("="*70)
        
        # File information
        print(f"File: {os.path.basename(self.file_path)}")
        print(f"File size: {os.path.getsize(self.file_path) / (1024**2):.2f} MB")
        
        # Dataset dimensions
        print(f"\nDimensions:")
        for dim, size in self.dataset.dims.items():
            print(f"  {dim}: {size:,}")
        
        # Dataset variables
        print(f"\nData Variables:")
        for var in self.dataset.data_vars:
            var_info = self.dataset[var]
            print(f"  {var}: {var_info.dims} - {var_info.shape}")
            if hasattr(var_info, 'long_name'):
                print(f"    Description: {var_info.long_name}")
            if hasattr(var_info, 'units'):
                print(f"    Units: {var_info.units}")
        
        # Coordinate variables
        print(f"\nCoordinate Variables:")
        for coord in self.dataset.coords:
            coord_info = self.dataset[coord]
            print(f"  {coord}: {coord_info.dims} - {coord_info.shape}")
            if hasattr(coord_info, 'long_name'):
                print(f"    Description: {coord_info.long_name}")
            if hasattr(coord_info, 'units'):
                print(f"    Units: {coord_info.units}")
        
        # Global attributes
        print(f"\nGlobal Attributes:")
        for attr, value in self.dataset.attrs.items():
            # Truncate very long attribute values
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {attr}: {value}")
    
    def analyze_temporal_coverage(self):
        """Analyze temporal coverage of the dataset."""
        if self.dataset is None:
            return
        
        print("\n" + "="*70)
        print("TEMPORAL ANALYSIS")
        print("="*70)
        
        if 'time' in self.dataset.coords:
            time_coord = self.dataset.time
            
            # Convert to pandas datetime for easier handling
            time_values = pd.to_datetime(time_coord.values)
            
            print(f"Time range: {time_values.min()} to {time_values.max()}")
            print(f"Number of time steps: {len(time_values):,}")
            print(f"Duration: {(time_values.max() - time_values.min()).days} days")
            
            # Determine frequency
            time_diff = pd.Series(time_values).diff().dropna()
            most_common_freq = time_diff.mode()[0] if len(time_diff) > 0 else None
            print(f"Most common time step: {most_common_freq}")
            
            # Store temporal info
            self.analysis_results['temporal'] = {
                'start_date': time_values.min(),
                'end_date': time_values.max(),
                'n_timesteps': len(time_values),
                'duration_days': (time_values.max() - time_values.min()).days,
                'most_common_freq': str(most_common_freq)
            }
            
            # Yearly distribution
            yearly_counts = time_values.to_series().dt.year.value_counts().sort_index()
            print(f"\nYearly distribution:")
            for year, count in yearly_counts.items():
                print(f"  {year}: {count:,} timesteps")
            
            # Monthly distribution
            monthly_counts = time_values.to_series().dt.month.value_counts().sort_index()
            print(f"\nMonthly distribution:")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, count in monthly_counts.items():
                print(f"  {month_names[month-1]}: {count:,} timesteps")
        else:
            print("No time coordinate found in dataset.")
    
    def analyze_spatial_coverage(self):
        """Analyze spatial coverage and grid information."""
        if self.dataset is None:
            return
        
        print("\n" + "="*70)
        print("SPATIAL ANALYSIS")
        print("="*70)
        
        # Check for different coordinate systems
        spatial_coords = []
        for coord in ['rlon', 'rlat', 'lon', 'lat', 'x', 'y']:
            if coord in self.dataset.coords:
                spatial_coords.append(coord)
        
        print(f"Spatial coordinates found: {spatial_coords}")
        
        # Analyze rotated coordinates if present
        if 'rlon' in self.dataset.coords and 'rlat' in self.dataset.coords:
            rlon = self.dataset.rlon.values
            rlat = self.dataset.rlat.values
            
            print(f"\nRotated Longitude (rlon):")
            print(f"  Range: {rlon.min():.3f}° to {rlon.max():.3f}°")
            print(f"  Resolution: {np.diff(rlon).mean():.6f}°")
            print(f"  Grid points: {len(rlon)}")
            
            print(f"\nRotated Latitude (rlat):")
            print(f"  Range: {rlat.min():.3f}° to {rlat.max():.3f}°")
            print(f"  Resolution: {np.diff(rlat).mean():.6f}°")
            print(f"  Grid points: {len(rlat)}")
            
            # Store spatial info
            self.analysis_results['spatial'] = {
                'rlon_range': (float(rlon.min()), float(rlon.max())),
                'rlat_range': (float(rlat.min()), float(rlat.max())),
                'rlon_resolution': float(np.diff(rlon).mean()),
                'rlat_resolution': float(np.diff(rlat).mean()),
                'grid_size': (len(rlon), len(rlat))
            }
        
        # Check for regular lat/lon coordinates
        if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
            lon = self.dataset.lon.values
            lat = self.dataset.lat.values
            
            print(f"\nGeographic Coordinates:")
            print(f"  Longitude range: {lon.min():.3f}° to {lon.max():.3f}°")
            print(f"  Latitude range: {lat.min():.3f}° to {lat.max():.3f}°")
            
            # Estimate coverage area
            if lon.ndim == 2 and lat.ndim == 2:
                # Calculate approximate area coverage
                lon_span = lon.max() - lon.min()
                lat_span = lat.max() - lat.min()
                print(f"  Approximate coverage: {lon_span:.2f}° × {lat_span:.2f}°")
        
        # Perform detailed coordinate analysis
        self.analyze_coordinate_data()
    
    def analyze_coordinate_data(self):
        """Perform detailed analysis of coordinate data."""
        print("\n" + "="*70)
        print("DETAILED COORDINATE ANALYSIS")
        print("="*70)
        
        # Analyze rotated pole grid mapping
        if 'rotated_pole' in self.dataset.data_vars:
            self._analyze_rotated_pole_mapping()
        
        # Analyze coordinate transformations
        if 'rlon' in self.dataset.coords and 'rlat' in self.dataset.coords:
            if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
                self._analyze_coordinate_transformation()
        
        # Analyze grid properties
        self._analyze_grid_properties()
        
        # Analyze coordinate quality
        self._analyze_coordinate_quality()
    
    def _analyze_rotated_pole_mapping(self):
        """Analyze the rotated pole grid mapping."""
        print("\nRotated Pole Grid Mapping:")
        
        rotated_pole = self.dataset.rotated_pole
        
        # Check for grid mapping attributes
        if hasattr(rotated_pole, 'grid_mapping_name'):
            print(f"  Grid mapping: {rotated_pole.grid_mapping_name}")
        
        if hasattr(rotated_pole, 'grid_north_pole_latitude'):
            print(f"  North pole latitude: {rotated_pole.grid_north_pole_latitude}°")
        
        if hasattr(rotated_pole, 'grid_north_pole_longitude'):
            print(f"  North pole longitude: {rotated_pole.grid_north_pole_longitude}°")
        
        if hasattr(rotated_pole, 'north_pole_grid_longitude'):
            print(f"  North pole grid longitude: {rotated_pole.north_pole_grid_longitude}°")
        
        # Store rotated pole info
        self.analysis_results['rotated_pole_mapping'] = {}
        for attr in ['grid_mapping_name', 'grid_north_pole_latitude', 
                     'grid_north_pole_longitude', 'north_pole_grid_longitude']:
            if hasattr(rotated_pole, attr):
                self.analysis_results['rotated_pole_mapping'][attr] = getattr(rotated_pole, attr)
    
    def _analyze_coordinate_transformation(self):
        """Analyze the transformation between rotated and geographic coordinates."""
        print("\nCoordinate Transformation Analysis:")
        
        rlon = self.dataset.rlon.values
        rlat = self.dataset.rlat.values
        lon = self.dataset.lon.values
        lat = self.dataset.lat.values
        
        # Check coordinate consistency
        print(f"  Rotated grid shape: {len(rlon)} × {len(rlat)}")
        print(f"  Geographic grid shape: {lon.shape}")
        
        # Analyze coordinate mapping
        if lon.ndim == 2 and lat.ndim == 2:
            # Calculate coordinate gradients
            dlon_drlon = np.gradient(lon, axis=1)
            dlon_drlat = np.gradient(lon, axis=0)
            dlat_drlon = np.gradient(lat, axis=1)
            dlat_drlat = np.gradient(lat, axis=0)
            
            print(f"  Longitude gradient (∂lon/∂rlon): {dlon_drlon.mean():.6f} ± {dlon_drlon.std():.6f}")
            print(f"  Longitude gradient (∂lon/∂rlat): {dlon_drlat.mean():.6f} ± {dlon_drlat.std():.6f}")
            print(f"  Latitude gradient (∂lat/∂rlon): {dlat_drlon.mean():.6f} ± {dlat_drlon.std():.6f}")
            print(f"  Latitude gradient (∂lat/∂rlat): {dlat_drlat.mean():.6f} ± {dlat_drlat.std():.6f}")
            
            # Calculate grid cell areas (approximate)
            earth_radius = 6371000  # meters
            lat_rad = np.radians(lat)
            
            # Grid cell dimensions in meters
            dx = earth_radius * np.cos(lat_rad) * np.radians(dlon_drlon)
            dy = earth_radius * np.radians(dlat_drlat)
            
            cell_area = np.abs(dx * dy)  # m²
            
            print(f"  Grid cell area: {cell_area.mean()/1e6:.2f} ± {cell_area.std()/1e6:.2f} km²")
            print(f"  Min cell area: {cell_area.min()/1e6:.2f} km²")
            print(f"  Max cell area: {cell_area.max()/1e6:.2f} km²")
            
            # Store transformation info
            self.analysis_results['coordinate_transformation'] = {
                'dlon_drlon_mean': float(dlon_drlon.mean()),
                'dlon_drlon_std': float(dlon_drlon.std()),
                'dlon_drlat_mean': float(dlon_drlat.mean()),
                'dlon_drlat_std': float(dlon_drlat.std()),
                'dlat_drlon_mean': float(dlat_drlon.mean()),
                'dlat_drlon_std': float(dlat_drlon.std()),
                'dlat_drlat_mean': float(dlat_drlat.mean()),
                'dlat_drlat_std': float(dlat_drlat.std()),
                'cell_area_mean_km2': float(cell_area.mean()/1e6),
                'cell_area_std_km2': float(cell_area.std()/1e6),
                'cell_area_min_km2': float(cell_area.min()/1e6),
                'cell_area_max_km2': float(cell_area.max()/1e6)
            }
    
    def _analyze_grid_properties(self):
        """Analyze grid properties and regularity."""
        print("\nGrid Properties Analysis:")
        
        if 'rlon' in self.dataset.coords and 'rlat' in self.dataset.coords:
            rlon = self.dataset.rlon.values
            rlat = self.dataset.rlat.values
            
            # Check grid regularity
            rlon_diff = np.diff(rlon)
            rlat_diff = np.diff(rlat)
            
            print(f"  Rotated longitude spacing:")
            print(f"    Mean: {rlon_diff.mean():.6f}°")
            print(f"    Std: {rlon_diff.std():.8f}°")
            print(f"    Min: {rlon_diff.min():.6f}°")
            print(f"    Max: {rlon_diff.max():.6f}°")
            print(f"    Regular grid: {'Yes' if rlon_diff.std() < 1e-6 else 'No'}")
            
            print(f"  Rotated latitude spacing:")
            print(f"    Mean: {rlat_diff.mean():.6f}°")
            print(f"    Std: {rlat_diff.std():.8f}°")
            print(f"    Min: {rlat_diff.min():.6f}°")
            print(f"    Max: {rlat_diff.max():.6f}°")
            print(f"    Regular grid: {'Yes' if rlat_diff.std() < 1e-6 else 'No'}")
            
            # Store grid properties
            self.analysis_results['grid_properties'] = {
                'rlon_spacing_mean': float(rlon_diff.mean()),
                'rlon_spacing_std': float(rlon_diff.std()),
                'rlon_spacing_min': float(rlon_diff.min()),
                'rlon_spacing_max': float(rlon_diff.max()),
                'rlon_regular': bool(rlon_diff.std() < 1e-6),
                'rlat_spacing_mean': float(rlat_diff.mean()),
                'rlat_spacing_std': float(rlat_diff.std()),
                'rlat_spacing_min': float(rlat_diff.min()),
                'rlat_spacing_max': float(rlat_diff.max()),
                'rlat_regular': bool(rlat_diff.std() < 1e-6)
            }
        
        # Analyze geographic coordinate properties
        if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
            lon = self.dataset.lon.values
            lat = self.dataset.lat.values
            
            if lon.ndim == 2 and lat.ndim == 2:
                print(f"\nGeographic coordinate properties:")
                print(f"  Longitude variation:")
                print(f"    Along rows (constant rlat): {np.std(np.diff(lon, axis=1)):.6f}°")
                print(f"    Along columns (constant rlon): {np.std(np.diff(lon, axis=0)):.6f}°")
                print(f"  Latitude variation:")
                print(f"    Along rows (constant rlat): {np.std(np.diff(lat, axis=1)):.6f}°")
                print(f"    Along columns (constant rlon): {np.std(np.diff(lat, axis=0)):.6f}°")
    
    def _analyze_coordinate_quality(self):
        """Analyze coordinate data quality and potential issues."""
        print("\nCoordinate Quality Assessment:")
        
        # Check for missing coordinates
        coords_to_check = ['rlon', 'rlat', 'lon', 'lat']
        for coord in coords_to_check:
            if coord in self.dataset.coords:
                coord_data = self.dataset[coord].values
                if np.any(np.isnan(coord_data)):
                    nan_count = np.sum(np.isnan(coord_data))
                    total_count = coord_data.size
                    print(f"  {coord}: {nan_count}/{total_count} NaN values ({100*nan_count/total_count:.1f}%)")
                else:
                    print(f"  {coord}: No missing values")
        
        # Check coordinate bounds
        if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
            lon = self.dataset.lon.values
            lat = self.dataset.lat.values
            
            print(f"\nCoordinate bounds check:")
            print(f"  Longitude range: {lon.min():.3f}° to {lon.max():.3f}°")
            if lon.min() < -180 or lon.max() > 360:
                print(f"    WARNING: Longitude values outside expected range [-180, 360]")
            
            print(f"  Latitude range: {lat.min():.3f}° to {lat.max():.3f}°")
            if lat.min() < -90 or lat.max() > 90:
                print(f"    WARNING: Latitude values outside expected range [-90, 90]")
        
        # Check for coordinate monotonicity
        if 'rlon' in self.dataset.coords and 'rlat' in self.dataset.coords:
            rlon = self.dataset.rlon.values
            rlat = self.dataset.rlat.values
            
            rlon_monotonic = np.all(np.diff(rlon) > 0) or np.all(np.diff(rlon) < 0)
            rlat_monotonic = np.all(np.diff(rlat) > 0) or np.all(np.diff(rlat) < 0)
            
            print(f"\nCoordinate monotonicity:")
            print(f"  Rotated longitude monotonic: {'Yes' if rlon_monotonic else 'No'}")
            print(f"  Rotated latitude monotonic: {'Yes' if rlat_monotonic else 'No'}")
    
    def analyze_precipitation_data(self):
        """Analyze precipitation data specifically."""
        if self.dataset is None:
            return
        
        print("\n" + "="*70)
        print("PRECIPITATION DATA ANALYSIS")
        print("="*70)
        
        # Find the main precipitation variable
        precip_var = None
        for var_name in self.dataset.data_vars:
            if var_name != 'rotated_pole':  # Skip grid mapping variable
                precip_var = var_name
                break
        
        if precip_var is None:
            print("No precipitation variable found!")
            return
        
        var = self.dataset[precip_var]
        print(f"Analyzing variable: {precip_var}")
        print(f"Shape: {var.shape}")
        print(f"Dimensions: {var.dims}")
        
        # Calculate statistics
        var_values = var.values
        valid_mask = ~np.isnan(var_values)
        
        if np.any(valid_mask):
            valid_values = var_values[valid_mask]
            
            print(f"\nData Coverage:")
            print(f"  Valid points: {np.sum(valid_mask):,} / {var_values.size:,} ({100*np.sum(valid_mask)/var_values.size:.1f}%)")
            print(f"  Missing/Invalid points: {var_values.size - np.sum(valid_mask):,} ({100*(1-np.sum(valid_mask)/var_values.size):.1f}%)")
            
            print(f"\nPrecipitation Statistics:")
            print(f"  Min: {valid_values.min():.6f} m ({valid_values.min()*1000:.3f} mm)")
            print(f"  Max: {valid_values.max():.6f} m ({valid_values.max()*1000:.3f} mm)")
            print(f"  Mean: {valid_values.mean():.6f} m ({valid_values.mean()*1000:.3f} mm)")
            print(f"  Median: {np.median(valid_values):.6f} m ({np.median(valid_values)*1000:.3f} mm)")
            print(f"  Std: {valid_values.std():.6f} m ({valid_values.std()*1000:.3f} mm)")
            
            # Precipitation intensity categories
            zero_precip = np.sum(valid_values == 0)
            light_precip = np.sum((valid_values > 0) & (valid_values <= 0.001))  # 0-1mm
            moderate_precip = np.sum((valid_values > 0.001) & (valid_values <= 0.010))  # 1-10mm
            heavy_precip = np.sum(valid_values > 0.010)  # >10mm
            
            print(f"\nPrecipitation Categories:")
            print(f"  No precipitation (0 mm): {zero_precip:,} ({100*zero_precip/len(valid_values):.1f}%)")
            print(f"  Light (0-1 mm): {light_precip:,} ({100*light_precip/len(valid_values):.1f}%)")
            print(f"  Moderate (1-10 mm): {moderate_precip:,} ({100*moderate_precip/len(valid_values):.1f}%)")
            print(f"  Heavy (>10 mm): {heavy_precip:,} ({100*heavy_precip/len(valid_values):.1f}%)")
            
            # Store precipitation statistics
            self.analysis_results[precip_var] = {
                'shape': var.shape,
                'valid_points': int(np.sum(valid_mask)),
                'total_points': int(var_values.size),
                'valid_percentage': float(100*np.sum(valid_mask)/var_values.size),
                'min_m': float(valid_values.min()),
                'max_m': float(valid_values.max()),
                'mean_m': float(valid_values.mean()),
                'median_m': float(np.median(valid_values)),
                'std_m': float(valid_values.std()),
                'min_mm': float(valid_values.min()*1000),
                'max_mm': float(valid_values.max()*1000),
                'mean_mm': float(valid_values.mean()*1000),
                'median_mm': float(np.median(valid_values)*1000),
                'std_mm': float(valid_values.std()*1000),
                'zero_precip_count': int(zero_precip),
                'light_precip_count': int(light_precip),
                'moderate_precip_count': int(moderate_precip),
                'heavy_precip_count': int(heavy_precip)
            }
            
            # Temporal analysis if time dimension exists
            if 'time' in var.dims:
                self._analyze_temporal_precipitation_patterns(var, precip_var)
                
        else:
            print("No valid data points found!")
    
    def _analyze_temporal_precipitation_patterns(self, var, var_name):
        """Analyze temporal patterns in precipitation data."""
        print(f"\nTemporal Precipitation Patterns:")
        
        # Calculate daily totals (assuming hourly data)
        if 'time' in var.dims:
            # Calculate spatial mean for each time step
            time_series = var.mean(dim=[d for d in var.dims if d != 'time'], skipna=True)
            
            # Convert to pandas for easier time analysis
            time_values = pd.to_datetime(var.time.values)
            precip_series = pd.Series(time_series.values, index=time_values)
            
            # Remove NaN values
            precip_series = precip_series.dropna()
            
            if len(precip_series) > 0:
                # Daily statistics
                daily_precip = precip_series.resample('D').sum()
                print(f"  Daily precipitation (spatial mean):")
                print(f"    Mean daily total: {daily_precip.mean()*1000:.3f} mm")
                print(f"    Max daily total: {daily_precip.max()*1000:.3f} mm")
                print(f"    Days with precipitation > 1mm: {(daily_precip*1000 > 1).sum()}")
                print(f"    Days with precipitation > 10mm: {(daily_precip*1000 > 10).sum()}")
                
                # Monthly statistics
                monthly_precip = precip_series.resample('M').sum()
                print(f"  Monthly precipitation (spatial mean):")
                print(f"    Mean monthly total: {monthly_precip.mean()*1000:.1f} mm")
                print(f"    Max monthly total: {monthly_precip.max()*1000:.1f} mm")
                print(f"    Min monthly total: {monthly_precip.min()*1000:.1f} mm")
    
    def create_enhanced_visualizations(self, output_dir="output_plots"):
        """Create enhanced visualizations of the precipitation data."""
        if self.dataset is None:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("CREATING ENHANCED VISUALIZATIONS")
        print("="*70)
        
        # Find the main precipitation variable
        precip_var = None
        for var_name in self.dataset.data_vars:
            if var_name != 'rotated_pole':
                precip_var = var_name
                break
        
        if precip_var is None:
            print("No precipitation variable found for visualization!")
            return
        
        data_var = self.dataset[precip_var]
        
        # 1. Time series analysis
        if 'time' in data_var.dims:
            self._plot_precipitation_time_series(data_var, precip_var, output_dir)
        
        # 2. Spatial precipitation maps
        self._plot_precipitation_spatial_maps(data_var, precip_var, output_dir)
        
        # 3. Precipitation distribution analysis
        self._plot_precipitation_distribution(data_var, precip_var, output_dir)
        
        # 4. Data availability and coverage
        self._plot_data_coverage(data_var, precip_var, output_dir)
        
        # 5. Seasonal analysis
        if 'time' in data_var.dims:
            self._plot_seasonal_analysis(data_var, precip_var, output_dir)
        
        # 6. Coordinate analysis visualizations
        self._plot_coordinate_analysis(output_dir)
        
        print(f"✓ Enhanced visualizations saved to {output_dir}/")
    
    def _plot_coordinate_analysis(self, output_dir):
        """Create coordinate analysis visualizations."""
        try:
            print(f"  Creating coordinate analysis plots...")
            
            # Plot coordinate grids
            if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
                self._plot_coordinate_grids(output_dir)
            
            # Plot coordinate transformations
            if ('rlon' in self.dataset.coords and 'rlat' in self.dataset.coords and
                'lon' in self.dataset.coords and 'lat' in self.dataset.coords):
                self._plot_coordinate_transformation_analysis(output_dir)
            
            # Plot grid properties
            if 'rlon' in self.dataset.coords and 'rlat' in self.dataset.coords:
                self._plot_grid_properties(output_dir)
            
            print(f"  ✓ Coordinate analysis plots saved")
        except Exception as e:
            print(f"  ✗ Error creating coordinate plots: {e}")
    
    def _plot_coordinate_grids(self, output_dir):
        """Plot coordinate grid visualizations."""
        lon = self.dataset.lon.values
        lat = self.dataset.lat.values
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Longitude grid
        im1 = ax1.imshow(lon, cmap='viridis', aspect='auto')
        ax1.set_title('Longitude Grid')
        ax1.set_xlabel('Grid X (rlon index)')
        ax1.set_ylabel('Grid Y (rlat index)')
        plt.colorbar(im1, ax=ax1, label='Longitude (°)')
        
        # Latitude grid
        im2 = ax2.imshow(lat, cmap='plasma', aspect='auto')
        ax2.set_title('Latitude Grid')
        ax2.set_xlabel('Grid X (rlon index)')
        ax2.set_ylabel('Grid Y (rlat index)')
        plt.colorbar(im2, ax=ax2, label='Latitude (°)')
        
        # Geographic coordinate scatter
        ax3.scatter(lon.flatten(), lat.flatten(), c='blue', alpha=0.6, s=1)
        ax3.set_xlabel('Longitude (°)')
        ax3.set_ylabel('Latitude (°)')
        ax3.set_title('Geographic Coordinate Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Grid cell boundaries
        if 'rlon' in self.dataset.coords and 'rlat' in self.dataset.coords:
            rlon = self.dataset.rlon.values
            rlat = self.dataset.rlat.values
            
            # Plot every 5th grid line for clarity
            step = max(1, len(rlon) // 10)
            for i in range(0, len(rlat), step):
                ax4.plot(lon[i, :], lat[i, :], 'b-', alpha=0.5, linewidth=0.5)
            for j in range(0, len(rlon), step):
                ax4.plot(lon[:, j], lat[:, j], 'r-', alpha=0.5, linewidth=0.5)
            
            ax4.set_xlabel('Longitude (°)')
            ax4.set_ylabel('Latitude (°)')
            ax4.set_title('Grid Cell Structure')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/coordinate_grids.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_coordinate_transformation_analysis(self, output_dir):
        """Plot coordinate transformation analysis."""
        rlon = self.dataset.rlon.values
        rlat = self.dataset.rlat.values
        lon = self.dataset.lon.values
        lat = self.dataset.lat.values
        
        # Calculate gradients
        dlon_drlon = np.gradient(lon, axis=1)
        dlon_drlat = np.gradient(lon, axis=0)
        dlat_drlon = np.gradient(lat, axis=1)
        dlat_drlat = np.gradient(lat, axis=0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Longitude gradients
        im1 = ax1.imshow(dlon_drlon, cmap='RdBu_r', aspect='auto')
        ax1.set_title('∂lon/∂rlon')
        ax1.set_xlabel('Grid X (rlon index)')
        ax1.set_ylabel('Grid Y (rlat index)')
        plt.colorbar(im1, ax=ax1, label='°/°')
        
        im2 = ax2.imshow(dlon_drlat, cmap='RdBu_r', aspect='auto')
        ax2.set_title('∂lon/∂rlat')
        ax2.set_xlabel('Grid X (rlon index)')
        ax2.set_ylabel('Grid Y (rlat index)')
        plt.colorbar(im2, ax=ax2, label='°/°')
        
        # Latitude gradients
        im3 = ax3.imshow(dlat_drlon, cmap='RdBu_r', aspect='auto')
        ax3.set_title('∂lat/∂rlon')
        ax3.set_xlabel('Grid X (rlon index)')
        ax3.set_ylabel('Grid Y (rlat index)')
        plt.colorbar(im3, ax=ax3, label='°/°')
        
        im4 = ax4.imshow(dlat_drlat, cmap='RdBu_r', aspect='auto')
        ax4.set_title('∂lat/∂rlat')
        ax4.set_xlabel('Grid X (rlon index)')
        ax4.set_ylabel('Grid Y (rlat index)')
        plt.colorbar(im4, ax=ax4, label='°/°')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/coordinate_gradients.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Grid cell areas
        earth_radius = 6371000  # meters
        lat_rad = np.radians(lat)
        dx = earth_radius * np.cos(lat_rad) * np.radians(dlon_drlon)
        dy = earth_radius * np.radians(dlat_drlat)
        cell_area = np.abs(dx * dy) / 1e6  # km²
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(cell_area, cmap='viridis', aspect='auto')
        plt.title('Grid Cell Areas')
        plt.xlabel('Grid X (rlon index)')
        plt.ylabel('Grid Y (rlat index)')
        plt.colorbar(im, label='Area (km²)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/grid_cell_areas.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_grid_properties(self, output_dir):
        """Plot grid properties analysis."""
        rlon = self.dataset.rlon.values
        rlat = self.dataset.rlat.values
        
        # Grid spacing analysis
        rlon_diff = np.diff(rlon)
        rlat_diff = np.diff(rlat)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Rotated longitude spacing
        ax1.plot(rlon[1:], rlon_diff, 'b-', linewidth=2)
        ax1.set_xlabel('Rotated Longitude (°)')
        ax1.set_ylabel('Grid Spacing (°)')
        ax1.set_title('Rotated Longitude Grid Spacing')
        ax1.grid(True, alpha=0.3)
        
        # Rotated latitude spacing
        ax2.plot(rlat[1:], rlat_diff, 'r-', linewidth=2)
        ax2.set_xlabel('Rotated Latitude (°)')
        ax2.set_ylabel('Grid Spacing (°)')
        ax2.set_title('Rotated Latitude Grid Spacing')
        ax2.grid(True, alpha=0.3)
        
        # Grid spacing histograms
        ax3.hist(rlon_diff, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Grid Spacing (°)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Rotated Longitude Spacing Distribution')
        ax3.grid(True, alpha=0.3)
        
        ax4.hist(rlat_diff, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Grid Spacing (°)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Rotated Latitude Spacing Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/grid_spacing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create coordinate system comparison
        if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
            lon = self.dataset.lon.values
            lat = self.dataset.lat.values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Rotated coordinate system
            rlon_2d, rlat_2d = np.meshgrid(rlon, rlat)
            ax1.scatter(rlon_2d.flatten(), rlat_2d.flatten(), c='blue', alpha=0.6, s=1)
            ax1.set_xlabel('Rotated Longitude (°)')
            ax1.set_ylabel('Rotated Latitude (°)')
            ax1.set_title('Rotated Coordinate System')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Geographic coordinate system
            ax2.scatter(lon.flatten(), lat.flatten(), c='red', alpha=0.6, s=1)
            ax2.set_xlabel('Longitude (°)')
            ax2.set_ylabel('Latitude (°)')
            ax2.set_title('Geographic Coordinate System')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/coordinate_systems_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_precipitation_time_series(self, data_var, var_name, output_dir):
        """Create precipitation time series plots."""
        try:
            # Calculate spatial mean for each time step
            time_series = data_var.mean(dim=[d for d in data_var.dims if d != 'time'], skipna=True)
            
            # Convert to mm for better readability
            time_series_mm = time_series * 1000
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Full time series
            time_series_mm.plot(ax=ax1)
            ax1.set_title(f'Precipitation Time Series (Spatial Mean) - {var_name}')
            ax1.set_ylabel('Precipitation (mm)')
            ax1.grid(True, alpha=0.3)
            
            # Daily aggregated time series
            time_values = pd.to_datetime(data_var.time.values)
            precip_series = pd.Series(time_series_mm.values, index=time_values)
            daily_precip = precip_series.resample('D').sum()
            
            daily_precip.plot(ax=ax2, alpha=0.7)
            ax2.set_title('Daily Precipitation Totals (Spatial Mean)')
            ax2.set_ylabel('Daily Precipitation (mm)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/precipitation_time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Precipitation time series plots saved")
        except Exception as e:
            print(f"  ✗ Error creating time series plots: {e}")
    
    def _plot_precipitation_spatial_maps(self, data_var, var_name, output_dir):
        """Create spatial precipitation maps."""
        try:
            # Select different time periods for comparison
            n_times = len(data_var.time)
            time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, time_idx in enumerate(time_indices):
                if i >= len(axes):
                    break
                    
                spatial_data = data_var.isel(time=time_idx) * 1000  # Convert to mm
                time_str = str(data_var.time.values[time_idx])[:19]
                
                im = spatial_data.plot(ax=axes[i], cmap='Blues', add_colorbar=False, 
                                     vmin=0, vmax=spatial_data.quantile(0.95))
                axes[i].set_title(f'Precipitation: {time_str}')
                axes[i].set_xlabel('Grid X')
                axes[i].set_ylabel('Grid Y')
            
            # Remove empty subplot
            if len(time_indices) < len(axes):
                fig.delaxes(axes[-1])
            
            # Add colorbar
            plt.colorbar(im, ax=axes, label='Precipitation (mm)', shrink=0.8)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/precipitation_spatial_maps.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create mean precipitation map
            plt.figure(figsize=(10, 8))
            mean_precip = data_var.mean(dim='time', skipna=True) * 1000
            mean_precip.plot(cmap='Blues', cbar_kwargs={'label': 'Mean Precipitation (mm)'})
            plt.title('Mean Precipitation Distribution')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/mean_precipitation_map.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Spatial precipitation maps saved")
        except Exception as e:
            print(f"  ✗ Error creating spatial maps: {e}")
    
    def _plot_precipitation_distribution(self, data_var, var_name, output_dir):
        """Create precipitation distribution plots."""
        try:
            # Get valid precipitation data in mm
            flat_data = data_var.values.flatten()
            valid_data = flat_data[~np.isnan(flat_data)] * 1000  # Convert to mm
            
            if len(valid_data) == 0:
                print(f"  ✗ No valid data for distribution plots")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Histogram of all data
            ax1.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Precipitation (mm)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of All Precipitation Values')
            ax1.grid(True, alpha=0.3)
            
            # Histogram of non-zero precipitation
            non_zero_data = valid_data[valid_data > 0]
            if len(non_zero_data) > 0:
                ax2.hist(non_zero_data, bins=50, alpha=0.7, edgecolor='black', color='orange')
                ax2.set_xlabel('Precipitation (mm)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Non-Zero Precipitation')
                ax2.grid(True, alpha=0.3)
            
            # Box plot
            ax3.boxplot(valid_data)
            ax3.set_ylabel('Precipitation (mm)')
            ax3.set_title('Box Plot of Precipitation')
            ax3.grid(True, alpha=0.3)
            
            # Log-scale histogram for better visualization of range
            if len(non_zero_data) > 0:
                ax4.hist(non_zero_data, bins=50, alpha=0.7, edgecolor='black', color='green')
                ax4.set_xlabel('Precipitation (mm)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Non-Zero Precipitation (Log Scale)')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/precipitation_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Precipitation distribution plots saved")
        except Exception as e:
            print(f"  ✗ Error creating distribution plots: {e}")
    
    def _plot_data_coverage(self, data_var, var_name, output_dir):
        """Create data coverage visualization."""
        try:
            # Data availability over time
            if 'time' in data_var.dims:
                valid_mask = ~np.isnan(data_var.values)
                time_axis = data_var.dims.index('time')
                other_axes = tuple(i for i in range(len(data_var.dims)) if i != time_axis)
                
                valid_percentage = np.mean(valid_mask, axis=other_axes) * 100
                
                plt.figure(figsize=(15, 6))
                plt.plot(data_var.time.values, valid_percentage, alpha=0.7, linewidth=1)
                plt.title(f'Data Availability Over Time - {var_name}')
                plt.xlabel('Time')
                plt.ylabel('Percentage of Valid Data Points (%)')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/data_availability.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Data availability plot saved")
            
            # Spatial coverage map
            valid_points_per_location = np.sum(~np.isnan(data_var.values), axis=0)
            total_time_steps = data_var.shape[0]
            coverage_percentage = (valid_points_per_location / total_time_steps) * 100
            
            plt.figure(figsize=(10, 8))
            plt.imshow(coverage_percentage, cmap='RdYlBu', vmin=0, vmax=100, aspect='auto')
            plt.colorbar(label='Data Coverage (%)')
            plt.title('Spatial Data Coverage')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/spatial_data_coverage.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Spatial coverage plot saved")
            
        except Exception as e:
            print(f"  ✗ Error creating coverage plots: {e}")
    
    def _plot_seasonal_analysis(self, data_var, var_name, output_dir):
        """Create seasonal analysis plots."""
        try:
            # Calculate monthly means
            time_values = pd.to_datetime(data_var.time.values)
            
            # Group by month and calculate spatial mean
            monthly_data = []
            months = []
            
            for month in range(1, 13):
                month_mask = pd.Series(time_values).dt.month == month
                if np.any(month_mask):
                    month_data = data_var.isel(time=month_mask).mean(dim='time', skipna=True)
                    monthly_data.append(month_data * 1000)  # Convert to mm
                    months.append(month)
            
            if len(monthly_data) > 0:
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                axes = axes.flatten()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                vmin = min([data.min().values for data in monthly_data])
                vmax = max([data.max().values for data in monthly_data])
                
                for i, (month_data, month) in enumerate(zip(monthly_data, months)):
                    im = month_data.plot(ax=axes[i], cmap='Blues', add_colorbar=False,
                                       vmin=vmin, vmax=vmax)
                    axes[i].set_title(f'{month_names[month-1]} Mean Precipitation')
                    axes[i].set_xlabel('Grid X')
                    axes[i].set_ylabel('Grid Y')
                
                # Remove empty subplots
                for i in range(len(monthly_data), len(axes)):
                    fig.delaxes(axes[i])
                
                # Add colorbar
                plt.colorbar(im, ax=axes[:len(monthly_data)], label='Precipitation (mm)', shrink=0.8)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/seasonal_precipitation.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Seasonal analysis plots saved")
            
        except Exception as e:
            print(f"  ✗ Error creating seasonal plots: {e}")
    
    def export_comprehensive_report(self, output_file="casr_precipitation_analysis_report.txt"):
        """Export a comprehensive analysis report."""
        try:
            with open(output_file, 'w') as f:
                f.write("CaSR Precipitation Data Analysis Report\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"File analyzed: {self.file_path}\n\n")
                
                # Dataset overview
                f.write("DATASET OVERVIEW\n")
                f.write("-" * 30 + "\n")
                f.write(f"File size: {os.path.getsize(self.file_path) / (1024**2):.2f} MB\n")
                f.write(f"Dimensions: {dict(self.dataset.dims)}\n")
                f.write(f"Variables: {list(self.dataset.data_vars)}\n")
                f.write(f"Coordinates: {list(self.dataset.coords)}\n\n")
                
                # Analysis results
                for key, value in self.analysis_results.items():
                    f.write(f"{key.upper().replace('_', ' ')} ANALYSIS\n")
                    f.write("-" * 30 + "\n")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"  {k.replace('_', ' ').title()}: {v}\n")
                    else:
                        f.write(f"  {value}\n")
                    f.write("\n")
                
                # Summary insights
                f.write("KEY INSIGHTS\n")
                f.write("-" * 30 + "\n")
                
                # Find precipitation variable for insights
                precip_var = None
                for var_name in self.dataset.data_vars:
                    if var_name != 'rotated_pole':
                        precip_var = var_name
                        break
                
                if precip_var and precip_var in self.analysis_results:
                    stats = self.analysis_results[precip_var]
                    f.write(f"- Dataset contains {stats['valid_points']:,} valid precipitation measurements\n")
                    f.write(f"- Data coverage: {stats['valid_percentage']:.1f}% of total grid points\n")
                    f.write(f"- Mean precipitation: {stats['mean_mm']:.3f} mm\n")
                    f.write(f"- Maximum precipitation: {stats['max_mm']:.3f} mm\n")
                    f.write(f"- {stats['zero_precip_count']:,} measurements show no precipitation\n")
                    f.write(f"- {stats['heavy_precip_count']:,} measurements show heavy precipitation (>10mm)\n")
                
                f.write("\nRECOMMENDations\n")
                f.write("-" * 30 + "\n")
                f.write("- Consider gap-filling techniques for missing data points\n")
                f.write("- Analyze seasonal patterns for climate studies\n")
                f.write("- Investigate spatial patterns for regional analysis\n")
                f.write("- Consider temporal aggregation for trend analysis\n")
            
            print(f"✓ Comprehensive report saved to {output_file}")
        except Exception as e:
            print(f"✗ Error creating comprehensive report: {e}")
    
    def close(self):
        """Close the dataset."""
        if self.dataset is not None:
            self.dataset.close()
            print("Dataset closed.")


def main():
    """Main function to run the improved analysis."""
    # File path
    file_path = r"data\input_data\CaSR_SWE\CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1980-1983.nc"
    
    print("CaSR Precipitation Data Analysis (Improved)")
    print("=" * 60)
    print(f"Target file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("Please check the file path and try again.")
        return
    
    # Initialize analyzer
    analyzer = CaSRDataAnalyzer(file_path)
    
    try:
        # Load data
        if not analyzer.load_data():
            return
        
        # Perform comprehensive analysis
        analyzer.display_basic_info()
        analyzer.analyze_temporal_coverage()
        analyzer.analyze_spatial_coverage()
        analyzer.analyze_precipitation_data()
        
        # Create enhanced visualizations
        analyzer.create_enhanced_visualizations()
        
        # Export comprehensive report
        analyzer.export_comprehensive_report()
        
        print("\n" + "="*70)
        print("ENHANCED ANALYSIS COMPLETE!")
        print("="*70)
        print("Check the following outputs:")
        print("- Enhanced visualizations in 'output_plots/' directory")
        print("- Comprehensive report: 'casr_precipitation_analysis_report.txt'")
        print("\nKey findings:")
        
        # Display key findings
        if hasattr(analyzer, 'analysis_results'):
            for var_name in analyzer.dataset.data_vars:
                if var_name != 'rotated_pole' and var_name in analyzer.analysis_results:
                    stats = analyzer.analysis_results[var_name]
                    print(f"- {stats['valid_points']:,} valid precipitation measurements")
                    print(f"- Data coverage: {stats['valid_percentage']:.1f}%")
                    print(f"- Mean precipitation: {stats['mean_mm']:.3f} mm")
                    print(f"- Maximum precipitation: {stats['max_mm']:.3f} mm")
                    break
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        analyzer.close()


if __name__ == "__main__":
    main()
