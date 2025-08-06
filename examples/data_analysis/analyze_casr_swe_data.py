#!/usr/bin/env python3
"""
CaSR SWE Data Analysis Script

This script analyzes Canadian Snow and Sea Ice Service Reanalysis (CaSR) 
Snow Water Equivalent (SWE) data from NetCDF files.

Author: Data Analysis Script
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

class CaSRSWEAnalyzer:
    """
    A class to analyze CaSR SWE NetCDF data files.
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
        
        print("\n" + "="*60)
        print("DATASET BASIC INFORMATION")
        print("="*60)
        
        # File information
        print(f"File: {os.path.basename(self.file_path)}")
        print(f"File size: {os.path.getsize(self.file_path) / (1024**2):.2f} MB")
        
        # Dataset dimensions
        print(f"\nDimensions:")
        for dim, size in self.dataset.dims.items():
            print(f"  {dim}: {size}")
        
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
            print(f"  {attr}: {value}")
    
    def analyze_temporal_coverage(self):
        """Analyze temporal coverage of the dataset."""
        if self.dataset is None:
            return
        
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS")
        print("="*60)
        
        if 'time' in self.dataset.coords:
            time_coord = self.dataset.time
            
            # Convert to pandas datetime for easier handling
            time_values = pd.to_datetime(time_coord.values)
            
            print(f"Time range: {time_values.min()} to {time_values.max()}")
            print(f"Number of time steps: {len(time_values)}")
            print(f"Time frequency: {pd.infer_freq(time_values)}")
            
            # Store temporal info
            self.analysis_results['temporal'] = {
                'start_date': time_values.min(),
                'end_date': time_values.max(),
                'n_timesteps': len(time_values),
                'frequency': pd.infer_freq(time_values)
            }
            
            # Monthly distribution
            monthly_counts = time_values.to_series().dt.month.value_counts().sort_index()
            print(f"\nMonthly distribution:")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, count in monthly_counts.items():
                print(f"  {month_names[month-1]}: {count} timesteps")
        else:
            print("No time coordinate found in dataset.")
    
    def analyze_spatial_coverage(self):
        """Analyze spatial coverage and grid information."""
        if self.dataset is None:
            return
        
        print("\n" + "="*60)
        print("SPATIAL ANALYSIS")
        print("="*60)
        
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
            print(f"  Range: {rlon.min():.3f} to {rlon.max():.3f}")
            print(f"  Resolution: {np.diff(rlon).mean():.6f}")
            print(f"  Grid points: {len(rlon)}")
            
            print(f"\nRotated Latitude (rlat):")
            print(f"  Range: {rlat.min():.3f} to {rlat.max():.3f}")
            print(f"  Resolution: {np.diff(rlat).mean():.6f}")
            print(f"  Grid points: {len(rlat)}")
            
            # Store spatial info
            self.analysis_results['spatial'] = {
                'rlon_range': (rlon.min(), rlon.max()),
                'rlat_range': (rlat.min(), rlat.max()),
                'rlon_resolution': np.diff(rlon).mean(),
                'rlat_resolution': np.diff(rlat).mean(),
                'grid_size': (len(rlon), len(rlat))
            }
        
        # Check for regular lat/lon coordinates
        if 'lon' in self.dataset.coords and 'lat' in self.dataset.coords:
            lon = self.dataset.lon.values
            lat = self.dataset.lat.values
            
            print(f"\nLongitude:")
            print(f"  Range: {lon.min():.3f} to {lon.max():.3f}")
            
            print(f"\nLatitude:")
            print(f"  Range: {lat.min():.3f} to {lat.max():.3f}")
    
    def analyze_data_variables(self):
        """Analyze the main data variables in the dataset."""
        if self.dataset is None:
            return
        
        print("\n" + "="*60)
        print("DATA VARIABLE ANALYSIS")
        print("="*60)
        
        for var_name in self.dataset.data_vars:
            var = self.dataset[var_name]
            print(f"\nVariable: {var_name}")
            print(f"  Shape: {var.shape}")
            print(f"  Dimensions: {var.dims}")
            
            # Calculate statistics
            var_values = var.values
            valid_mask = ~np.isnan(var_values)
            
            if np.any(valid_mask):
                valid_values = var_values[valid_mask]
                
                print(f"  Statistics:")
                print(f"    Valid points: {np.sum(valid_mask):,} / {var_values.size:,} ({100*np.sum(valid_mask)/var_values.size:.1f}%)")
                print(f"    Min: {valid_values.min():.6f}")
                print(f"    Max: {valid_values.max():.6f}")
                print(f"    Mean: {valid_values.mean():.6f}")
                print(f"    Std: {valid_values.std():.6f}")
                print(f"    Median: {np.median(valid_values):.6f}")
                
                # Store variable statistics
                self.analysis_results[var_name] = {
                    'shape': var.shape,
                    'valid_points': int(np.sum(valid_mask)),
                    'total_points': int(var_values.size),
                    'valid_percentage': float(100*np.sum(valid_mask)/var_values.size),
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'median': float(np.median(valid_values))
                }
                
                # Check for units and other attributes
                if hasattr(var, 'units'):
                    print(f"    Units: {var.units}")
                if hasattr(var, 'long_name'):
                    print(f"    Description: {var.long_name}")
            else:
                print(f"  No valid data points found!")
    
    def create_visualizations(self, output_dir="output_plots"):
        """Create visualizations of the data."""
        if self.dataset is None:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Get the main data variable (usually the first one)
        main_var = list(self.dataset.data_vars)[0]
        data_var = self.dataset[main_var]
        
        # 1. Time series plot (if time dimension exists)
        if 'time' in data_var.dims:
            self._plot_time_series(data_var, main_var, output_dir)
        
        # 2. Spatial plots
        self._plot_spatial_maps(data_var, main_var, output_dir)
        
        # 3. Statistical distribution
        self._plot_data_distribution(data_var, main_var, output_dir)
        
        # 4. Data availability plot
        self._plot_data_availability(data_var, main_var, output_dir)
        
        print(f"✓ Visualizations saved to {output_dir}/")
    
    def _plot_time_series(self, data_var, var_name, output_dir):
        """Create time series plots."""
        try:
            # Calculate spatial mean for each time step
            if len(data_var.dims) > 1:
                time_series = data_var.mean(dim=[d for d in data_var.dims if d != 'time'])
            else:
                time_series = data_var
            
            plt.figure(figsize=(12, 6))
            time_series.plot()
            plt.title(f'Time Series of {var_name} (Spatial Mean)')
            plt.xlabel('Time')
            plt.ylabel(f'{var_name} ({getattr(data_var, "units", "units")})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/time_series_{var_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Time series plot saved")
        except Exception as e:
            print(f"  ✗ Error creating time series plot: {e}")
    
    def _plot_spatial_maps(self, data_var, var_name, output_dir):
        """Create spatial maps."""
        try:
            # Select a representative time slice (middle of dataset)
            if 'time' in data_var.dims:
                mid_time = len(data_var.time) // 2
                spatial_data = data_var.isel(time=mid_time)
                time_str = str(data_var.time.values[mid_time])[:10]
            else:
                spatial_data = data_var
                time_str = "static"
            
            plt.figure(figsize=(10, 8))
            
            # Use pcolormesh for better performance with large datasets
            if hasattr(spatial_data, 'plot'):
                spatial_data.plot(cmap='viridis', add_colorbar=True)
            else:
                plt.imshow(spatial_data.values, cmap='viridis', aspect='auto')
                plt.colorbar()
            
            plt.title(f'{var_name} Spatial Distribution ({time_str})')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/spatial_map_{var_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Spatial map saved")
        except Exception as e:
            print(f"  ✗ Error creating spatial map: {e}")
    
    def _plot_data_distribution(self, data_var, var_name, output_dir):
        """Create data distribution plots."""
        try:
            # Flatten the data and remove NaN values
            flat_data = data_var.values.flatten()
            valid_data = flat_data[~np.isnan(flat_data)]
            
            if len(valid_data) == 0:
                print(f"  ✗ No valid data for distribution plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel(f'{var_name} ({getattr(data_var, "units", "units")})')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Distribution of {var_name}')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(valid_data)
            ax2.set_ylabel(f'{var_name} ({getattr(data_var, "units", "units")})')
            ax2.set_title(f'Box Plot of {var_name}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/distribution_{var_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Distribution plots saved")
        except Exception as e:
            print(f"  ✗ Error creating distribution plots: {e}")
    
    def _plot_data_availability(self, data_var, var_name, output_dir):
        """Create data availability visualization."""
        try:
            # Create a mask of valid data
            valid_mask = ~np.isnan(data_var.values)
            
            if 'time' in data_var.dims:
                # Calculate percentage of valid data for each time step
                time_axis = data_var.dims.index('time')
                other_axes = tuple(i for i in range(len(data_var.dims)) if i != time_axis)
                
                valid_percentage = np.mean(valid_mask, axis=other_axes) * 100
                
                plt.figure(figsize=(12, 6))
                plt.plot(data_var.time.values, valid_percentage, marker='o', markersize=3)
                plt.title(f'Data Availability Over Time - {var_name}')
                plt.xlabel('Time')
                plt.ylabel('Percentage of Valid Data Points (%)')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/data_availability_{var_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Data availability plot saved")
            else:
                print(f"  - No time dimension for availability plot")
                
        except Exception as e:
            print(f"  ✗ Error creating data availability plot: {e}")
    
    def export_summary_report(self, output_file="casr_swe_analysis_report.txt"):
        """Export a summary report of the analysis."""
        try:
            with open(output_file, 'w') as f:
                f.write("CaSR SWE Data Analysis Report\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"File analyzed: {self.file_path}\n\n")
                
                # Dataset info
                f.write("Dataset Information:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Dimensions: {dict(self.dataset.dims)}\n")
                f.write(f"Variables: {list(self.dataset.data_vars)}\n")
                f.write(f"Coordinates: {list(self.dataset.coords)}\n\n")
                
                # Analysis results
                for key, value in self.analysis_results.items():
                    f.write(f"{key.upper()} Analysis:\n")
                    f.write("-" * 20 + "\n")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"  {value}\n")
                    f.write("\n")
            
            print(f"✓ Summary report saved to {output_file}")
        except Exception as e:
            print(f"✗ Error creating summary report: {e}")
    
    def close(self):
        """Close the dataset."""
        if self.dataset is not None:
            self.dataset.close()
            print("Dataset closed.")


def main():
    """Main function to run the analysis."""
    # File path
    file_path = r"data\input_data\CaSR_SWE\CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1980-1983.nc"
    
    print("CaSR SWE Data Analysis")
    print("=" * 50)
    print(f"Target file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("Please check the file path and try again.")
        return
    
    # Initialize analyzer
    analyzer = CaSRSWEAnalyzer(file_path)
    
    try:
        # Load data
        if not analyzer.load_data():
            return
        
        # Perform analysis
        analyzer.display_basic_info()
        analyzer.analyze_temporal_coverage()
        analyzer.analyze_spatial_coverage()
        analyzer.analyze_data_variables()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Export summary report
        analyzer.export_summary_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Check the following outputs:")
        print("- Visualizations in 'output_plots/' directory")
        print("- Summary report: 'casr_swe_analysis_report.txt'")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        analyzer.close()


if __name__ == "__main__":
    main()
