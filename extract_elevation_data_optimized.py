#!/usr/bin/env python3
"""
Optimized version of elevation data extraction script.

This version handles large time series data more efficiently by providing
options to sample or aggregate temporal data.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedElevationDataExtractor:
    """Extract CaSR data at elevation point locations with optimization for large datasets."""
    
    def __init__(self, elevation_dir, combined_casr_dir, output_dir=None):
        """
        Initialize the extractor.
        
        Parameters:
        -----------
        elevation_dir : str
            Path to directory containing elevation shapefiles
        combined_casr_dir : str
            Path to directory containing combined CaSR NetCDF files
        output_dir : str, optional
            Output directory for extracted data (default: data/output_data/elevation)
        """
        self.elevation_dir = Path(elevation_dir)
        self.combined_casr_dir = Path(combined_casr_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("data/output_data/elevation")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store elevation data
        self.elevation_gdf = None
        
    def load_elevation_data(self):
        """Load elevation shapefile data."""
        logger.info("Loading elevation data...")
        
        # Find shapefile in elevation directory
        shp_files = list(self.elevation_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No shapefile found in {self.elevation_dir}")
        
        # Load the first shapefile found
        shp_file = shp_files[0]
        logger.info(f"Loading shapefile: {shp_file}")
        
        try:
            self.elevation_gdf = gpd.read_file(shp_file)
            logger.info(f"Loaded {len(self.elevation_gdf)} elevation points")
            logger.info(f"Elevation data columns: {list(self.elevation_gdf.columns)}")
            logger.info(f"CRS: {self.elevation_gdf.crs}")
            
            # Display basic statistics for elevation-related columns
            elev_cols = [col for col in self.elevation_gdf.columns if 'elev' in col.lower() or col in ['min', 'max', 'mean', 'median']]
            if elev_cols:
                logger.info(f"Elevation-related columns: {elev_cols}")
                for col in elev_cols:
                    if pd.api.types.is_numeric_dtype(self.elevation_gdf[col]):
                        logger.info(f"{col} range: {self.elevation_gdf[col].min():.1f} - {self.elevation_gdf[col].max():.1f}")
            
        except Exception as e:
            logger.error(f"Error loading shapefile: {e}")
            raise
            
    def get_combined_casr_files(self):
        """Get list of combined CaSR files."""
        nc_files = list(self.combined_casr_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {self.combined_casr_dir}")
        
        # Categorize files
        temporal_files = [f for f in nc_files if "temporal_combined" in f.name]
        full_files = [f for f in nc_files if "combined_full" in f.name]
        
        logger.info(f"Found {len(temporal_files)} temporal combined files")
        logger.info(f"Found {len(full_files)} full combined files")
        
        return temporal_files, full_files
    
    def extract_spatial_windows_to_netcdf(self, nc_file, points_gdf, window_size=3, time_sampling='monthly', max_records=10000):
        """
        Extract spatial windows around elevation points and save as NetCDF.
        
        Parameters:
        -----------
        nc_file : Path
            Path to NetCDF file
        points_gdf : GeoDataFrame
            Points where to extract data
        window_size : int
            Size of spatial window (e.g., 3 for 3x3 window)
        time_sampling : str
            Time sampling strategy ('all', 'monthly', 'yearly', 'sample')
        max_records : int
            Maximum number of records to extract per point
            
        Returns:
        --------
        xarray.Dataset
            Dataset with spatial windows for all elevation points
        """
        logger.info(f"Processing {nc_file.name} with spatial windows (size: {window_size}x{window_size})...")
        
        try:
            # Open NetCDF file
            ds = xr.open_dataset(nc_file)
            logger.info(f"Dataset variables: {list(ds.data_vars)}")
            logger.info(f"Dataset dimensions: {dict(ds.dims)}")
            logger.info(f"Dataset coordinates: {list(ds.coords)}")
            
            # Prioritize actual geographic coordinates over rotated coordinates
            has_2d_coords = False
            lon_coord, lat_coord = None, None
            
            # Check for 2D geographic coordinate arrays first (preferred)
            if 'lon' in ds.coords and 'lat' in ds.coords:
                if ds.lon.ndim == 2 and ds.lat.ndim == 2:
                    lon_coord, lat_coord = 'lon', 'lat'
                    has_2d_coords = True
                    logger.info("Found 2D geographic coordinates (lon, lat) - using actual grid coordinates")
                elif ds.lon.ndim == 1 and ds.lat.ndim == 1:
                    lon_coord, lat_coord = 'lon', 'lat'
                    logger.info("Found 1D geographic coordinates (lon, lat)")
            elif 'longitude' in ds.coords and 'latitude' in ds.coords:
                if ds.longitude.ndim == 2 and ds.latitude.ndim == 2:
                    lon_coord, lat_coord = 'longitude', 'latitude'
                    has_2d_coords = True
                    logger.info("Found 2D geographic coordinates (longitude, latitude) - using actual grid coordinates")
                elif ds.longitude.ndim == 1 and ds.latitude.ndim == 1:
                    lon_coord, lat_coord = 'longitude', 'latitude'
                    logger.info("Found 1D geographic coordinates (longitude, latitude)")
            # Fall back to rotated coordinates only if no geographic coordinates found
            elif 'rlon' in ds.coords and 'rlat' in ds.coords:
                lon_coord, lat_coord = 'rlon', 'rlat'
                logger.info("Using rotated coordinates (rlon, rlat) as fallback")
            else:
                logger.warning(f"Could not identify coordinate variables in {nc_file.name}")
                logger.warning(f"Available coordinates: {list(ds.coords)}")
                return None
            
            # Convert points to same CRS as NetCDF if needed
            points_proj = points_gdf.copy()
            
            # Extract coordinates from points
            if points_proj.crs and points_proj.crs.to_string() != 'EPSG:4326':
                points_proj = points_proj.to_crs('EPSG:4326')
            
            # Get point coordinates (handle both points and polygons)
            point_lons = []
            point_lats = []
            for geom in points_proj.geometry:
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    # Point geometry
                    point_lons.append(geom.x)
                    point_lats.append(geom.y)
                else:
                    # Polygon or other geometry - use centroid
                    centroid = geom.centroid
                    point_lons.append(centroid.x)
                    point_lats.append(centroid.y)
            
            # Handle time sampling for large datasets
            if 'time' in ds.dims:
                time_size = ds.dims['time']
                logger.info(f"Time dimension size: {time_size}")
                
                if time_size > max_records and time_sampling != 'all':
                    logger.info(f"Large time series detected. Applying {time_sampling} sampling...")
                    
                    if time_sampling == 'monthly':
                        # Sample monthly (every ~30 days assuming daily data)
                        step = max(1, time_size // (time_size // 30))
                        ds = ds.isel(time=slice(0, None, step))
                    elif time_sampling == 'yearly':
                        # Sample yearly (every ~365 days assuming daily data)
                        step = max(1, time_size // (time_size // 365))
                        ds = ds.isel(time=slice(0, None, step))
                    elif time_sampling == 'sample':
                        # Random sampling
                        indices = np.sort(np.random.choice(time_size, min(max_records, time_size), replace=False))
                        ds = ds.isel(time=indices)
                    
                    logger.info(f"Reduced time dimension to: {ds.dims['time']}")
            
            # Initialize output dataset
            n_points = len(points_gdf)
            output_ds = self.initialize_output_dataset(ds, n_points, window_size, lon_coord, lat_coord)
            
            # Extract spatial windows for each elevation point
            for i, (idx, point_row) in enumerate(points_gdf.iterrows()):
                try:
                    # Get point coordinates
                    geom = point_row.geometry
                    if hasattr(geom, 'x') and hasattr(geom, 'y'):
                        point_lon, point_lat = geom.x, geom.y
                    else:
                        centroid = geom.centroid
                        point_lon, point_lat = centroid.x, centroid.y
                    
                    # Extract spatial window
                    window_data, center_idx, bounds = self.extract_spatial_window(
                        ds, point_lon, point_lat, window_size, lon_coord, lat_coord, has_2d_coords
                    )
                    
                    if window_data is not None:
                        # Store window in output dataset
                        self.store_window_in_dataset(output_ds, i, window_data, point_row, center_idx, bounds)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract window for point {i}: {e}")
                    continue
            
            ds.close()
            
            # Add metadata
            output_ds.attrs['title'] = f'Spatial windows extracted from {nc_file.name}'
            output_ds.attrs['source_file'] = str(nc_file)
            output_ds.attrs['window_size'] = window_size
            output_ds.attrs['extraction_date'] = datetime.now().isoformat()
            output_ds.attrs['n_elevation_points'] = n_points
            
            logger.info(f"Extracted spatial windows for {n_points} elevation points")
            return output_ds
                
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}")
            return None
    
    def initialize_output_dataset(self, source_ds, n_points, window_size, lon_coord, lat_coord):
        """Initialize the output NetCDF dataset structure"""
        
        # Get time coordinate
        time_coord = source_ds.time if 'time' in source_ds.coords else None
        
        # Create coordinate arrays
        coords = {
            'elevation_points': np.arange(n_points),
            'window_lat': np.arange(window_size),
            'window_lon': np.arange(window_size)
        }
        
        if time_coord is not None:
            coords['time'] = time_coord
        
        # Initialize data variables
        data_vars = {}
        
        # Coordinate arrays for each window
        data_vars['lon_windows'] = (
            ['elevation_points', 'window_lat', 'window_lon'], 
            np.full((n_points, window_size, window_size), np.nan)
        )
        data_vars['lat_windows'] = (
            ['elevation_points', 'window_lat', 'window_lon'], 
            np.full((n_points, window_size, window_size), np.nan)
        )
        
        # Original point coordinates
        data_vars['original_lon'] = (['elevation_points'], np.full(n_points, np.nan))
        data_vars['original_lat'] = (['elevation_points'], np.full(n_points, np.nan))
        data_vars['window_center_lon'] = (['elevation_points'], np.full(n_points, np.nan))
        data_vars['window_center_lat'] = (['elevation_points'], np.full(n_points, np.nan))
        
        # Elevation metadata
        data_vars['elevation_mean'] = (['elevation_points'], np.full(n_points, np.nan))
        data_vars['elevation_min'] = (['elevation_points'], np.full(n_points, np.nan))
        data_vars['elevation_max'] = (['elevation_points'], np.full(n_points, np.nan))
        data_vars['elevation_median'] = (['elevation_points'], np.full(n_points, np.nan))
        
        # Data variables from source
        for var_name in source_ds.data_vars:
            if var_name == 'rotated_pole':
                continue
                
            if time_coord is not None and 'time' in source_ds[var_name].dims:
                # Time-varying data
                data_vars[var_name] = (
                    ['time', 'elevation_points', 'window_lat', 'window_lon'],
                    np.full((len(time_coord), n_points, window_size, window_size), np.nan)
                )
            else:
                # Static data
                data_vars[var_name] = (
                    ['elevation_points', 'window_lat', 'window_lon'],
                    np.full((n_points, window_size, window_size), np.nan)
                )
        
        return xr.Dataset(data_vars, coords=coords)
    
    def extract_spatial_window(self, ds, target_lon, target_lat, window_size, lon_coord, lat_coord, has_2d_coords):
        """Extract spatial window around target coordinates"""
        
        try:
            # Get 2D coordinate arrays
            if has_2d_coords and lon_coord in ['lon', 'longitude'] and lat_coord in ['lat', 'latitude']:
                lon_2d = ds[lon_coord].values
                lat_2d = ds[lat_coord].values
                
                # Convert longitude ranges for consistency
                target_lon_adj = target_lon if target_lon >= -180 else target_lon + 360
                lon_2d_adj = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
                
                # Find nearest grid point
                dist = np.sqrt((lon_2d_adj - target_lon_adj)**2 + (lat_2d - target_lat)**2)
                center_idx = np.unravel_index(np.argmin(dist), dist.shape)
                
            elif lon_coord == 'rlon' and lat_coord == 'rlat':
                # Handle rotated coordinates
                rlon_vals = ds[lon_coord].values
                rlat_vals = ds[lat_coord].values
                
                rlon_idx = np.argmin(np.abs(rlon_vals - target_lon))
                rlat_idx = np.argmin(np.abs(rlat_vals - target_lat))
                center_idx = (rlat_idx, rlon_idx)
                
            else:
                # Handle 1D coordinates
                lon_vals = ds[lon_coord].values
                lat_vals = ds[lat_coord].values
                
                lon_idx = np.argmin(np.abs(lon_vals - target_lon))
                lat_idx = np.argmin(np.abs(lat_vals - target_lat))
                center_idx = (lat_idx, lon_idx)
            
            # Calculate window bounds
            half_window = window_size // 2
            
            # Get grid dimensions
            if has_2d_coords:
                grid_shape = ds[lon_coord].shape
            else:
                # For 1D coordinates, get shape from data variables
                for var in ds.data_vars:
                    if var != 'rotated_pole' and len(ds[var].dims) >= 2:
                        # Find spatial dimensions
                        spatial_dims = [dim for dim in ds[var].dims if dim in ['rlat', 'rlon', 'lat', 'lon', 'y', 'x']]
                        if len(spatial_dims) >= 2:
                            grid_shape = (ds.dims[spatial_dims[0]], ds.dims[spatial_dims[1]])
                            break
                else:
                    grid_shape = (100, 100)  # fallback
            
            row_start = max(0, center_idx[0] - half_window)
            row_end = min(grid_shape[0], center_idx[0] + half_window + 1)
            col_start = max(0, center_idx[1] - half_window)
            col_end = min(grid_shape[1], center_idx[1] + half_window + 1)
            
            # Extract window using appropriate dimension names
            if 'rlat' in ds.dims and 'rlon' in ds.dims:
                window_data = ds.isel(
                    rlat=slice(row_start, row_end),
                    rlon=slice(col_start, col_end)
                )
            elif 'lat' in ds.dims and 'lon' in ds.dims:
                window_data = ds.isel(
                    lat=slice(row_start, row_end),
                    lon=slice(col_start, col_end)
                )
            elif 'y' in ds.dims and 'x' in ds.dims:
                window_data = ds.isel(
                    y=slice(row_start, row_end),
                    x=slice(col_start, col_end)
                )
            else:
                logger.warning("Could not determine spatial dimension names")
                return None, None, None
            
            bounds = (row_start, row_end, col_start, col_end)
            return window_data, center_idx, bounds
            
        except Exception as e:
            logger.error(f"Error extracting spatial window: {e}")
            return None, None, None
    
    def store_window_in_dataset(self, output_ds, point_idx, window_data, point_row, center_idx, bounds):
        """Store extracted window data in the output dataset"""
        
        try:
            # Get window coordinates
            lon_coord = None
            lat_coord = None
            
            # Find coordinate variables in window data
            if 'lon' in window_data.coords:
                lon_coord, lat_coord = 'lon', 'lat'
            elif 'longitude' in window_data.coords:
                lon_coord, lat_coord = 'longitude', 'latitude'
            elif 'rlon' in window_data.coords:
                lon_coord, lat_coord = 'rlon', 'rlat'
            
            if lon_coord and lat_coord:
                # Store coordinate arrays (pad/crop to match window size)
                window_size = output_ds.dims['window_lat']
                
                lon_vals = window_data[lon_coord].values
                lat_vals = window_data[lat_coord].values
                
                # Ensure arrays are 2D
                if lon_vals.ndim == 1:
                    lon_vals, lat_vals = np.meshgrid(lon_vals, lat_vals)
                
                # Pad or crop to match expected window size
                actual_rows, actual_cols = lon_vals.shape
                
                # Create arrays of the expected size, filled with NaN
                lon_window = np.full((window_size, window_size), np.nan)
                lat_window = np.full((window_size, window_size), np.nan)
                
                # Copy actual data to the center of the window
                row_offset = (window_size - actual_rows) // 2
                col_offset = (window_size - actual_cols) // 2
                
                end_row = row_offset + actual_rows
                end_col = col_offset + actual_cols
                
                lon_window[row_offset:end_row, col_offset:end_col] = lon_vals
                lat_window[row_offset:end_row, col_offset:end_col] = lat_vals
                
                # Store in output dataset
                output_ds['lon_windows'][point_idx, :, :] = lon_window
                output_ds['lat_windows'][point_idx, :, :] = lat_window
            
            # Store original point coordinates
            geom = point_row.geometry
            if hasattr(geom, 'x') and hasattr(geom, 'y'):
                orig_lon, orig_lat = geom.x, geom.y
            else:
                centroid = geom.centroid
                orig_lon, orig_lat = centroid.x, centroid.y
            
            output_ds['original_lon'][point_idx] = orig_lon
            output_ds['original_lat'][point_idx] = orig_lat
            
            # Store window center coordinates
            if lon_coord and lat_coord:
                center_lon = float(window_data[lon_coord].values.flat[0]) if window_data[lon_coord].size > 0 else np.nan
                center_lat = float(window_data[lat_coord].values.flat[0]) if window_data[lat_coord].size > 0 else np.nan
                output_ds['window_center_lon'][point_idx] = center_lon
                output_ds['window_center_lat'][point_idx] = center_lat
            
            # Store elevation metadata
            elev_cols = [col for col in point_row.index if 'elev' in col.lower() or col in ['min', 'max', 'mean', 'median']]
            for col in elev_cols:
                if pd.api.types.is_numeric_dtype(point_row[col]):
                    if col.lower() in ['mean', 'min', 'max', 'median']:
                        output_ds[f'elevation_{col.lower()}'][point_idx] = point_row[col]
                    elif 'mean' in col.lower():
                        output_ds['elevation_mean'][point_idx] = point_row[col]
                    elif 'min' in col.lower():
                        output_ds['elevation_min'][point_idx] = point_row[col]
                    elif 'max' in col.lower():
                        output_ds['elevation_max'][point_idx] = point_row[col]
                    elif 'median' in col.lower():
                        output_ds['elevation_median'][point_idx] = point_row[col]
            
            # Store data variables
            for var_name in window_data.data_vars:
                if var_name == 'rotated_pole':
                    continue
                
                var_data = window_data[var_name]
                
                if 'time' in var_data.dims:
                    # Time-varying data
                    time_vals = var_data.values
                    
                    # Pad/crop spatial dimensions to match window size
                    if time_vals.ndim == 3:  # time, lat, lon
                        padded_data = np.full((time_vals.shape[0], window_size, window_size), np.nan)
                        actual_rows, actual_cols = time_vals.shape[1], time_vals.shape[2]
                        
                        row_offset = (window_size - actual_rows) // 2
                        col_offset = (window_size - actual_cols) // 2
                        
                        end_row = row_offset + actual_rows
                        end_col = col_offset + actual_cols
                        
                        padded_data[:, row_offset:end_row, col_offset:end_col] = time_vals
                        output_ds[var_name][:, point_idx, :, :] = padded_data
                else:
                    # Static data
                    static_vals = var_data.values
                    
                    if static_vals.ndim == 2:  # lat, lon
                        padded_data = np.full((window_size, window_size), np.nan)
                        actual_rows, actual_cols = static_vals.shape
                        
                        row_offset = (window_size - actual_rows) // 2
                        col_offset = (window_size - actual_cols) // 2
                        
                        end_row = row_offset + actual_rows
                        end_col = col_offset + actual_cols
                        
                        padded_data[row_offset:end_row, col_offset:end_col] = static_vals
                        output_ds[var_name][point_idx, :, :] = padded_data
                    
        except Exception as e:
            logger.warning(f"Error storing window data for point {point_idx}: {e}")
    
    def process_all_files(self, file_types=['temporal', 'full'], time_sampling='monthly', max_records=10000, window_size=3, output_format='netcdf'):
        """
        Process all CaSR files and extract elevation data with spatial windows.
        
        Parameters:
        -----------
        file_types : list
            Types of files to process ('temporal', 'full', or both)
        time_sampling : str
            Time sampling strategy ('all', 'monthly', 'yearly', 'sample')
        max_records : int
            Maximum number of records to extract per point
        window_size : int
            Size of spatial window (e.g., 3 for 3x3 window)
        output_format : str
            Output format ('netcdf' for spatial windows, 'csv' for legacy point extraction)
        """
        if self.elevation_gdf is None:
            self.load_elevation_data()
        
        temporal_files, full_files = self.get_combined_casr_files()
        
        all_results = {}
        
        # Process temporal files
        if 'temporal' in file_types:
            logger.info("Processing temporal combined files...")
            for nc_file in temporal_files:
                if output_format == 'netcdf':
                    result_ds = self.extract_spatial_windows_to_netcdf(
                        nc_file, self.elevation_gdf, window_size, time_sampling, max_records
                    )
                    if result_ds is not None:
                        all_results[f"temporal_{nc_file.stem}"] = result_ds
                else:
                    # Legacy CSV extraction (keeping for backward compatibility)
                    logger.warning("CSV extraction method is deprecated. Using NetCDF spatial windows instead.")
                    result_ds = self.extract_spatial_windows_to_netcdf(
                        nc_file, self.elevation_gdf, window_size, time_sampling, max_records
                    )
                    if result_ds is not None:
                        all_results[f"temporal_{nc_file.stem}"] = result_ds
        
        # Process full files
        if 'full' in file_types:
            logger.info("Processing full combined files...")
            for nc_file in full_files:
                if output_format == 'netcdf':
                    result_ds = self.extract_spatial_windows_to_netcdf(
                        nc_file, self.elevation_gdf, window_size, time_sampling, max_records
                    )
                    if result_ds is not None:
                        all_results[f"full_{nc_file.stem}"] = result_ds
                else:
                    # Legacy CSV extraction (keeping for backward compatibility)
                    logger.warning("CSV extraction method is deprecated. Using NetCDF spatial windows instead.")
                    result_ds = self.extract_spatial_windows_to_netcdf(
                        nc_file, self.elevation_gdf, window_size, time_sampling, max_records
                    )
                    if result_ds is not None:
                        all_results[f"full_{nc_file.stem}"] = result_ds
        
        return all_results
    
    def save_results(self, results, format='netcdf'):
        """
        Save extraction results to files.
        
        Parameters:
        -----------
        results : dict
            Dictionary of xarray Datasets with results
        format : str
            Output format ('netcdf', 'csv', 'parquet', 'both')
        """
        logger.info(f"Saving results to {self.output_dir}")
        
        for name, ds in results.items():
            if ds is None:
                continue
                
            base_filename = f"elevation_windows_{name}"
            
            # Save as NetCDF (primary format for spatial windows)
            if format in ['netcdf', 'both']:
                nc_file = self.output_dir / f"{base_filename}.nc"
                ds.to_netcdf(nc_file, engine='netcdf4')
                logger.info(f"Saved NetCDF: {nc_file}")
            
            # Legacy CSV/Parquet support (flattened data)
            if format in ['csv', 'parquet', 'both']:
                logger.info(f"Converting spatial windows to flattened format for {format} output...")
                try:
                    # Flatten the spatial windows to tabular format
                    flattened_df = self.flatten_spatial_windows_to_dataframe(ds)
                    
                    if format in ['csv', 'both']:
                        csv_file = self.output_dir / f"{base_filename}_flattened.csv"
                        flattened_df.to_csv(csv_file, index=False)
                        logger.info(f"Saved flattened CSV: {csv_file}")
                    
                    if format in ['parquet', 'both']:
                        parquet_file = self.output_dir / f"{base_filename}_flattened.parquet"
                        flattened_df.to_parquet(parquet_file, index=False)
                        logger.info(f"Saved flattened Parquet: {parquet_file}")
                        
                except Exception as e:
                    logger.warning(f"Could not create flattened format for {name}: {e}")
    
    def flatten_spatial_windows_to_dataframe(self, ds):
        """Convert spatial windows dataset to flattened DataFrame for CSV/Parquet output"""
        
        records = []
        n_points = ds.dims['elevation_points']
        window_size = ds.dims['window_lat']
        
        for point_idx in range(n_points):
            # Get point metadata
            orig_lon = float(ds['original_lon'][point_idx].values)
            orig_lat = float(ds['original_lat'][point_idx].values)
            center_lon = float(ds['window_center_lon'][point_idx].values) if not np.isnan(ds['window_center_lon'][point_idx].values) else None
            center_lat = float(ds['window_center_lat'][point_idx].values) if not np.isnan(ds['window_center_lat'][point_idx].values) else None
            
            # Get elevation data
            elev_mean = float(ds['elevation_mean'][point_idx].values) if not np.isnan(ds['elevation_mean'][point_idx].values) else None
            elev_min = float(ds['elevation_min'][point_idx].values) if not np.isnan(ds['elevation_min'][point_idx].values) else None
            elev_max = float(ds['elevation_max'][point_idx].values) if not np.isnan(ds['elevation_max'][point_idx].values) else None
            elev_median = float(ds['elevation_median'][point_idx].values) if not np.isnan(ds['elevation_median'][point_idx].values) else None
            
            # Get coordinate windows
            lon_window = ds['lon_windows'][point_idx, :, :].values
            lat_window = ds['lat_windows'][point_idx, :, :].values
            
            # Process each grid cell in the window
            for i in range(window_size):
                for j in range(window_size):
                    grid_lon = float(lon_window[i, j]) if not np.isnan(lon_window[i, j]) else None
                    grid_lat = float(lat_window[i, j]) if not np.isnan(lat_window[i, j]) else None
                    
                    if grid_lon is None or grid_lat is None:
                        continue  # Skip NaN grid cells
                    
                    # Calculate relative position from center
                    rel_row = i - (window_size // 2)
                    rel_col = j - (window_size // 2)
                    distance_from_center = np.sqrt(rel_row**2 + rel_col**2)
                    
                    # Base record for this grid cell
                    base_record = {
                        'point_id': point_idx,
                        'original_lon': orig_lon,
                        'original_lat': orig_lat,
                        'window_center_lon': center_lon,
                        'window_center_lat': center_lat,
                        'grid_lon': grid_lon,
                        'grid_lat': grid_lat,
                        'window_row': i,
                        'window_col': j,
                        'relative_row': rel_row,
                        'relative_col': rel_col,
                        'distance_from_center': distance_from_center,
                        'elevation_mean': elev_mean,
                        'elevation_min': elev_min,
                        'elevation_max': elev_max,
                        'elevation_median': elev_median
                    }
                    
                    # Add data variables
                    for var_name in ds.data_vars:
                        if var_name in ['lon_windows', 'lat_windows', 'original_lon', 'original_lat', 
                                       'window_center_lon', 'window_center_lat', 'elevation_mean', 
                                       'elevation_min', 'elevation_max', 'elevation_median']:
                            continue  # Skip coordinate and metadata variables
                        
                        var_data = ds[var_name]
                        
                        if 'time' in var_data.dims:
                            # Time-varying data - create separate records for each time step
                            for t, time_val in enumerate(ds.time.values):
                                record = base_record.copy()
                                record['time'] = pd.to_datetime(time_val)
                                
                                data_val = var_data[t, point_idx, i, j].values
                                record[var_name] = float(data_val) if not np.isnan(data_val) else None
                                
                                records.append(record)
                        else:
                            # Static data
                            data_val = var_data[point_idx, i, j].values
                            base_record[var_name] = float(data_val) if not np.isnan(data_val) else None
                    
                    # If no time dimension, add the base record
                    if 'time' not in ds.dims:
                        records.append(base_record)
        
        return pd.DataFrame(records)
    
    def generate_summary_report_netcdf(self, results):
        """Generate a summary report for NetCDF spatial window extraction results."""
        logger.info("Generating spatial window extraction summary report...")
        
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'extraction_type': 'spatial_windows',
            'elevation_points': len(self.elevation_gdf) if self.elevation_gdf is not None else 0,
            'files_processed': len(results),
        }
        
        # File-specific summaries for NetCDF datasets
        file_summaries = {}
        total_data_points = 0
        
        for name, ds in results.items():
            if ds is not None:
                # Get dataset dimensions
                n_points = ds.dims.get('elevation_points', 0)
                window_size = ds.dims.get('window_lat', 0)
                n_times = ds.dims.get('time', 1)
                
                # Calculate total data points (points × window cells × time steps)
                data_points = n_points * (window_size ** 2) * n_times
                total_data_points += data_points
                
                # Get data variables (excluding coordinate and metadata variables)
                data_vars = [var for var in ds.data_vars 
                           if var not in ['lon_windows', 'lat_windows', 'original_lon', 'original_lat',
                                        'window_center_lon', 'window_center_lat', 'elevation_mean',
                                        'elevation_min', 'elevation_max', 'elevation_median']]
                
                file_summaries[name] = {
                    'elevation_points': n_points,
                    'window_size': f"{window_size}x{window_size}",
                    'time_steps': n_times,
                    'total_data_points': data_points,
                    'data_variables': data_vars,
                    'time_range': None,
                    'window_size_numeric': window_size
                }
                
                # Get time range if available
                if 'time' in ds.coords:
                    time_vals = ds.time.values
                    file_summaries[name]['time_range'] = {
                        'start': pd.to_datetime(time_vals[0]).isoformat(),
                        'end': pd.to_datetime(time_vals[-1]).isoformat()
                    }
                
                # Get spatial extent
                if n_points > 0:
                    orig_lons = ds['original_lon'].values
                    orig_lats = ds['original_lat'].values
                    
                    # Remove NaN values for extent calculation
                    valid_lons = orig_lons[~np.isnan(orig_lons)]
                    valid_lats = orig_lats[~np.isnan(orig_lats)]
                    
                    if len(valid_lons) > 0:
                        file_summaries[name]['spatial_extent'] = {
                            'lon_range': [float(valid_lons.min()), float(valid_lons.max())],
                            'lat_range': [float(valid_lats.min()), float(valid_lats.max())]
                        }
        
        summary['total_data_points'] = total_data_points
        summary['file_details'] = file_summaries
        
        # Save summary
        summary_file = self.output_dir / "spatial_window_extraction_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("SPATIAL WINDOW EXTRACTION SUMMARY")
        print("="*70)
        print(f"Extraction date: {summary['extraction_date']}")
        print(f"Elevation points processed: {summary['elevation_points']}")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Total data points extracted: {summary['total_data_points']:,}")
        
        print("\nFile details:")
        for name, details in file_summaries.items():
            print(f"  {name}:")
            print(f"    Elevation points: {details['elevation_points']}")
            print(f"    Window size: {details['window_size']}")
            print(f"    Time steps: {details['time_steps']}")
            print(f"    Data points: {details['total_data_points']:,}")
            print(f"    Variables: {', '.join(details['data_variables'])}")
            
            if details['time_range']:
                print(f"    Time range: {details['time_range']['start']} to {details['time_range']['end']}")
            
            if 'spatial_extent' in details:
                extent = details['spatial_extent']
                print(f"    Spatial extent: Lon {extent['lon_range'][0]:.2f} to {extent['lon_range'][1]:.2f}, "
                      f"Lat {extent['lat_range'][0]:.2f} to {extent['lat_range'][1]:.2f}")
        
        print("="*70)
        print(f"\nOutput files saved to: {self.output_dir}")
        print("NetCDF files preserve full spatial window structure.")
        print("Use flattened CSV/Parquet files for tabular analysis.")
        print("="*70)
    
    def generate_summary_report(self, results):
        """Generate a summary report of the extraction."""
        logger.info("Generating summary report...")
        
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'elevation_points': len(self.elevation_gdf) if self.elevation_gdf is not None else 0,
            'files_processed': len(results),
            'total_records': sum(len(df) for df in results.values() if df is not None),
        }
        
        # File-specific summaries
        file_summaries = {}
        for name, df in results.items():
            if df is not None:
                file_summaries[name] = {
                    'records': len(df),
                    'variables': [col for col in df.columns if col not in 
                                ['point_id', 'original_lon', 'original_lat', 'grid_lon', 'grid_lat', 'time']],
                    'time_range': None
                }
                
                if 'time' in df.columns:
                    file_summaries[name]['time_range'] = {
                        'start': df['time'].min().isoformat() if pd.notna(df['time'].min()) else None,
                        'end': df['time'].max().isoformat() if pd.notna(df['time'].max()) else None
                    }
        
        summary['file_details'] = file_summaries
        
        # Save summary
        summary_file = self.output_dir / "extraction_summary_optimized.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZED EXTRACTION SUMMARY")
        print("="*60)
        print(f"Elevation points processed: {summary['elevation_points']}")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Total records extracted: {summary['total_records']}")
        print("\nFile details:")
        for name, details in file_summaries.items():
            print(f"  {name}:")
            print(f"    Records: {details['records']}")
            print(f"    Variables: {', '.join(details['variables'])}")
            if details['time_range']:
                print(f"    Time range: {details['time_range']['start']} to {details['time_range']['end']}")
        print("="*60)


def main():
    """Main function to run the extraction."""
    parser = argparse.ArgumentParser(description='Extract elevation data from combined CaSR files with spatial windows')
    parser.add_argument('--elevation-dir', default='data/input_data/Elevation',
                       help='Directory containing elevation shapefiles')
    parser.add_argument('--casr-dir', default='data/output_data/combined_casr',
                       help='Directory containing combined CaSR NetCDF files')
    parser.add_argument('--output-dir', default='data/output_data/elevation',
                       help='Output directory for extracted data')
    parser.add_argument('--file-types', nargs='+', choices=['temporal', 'full'], 
                       default=['temporal', 'full'],
                       help='Types of files to process')
    parser.add_argument('--format', choices=['netcdf', 'csv', 'parquet', 'both'], default='netcdf',
                       help='Output format (netcdf preserves spatial structure, others are flattened)')
    parser.add_argument('--window-size', type=int, default=3,
                       help='Size of spatial window to extract around each elevation point (default: 3 for 3x3)')
    parser.add_argument('--time-sampling', choices=['all', 'monthly', 'yearly', 'sample'], 
                       default='monthly',
                       help='Time sampling strategy for large datasets')
    parser.add_argument('--max-records', type=int, default=10000,
                       help='Maximum number of records to extract per point')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize extractor
        extractor = OptimizedElevationDataExtractor(
            elevation_dir=args.elevation_dir,
            combined_casr_dir=args.casr_dir,
            output_dir=args.output_dir
        )
        
        # Process files with spatial windows
        results = extractor.process_all_files(
            file_types=args.file_types,
            time_sampling=args.time_sampling,
            max_records=args.max_records,
            window_size=args.window_size,
            output_format='netcdf'  # Always use NetCDF for processing
        )
        
        if not results:
            logger.error("No data was extracted!")
            return 1
        
        # Save results in requested format
        extractor.save_results(results, format=args.format)
        
        # Generate summary (update for NetCDF datasets)
        extractor.generate_summary_report_netcdf(results)
        
        logger.info("Spatial window extraction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
