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
    
    def extract_data_from_netcdf(self, nc_file, points_gdf, time_sampling='monthly', max_records=10000):
        """
        Extract data from NetCDF file at point locations with optimization.
        
        Parameters:
        -----------
        nc_file : Path
            Path to NetCDF file
        points_gdf : GeoDataFrame
            Points where to extract data
        time_sampling : str
            Time sampling strategy ('all', 'monthly', 'yearly', 'sample')
        max_records : int
            Maximum number of records to extract per point
            
        Returns:
        --------
        pandas.DataFrame
            Extracted data with coordinates and values
        """
        logger.info(f"Processing {nc_file.name}...")
        
        try:
            # Open NetCDF file
            ds = xr.open_dataset(nc_file)
            logger.info(f"Dataset variables: {list(ds.data_vars)}")
            logger.info(f"Dataset dimensions: {dict(ds.dims)}")
            
            # Get coordinate information
            if 'lon' in ds.coords and 'lat' in ds.coords:
                lon_coord, lat_coord = 'lon', 'lat'
            elif 'longitude' in ds.coords and 'latitude' in ds.coords:
                lon_coord, lat_coord = 'longitude', 'latitude'
            elif 'rlon' in ds.coords and 'rlat' in ds.coords:
                lon_coord, lat_coord = 'rlon', 'rlat'
            else:
                logger.warning(f"Could not identify coordinate variables in {nc_file.name}")
                return None
            
            # Convert points to same CRS as NetCDF if needed
            points_proj = points_gdf.copy()
            
            # If the NetCDF uses rotated coordinates, we might need to handle projection
            if lon_coord == 'rlon' and lat_coord == 'rlat':
                logger.info("Detected rotated coordinates - using nearest neighbor approach")
            
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
            
            # Extract data at point locations
            extracted_data = []
            
            for i, (lon, lat) in enumerate(zip(point_lons, point_lats)):
                try:
                    # Find nearest grid point
                    if lon_coord == 'rlon' and lat_coord == 'rlat':
                        # For rotated coordinates, find nearest grid point
                        rlon_vals = ds[lon_coord].values
                        rlat_vals = ds[lat_coord].values
                        
                        # Find nearest indices
                        rlon_idx = np.argmin(np.abs(rlon_vals - lon))
                        rlat_idx = np.argmin(np.abs(rlat_vals - lat))
                        
                        # Select using indices
                        point_data = ds.isel({lon_coord: rlon_idx, lat_coord: rlat_idx})
                    elif 'lon' in ds.coords and 'lat' in ds.coords and ds.lon.ndim == 2:
                        # Handle 2D coordinate arrays (common in rotated grids)
                        # Convert longitude from 0-360 to -180-180 if needed
                        target_lon = lon if lon < 0 else lon - 360
                        
                        # Find nearest grid point using 2D coordinates
                        lon_2d = ds.lon.values
                        lat_2d = ds.lat.values
                        
                        # Convert lon_2d to -180-180 range if needed
                        lon_2d_adj = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
                        
                        # Calculate distance to all grid points
                        dist = np.sqrt((lon_2d_adj - target_lon)**2 + (lat_2d - lat)**2)
                        
                        # Find indices of minimum distance
                        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                        rlat_idx, rlon_idx = min_idx
                        
                        # Select using indices
                        point_data = ds.isel(rlat=rlat_idx, rlon=rlon_idx)
                    else:
                        point_data = ds.sel({lon_coord: lon, lat_coord: lat}, method='nearest')
                    
                    # Extract all variables
                    point_dict = {
                        'point_id': i,
                        'original_lon': lon,
                        'original_lat': lat,
                        'grid_lon': float(point_data[lon_coord].values),
                        'grid_lat': float(point_data[lat_coord].values),
                    }
                    
                    # Add elevation-related data if available
                    elev_cols = [col for col in points_proj.columns if 'elev' in col.lower() or col in ['min', 'max', 'mean', 'median']]
                    for col in elev_cols:
                        if pd.api.types.is_numeric_dtype(points_proj[col]):
                            point_dict[f'elevation_{col}'] = points_proj.iloc[i][col]
                    
                    # Add all data variables
                    for var in ds.data_vars:
                        if var == 'rotated_pole':  # Skip coordinate system info
                            continue
                            
                        var_data = point_data[var]
                        if var_data.dims:  # Has dimensions (time series)
                            # If it's a time series, we'll handle it separately
                            if 'time' in var_data.dims:
                                # Store time series data
                                times = var_data.time.values
                                values = var_data.values
                                
                                # Create time series entries
                                for t, v in zip(times, values):
                                    ts_dict = point_dict.copy()
                                    ts_dict['time'] = pd.to_datetime(t)
                                    ts_dict[var] = float(v) if not np.isnan(v) else None
                                    extracted_data.append(ts_dict)
                            else:
                                # Non-time dimensions
                                point_dict[var] = float(var_data.values)
                        else:
                            # Scalar value
                            point_dict[var] = float(var_data.values)
                    
                    # If no time dimension, add the point data
                    if not any('time' in ds[var].dims for var in ds.data_vars if var != 'rotated_pole'):
                        extracted_data.append(point_dict)
                        
                except Exception as e:
                    logger.warning(f"Could not extract data for point {i} ({lon}, {lat}): {e}")
                    continue
            
            ds.close()
            
            if extracted_data:
                df = pd.DataFrame(extracted_data)
                logger.info(f"Extracted data for {len(df)} records from {len(point_lons)} points")
                return df
            else:
                logger.warning(f"No data extracted from {nc_file.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}")
            return None
    
    def process_all_files(self, file_types=['temporal', 'full'], time_sampling='monthly', max_records=10000):
        """
        Process all CaSR files and extract elevation data.
        
        Parameters:
        -----------
        file_types : list
            Types of files to process ('temporal', 'full', or both)
        time_sampling : str
            Time sampling strategy ('all', 'monthly', 'yearly', 'sample')
        max_records : int
            Maximum number of records to extract per point
        """
        if self.elevation_gdf is None:
            self.load_elevation_data()
        
        temporal_files, full_files = self.get_combined_casr_files()
        
        all_results = {}
        
        # Process temporal files
        if 'temporal' in file_types:
            logger.info("Processing temporal combined files...")
            for nc_file in temporal_files:
                result_df = self.extract_data_from_netcdf(nc_file, self.elevation_gdf, time_sampling, max_records)
                if result_df is not None:
                    all_results[f"temporal_{nc_file.stem}"] = result_df
        
        # Process full files
        if 'full' in file_types:
            logger.info("Processing full combined files...")
            for nc_file in full_files:
                result_df = self.extract_data_from_netcdf(nc_file, self.elevation_gdf, time_sampling, max_records)
                if result_df is not None:
                    all_results[f"full_{nc_file.stem}"] = result_df
        
        return all_results
    
    def save_results(self, results, format='csv'):
        """
        Save extraction results to files.
        
        Parameters:
        -----------
        results : dict
            Dictionary of DataFrames with results
        format : str
            Output format ('csv', 'parquet', 'both')
        """
        logger.info(f"Saving results to {self.output_dir}")
        
        for name, df in results.items():
            if df is None or df.empty:
                continue
                
            base_filename = f"elevation_extracted_{name}"
            
            if format in ['csv', 'both']:
                csv_file = self.output_dir / f"{base_filename}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved CSV: {csv_file}")
            
            if format in ['parquet', 'both']:
                parquet_file = self.output_dir / f"{base_filename}.parquet"
                df.to_parquet(parquet_file, index=False)
                logger.info(f"Saved Parquet: {parquet_file}")
    
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
    parser = argparse.ArgumentParser(description='Extract elevation data from combined CaSR files (optimized)')
    parser.add_argument('--elevation-dir', default='data/input_data/Elevation',
                       help='Directory containing elevation shapefiles')
    parser.add_argument('--casr-dir', default='data/output_data/combined_casr',
                       help='Directory containing combined CaSR NetCDF files')
    parser.add_argument('--output-dir', default='data/output_data/elevation',
                       help='Output directory for extracted data')
    parser.add_argument('--file-types', nargs='+', choices=['temporal', 'full'], 
                       default=['temporal', 'full'],
                       help='Types of files to process')
    parser.add_argument('--format', choices=['csv', 'parquet', 'both'], default='csv',
                       help='Output format')
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
        
        # Process files
        results = extractor.process_all_files(
            file_types=args.file_types,
            time_sampling=args.time_sampling,
            max_records=args.max_records
        )
        
        if not results:
            logger.error("No data was extracted!")
            return 1
        
        # Save results
        extractor.save_results(results, format=args.format)
        
        # Generate summary
        extractor.generate_summary_report(results)
        
        logger.info("Extraction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
