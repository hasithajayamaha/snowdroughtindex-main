#!/usr/bin/env python3
"""
Filter and merge elevation data for non-null precipitation and snow water equivalent values.

This script:
1. Loads elevation data from shapefiles
2. Extracts CaSR data (precipitation and SWE) at elevation points
3. Filters for non-null values in both variables
4. Merges the data and provides analysis
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


class ElevationDataFilter:
    """Filter and merge elevation data based on non-null precipitation and SWE values."""
    
    def __init__(self, elevation_dir, casr_dir, output_dir=None):
        """
        Initialize the filter.
        
        Parameters:
        -----------
        elevation_dir : str
            Path to directory containing elevation shapefiles
        casr_dir : str
            Path to directory containing CaSR NetCDF files
        output_dir : str, optional
            Output directory for filtered data
        """
        self.elevation_dir = Path(elevation_dir)
        self.casr_dir = Path(casr_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("data/output_data/filtered_elevation")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store data
        self.elevation_gdf = None
        self.precipitation_data = None
        self.swe_data = None
        
    def load_elevation_data(self, sample_size=None):
        """
        Load elevation shapefile data.
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of points to sample for testing
        """
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
            
            # Sample data if requested
            if sample_size and len(self.elevation_gdf) > sample_size:
                logger.info(f"Sampling {sample_size} points from {len(self.elevation_gdf)} total points")
                self.elevation_gdf = self.elevation_gdf.sample(n=sample_size, random_state=42)
            
            logger.info(f"Loaded {len(self.elevation_gdf)} elevation points")
            logger.info(f"Elevation data columns: {list(self.elevation_gdf.columns)}")
            
            # Identify elevation columns
            self.elev_cols = [col for col in self.elevation_gdf.columns 
                             if 'elev' in col.lower() or col in ['min', 'max', 'mean', 'median']]
            
            if self.elev_cols:
                logger.info(f"Elevation columns found: {self.elev_cols}")
                for col in self.elev_cols:
                    if pd.api.types.is_numeric_dtype(self.elevation_gdf[col]):
                        logger.info(f"{col} range: {self.elevation_gdf[col].min():.1f} - {self.elevation_gdf[col].max():.1f}")
            
        except Exception as e:
            logger.error(f"Error loading shapefile: {e}")
            raise
    
    def find_casr_files(self):
        """Find precipitation and SWE files in the CaSR directory."""
        nc_files = list(self.casr_dir.glob("*.nc"))
        
        precip_files = []
        swe_files = []
        
        for f in nc_files:
            if "A_PR24_SFC" in f.name:
                precip_files.append(f)
            elif "P_SWE_LAND" in f.name:
                swe_files.append(f)
        
        logger.info(f"Found {len(precip_files)} precipitation files")
        logger.info(f"Found {len(swe_files)} SWE files")
        
        return precip_files, swe_files
    
    def extract_data_at_points(self, nc_file, points_gdf, variable_name, sample_time=None):
        """
        Extract data from NetCDF file at elevation points.
        
        Parameters:
        -----------
        nc_file : Path
            Path to NetCDF file
        points_gdf : GeoDataFrame
            Points where to extract data
        variable_name : str
            Name of the variable being extracted
        sample_time : int, optional
            Number of time steps to sample
            
        Returns:
        --------
        pandas.DataFrame
            Extracted data
        """
        logger.info(f"Extracting {variable_name} data from {nc_file.name}...")
        
        try:
            # Open NetCDF file
            ds = xr.open_dataset(nc_file)
            
            # Get data variable name
            data_vars = [v for v in ds.data_vars if v != 'rotated_pole']
            if not data_vars:
                logger.warning(f"No data variables found in {nc_file.name}")
                return None
            
            var_name = data_vars[0]  # Assume first variable is the main data
            logger.info(f"Extracting variable: {var_name}")
            
            # Sample time if requested
            if sample_time and 'time' in ds.dims and ds.dims['time'] > sample_time:
                logger.info(f"Sampling {sample_time} time steps from {ds.dims['time']} total")
                time_indices = np.linspace(0, ds.dims['time']-1, sample_time, dtype=int)
                ds = ds.isel(time=time_indices)
            
            # Get coordinate information
            if 'lon' in ds.coords and 'lat' in ds.coords:
                lon_coord, lat_coord = 'lon', 'lat'
            elif 'rlon' in ds.coords and 'rlat' in ds.coords:
                lon_coord, lat_coord = 'rlon', 'rlat'
            else:
                logger.warning(f"Could not identify coordinate variables")
                return None
            
            # Convert points to same CRS as NetCDF if needed
            points_proj = points_gdf.copy()
            if points_proj.crs and points_proj.crs.to_string() != 'EPSG:4326':
                points_proj = points_proj.to_crs('EPSG:4326')
            
            # Extract data at each point
            extracted_data = []
            
            for idx, row in points_proj.iterrows():
                # Get point coordinates
                geom = row.geometry
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    lon, lat = geom.x, geom.y
                else:
                    centroid = geom.centroid
                    lon, lat = centroid.x, centroid.y
                
                try:
                    # Find nearest grid point
                    if 'lon' in ds.coords and 'lat' in ds.coords and ds.lon.ndim == 2:
                        # Handle 2D coordinate arrays
                        lon_2d = ds.lon.values
                        lat_2d = ds.lat.values
                        
                        # Convert longitude if needed
                        target_lon = lon if lon < 0 else lon - 360
                        lon_2d_adj = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
                        
                        # Find nearest point
                        dist = np.sqrt((lon_2d_adj - target_lon)**2 + (lat_2d - lat)**2)
                        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                        rlat_idx, rlon_idx = min_idx
                        
                        point_data = ds.isel(rlat=rlat_idx, rlon=rlon_idx)
                    else:
                        # Simple nearest neighbor selection
                        point_data = ds.sel({lon_coord: lon, lat_coord: lat}, method='nearest')
                    
                    # Extract time series data
                    if 'time' in point_data[var_name].dims:
                        times = point_data.time.values
                        values = point_data[var_name].values
                        
                        for t, v in zip(times, values):
                            data_dict = {
                                'point_id': idx,
                                'lon': lon,
                                'lat': lat,
                                'time': pd.to_datetime(t),
                                variable_name: float(v) if not np.isnan(v) else np.nan
                            }
                            
                            # Add elevation data
                            for col in self.elev_cols:
                                if col in row and pd.api.types.is_numeric_dtype(type(row[col])):
                                    data_dict[f'elevation_{col}'] = row[col]
                            
                            extracted_data.append(data_dict)
                    else:
                        # Single value
                        data_dict = {
                            'point_id': idx,
                            'lon': lon,
                            'lat': lat,
                            variable_name: float(point_data[var_name].values)
                        }
                        
                        # Add elevation data
                        for col in self.elev_cols:
                            if col in row and pd.api.types.is_numeric_dtype(type(row[col])):
                                data_dict[f'elevation_{col}'] = row[col]
                        
                        extracted_data.append(data_dict)
                        
                except Exception as e:
                    logger.warning(f"Could not extract data for point {idx}: {e}")
                    continue
            
            ds.close()
            
            if extracted_data:
                df = pd.DataFrame(extracted_data)
                logger.info(f"Extracted {len(df)} records for {variable_name}")
                return df
            else:
                logger.warning(f"No data extracted from {nc_file.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}")
            return None
    
    def filter_and_merge_data(self, precip_df, swe_df):
        """
        Filter and merge precipitation and SWE data for non-null values.
        
        Parameters:
        -----------
        precip_df : DataFrame
            Precipitation data
        swe_df : DataFrame
            SWE data
            
        Returns:
        --------
        DataFrame
            Merged data with non-null values
        """
        logger.info("Filtering and merging data...")
        
        # Merge on common keys
        merge_keys = ['point_id', 'lon', 'lat']
        if 'time' in precip_df.columns and 'time' in swe_df.columns:
            merge_keys.append('time')
        
        # Merge dataframes
        merged_df = pd.merge(
            precip_df,
            swe_df,
            on=merge_keys,
            suffixes=('_precip', '_swe'),
            how='inner'
        )
        
        # Get elevation columns (handle duplicates from merge)
        elev_cols_merged = [col for col in merged_df.columns if col.startswith('elevation_')]
        
        # Remove duplicate elevation columns
        for col in elev_cols_merged:
            if col.endswith('_swe') and col.replace('_swe', '_precip') in merged_df.columns:
                # Keep only one version
                merged_df[col.replace('_swe', '')] = merged_df[col]
                merged_df = merged_df.drop([col, col.replace('_swe', '_precip')], axis=1)
        
        # Filter for non-null values
        precip_col = [col for col in merged_df.columns if 'precipitation' in col.lower() or 'PR24' in col][0]
        swe_col = [col for col in merged_df.columns if 'swe' in col.lower() or 'SWE' in col][0]
        
        logger.info(f"Total merged records: {len(merged_df)}")
        logger.info(f"Records with null precipitation: {merged_df[precip_col].isna().sum()}")
        logger.info(f"Records with null SWE: {merged_df[swe_col].isna().sum()}")
        
        # Filter for non-null values in both variables
        filtered_df = merged_df[
            merged_df[precip_col].notna() & 
            merged_df[swe_col].notna()
        ].copy()
        
        logger.info(f"Records with non-null values in both variables: {len(filtered_df)}")
        
        return filtered_df, precip_col, swe_col
    
    def analyze_elevation_patterns(self, filtered_df, precip_col, swe_col):
        """
        Analyze patterns in the filtered data by elevation.
        
        Parameters:
        -----------
        filtered_df : DataFrame
            Filtered data with non-null values
        precip_col : str
            Name of precipitation column
        swe_col : str
            Name of SWE column
        """
        logger.info("Analyzing elevation patterns...")
        
        # Find elevation columns
        elev_cols = [col for col in filtered_df.columns if col.startswith('elevation_')]
        
        if not elev_cols:
            logger.warning("No elevation columns found for analysis")
            return None
        
        # Use the first elevation column for analysis
        elev_col = elev_cols[0]
        
        # Create elevation bins
        filtered_df['elevation_bin'] = pd.cut(filtered_df[elev_col], bins=10)
        
        # Calculate statistics by elevation bin
        stats_by_elevation = filtered_df.groupby('elevation_bin').agg({
            precip_col: ['mean', 'std', 'count'],
            swe_col: ['mean', 'std', 'count'],
            'point_id': 'nunique'
        }).round(2)
        
        # Rename columns for clarity
        stats_by_elevation.columns = [
            'precip_mean', 'precip_std', 'precip_count',
            'swe_mean', 'swe_std', 'swe_count',
            'unique_points'
        ]
        
        # Calculate correlation between variables
        if len(filtered_df) > 1:
            correlation = filtered_df[[precip_col, swe_col]].corr().iloc[0, 1]
            logger.info(f"Correlation between precipitation and SWE: {correlation:.3f}")
        
        return stats_by_elevation
    
    def save_results(self, filtered_df, stats_df, format='csv'):
        """
        Save filtered data and statistics.
        
        Parameters:
        -----------
        filtered_df : DataFrame
            Filtered data
        stats_df : DataFrame
            Statistics by elevation
        format : str
            Output format ('csv', 'parquet', 'both')
        """
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save filtered data
        if format in ['csv', 'both']:
            csv_file = self.output_dir / "filtered_elevation_data.csv"
            filtered_df.to_csv(csv_file, index=False)
            logger.info(f"Saved filtered data to: {csv_file}")
            
            if stats_df is not None:
                stats_csv = self.output_dir / "elevation_statistics.csv"
                stats_df.to_csv(stats_csv)
                logger.info(f"Saved statistics to: {stats_csv}")
        
        if format in ['parquet', 'both']:
            parquet_file = self.output_dir / "filtered_elevation_data.parquet"
            filtered_df.to_parquet(parquet_file, index=False)
            logger.info(f"Saved filtered data to: {parquet_file}")
            
            if stats_df is not None:
                stats_parquet = self.output_dir / "elevation_statistics.parquet"
                stats_df.to_parquet(stats_parquet)
                logger.info(f"Saved statistics to: {stats_parquet}")
        
        # Generate summary report
        self.generate_summary_report(filtered_df, stats_df)
    
    def generate_summary_report(self, filtered_df, stats_df):
        """Generate a summary report of the filtering results."""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'elevation_points_loaded': len(self.elevation_gdf) if self.elevation_gdf is not None else 0,
            'filtered_records': len(filtered_df),
            'unique_points_with_data': filtered_df['point_id'].nunique() if 'point_id' in filtered_df.columns else 0,
            'time_range': None
        }
        
        if 'time' in filtered_df.columns:
            summary['time_range'] = {
                'start': filtered_df['time'].min().isoformat() if pd.notna(filtered_df['time'].min()) else None,
                'end': filtered_df['time'].max().isoformat() if pd.notna(filtered_df['time'].max()) else None
            }
        
        # Save summary
        summary_file = self.output_dir / "filtering_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("ELEVATION DATA FILTERING SUMMARY")
        print("="*60)
        print(f"Elevation points loaded: {summary['elevation_points_loaded']}")
        print(f"Filtered records (non-null precip & SWE): {summary['filtered_records']}")
        print(f"Unique points with valid data: {summary['unique_points_with_data']}")
        if summary['time_range']:
            print(f"Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        
        if stats_df is not None:
            print("\nElevation Statistics:")
            print(stats_df)
        print("="*60)
    
    def process(self, sample_points=100, sample_time=10):
        """
        Main processing function.
        
        Parameters:
        -----------
        sample_points : int
            Number of elevation points to sample
        sample_time : int
            Number of time steps to sample
        """
        # Load elevation data
        self.load_elevation_data(sample_size=sample_points)
        
        # Find CaSR files
        precip_files, swe_files = self.find_casr_files()
        
        if not precip_files or not swe_files:
            logger.error("Missing precipitation or SWE files")
            return
        
        # Process first file of each type as a sample
        logger.info("Processing sample files...")
        
        # Extract precipitation data
        precip_df = self.extract_data_at_points(
            precip_files[0], 
            self.elevation_gdf, 
            'precipitation',
            sample_time=sample_time
        )
        
        # Extract SWE data
        swe_df = self.extract_data_at_points(
            swe_files[0], 
            self.elevation_gdf, 
            'swe',
            sample_time=sample_time
        )
        
        if precip_df is None or swe_df is None:
            logger.error("Failed to extract data")
            return
        
        # Filter and merge data
        filtered_df, precip_col, swe_col = self.filter_and_merge_data(precip_df, swe_df)
        
        # Analyze elevation patterns
        stats_df = self.analyze_elevation_patterns(filtered_df, precip_col, swe_col)
        
        # Save results
        self.save_results(filtered_df, stats_df, format='both')


def main():
    """Main function to run the filtering."""
    parser = argparse.ArgumentParser(
        description='Filter and merge elevation data for non-null precipitation and SWE values'
    )
    parser.add_argument('--elevation-dir', default='data/input_data/Elevation',
                       help='Directory containing elevation shapefiles')
    parser.add_argument('--casr-dir', default='data/output_data/combined_casr',
                       help='Directory containing CaSR NetCDF files')
    parser.add_argument('--output-dir', default='data/output_data/filtered_elevation',
                       help='Output directory for filtered data')
    parser.add_argument('--sample-points', type=int, default=100,
                       help='Number of elevation points to sample (default: 100)')
    parser.add_argument('--sample-time', type=int, default=10,
                       help='Number of time steps to sample (default: 10)')
    parser.add_argument('--format', choices=['csv', 'parquet', 'both'], default='both',
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize filter
        filter = ElevationDataFilter(
            elevation_dir=args.elevation_dir,
            casr_dir=args.casr_dir,
            output_dir=args.output_dir
        )
        
        # Process data
        filter.process(
            sample_points=args.sample_points,
            sample_time=args.sample_time
        )
        
        logger.info("Filtering completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
