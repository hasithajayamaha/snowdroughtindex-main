#!/usr/bin/env python3
"""
Filter and merge elevation data for non-null precipitation and snow water equivalent values.

This script:
1. Loads pre-extracted elevation data from CSV files
2. Merges precipitation and SWE data at elevation points
3. Filters for non-null values in both variables
4. Provides analysis and saves results
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
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
    
    def __init__(self, elevation_data_dir=None, output_dir=None, elevation_dir=None, casr_dir=None):
        """
        Initialize the filter.
        
        Parameters:
        -----------
        elevation_data_dir : str, optional
            Path to directory containing flattened elevation CSV files (new approach)
        output_dir : str, optional
            Output directory for filtered data
        elevation_dir : str, optional
            Path to directory containing elevation shapefiles (legacy compatibility)
        casr_dir : str, optional
            Path to directory containing CaSR NetCDF files (legacy compatibility)
        """
        # Handle legacy constructor calls from notebook
        if elevation_dir is not None and casr_dir is not None:
            # Legacy mode: assume we're working with flattened CSV files in a standard location
            self.elevation_data_dir = Path("data/output_data/elevation")
            print("Legacy constructor detected. Using flattened CSV files from data/output_data/elevation")
        elif elevation_data_dir is not None:
            # New mode: direct path to CSV files
            self.elevation_data_dir = Path(elevation_data_dir)
        else:
            # Default mode
            self.elevation_data_dir = Path("data/output_data/elevation")
        
        self.output_dir = Path(output_dir) if output_dir else Path("data/output_data/filtered_elevation")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store data
        self.precipitation_df = None
        self.swe_df = None
        
    def find_elevation_csv_files(self):
        """Find precipitation and SWE CSV files in the elevation data directory."""
        csv_files = list(self.elevation_data_dir.glob("*.csv"))
        
        precip_file = None
        swe_file = None
        
        for f in csv_files:
            if "A_PR24_SFC" in f.name:
                precip_file = f
            elif "P_SWE_LAND" in f.name:
                swe_file = f
        
        if not precip_file:
            raise FileNotFoundError(f"No precipitation CSV file found in {self.elevation_data_dir}")
        if not swe_file:
            raise FileNotFoundError(f"No SWE CSV file found in {self.elevation_data_dir}")
            
        logger.info(f"Found precipitation file: {precip_file.name}")
        logger.info(f"Found SWE file: {swe_file.name}")
        
        return precip_file, swe_file
    
    def load_csv_data(self, csv_file, variable_name, sample_size=None, chunk_size=None):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        csv_file : Path
            Path to CSV file
        variable_name : str
            Name of the variable being loaded
        sample_size : int, optional
            Number of records to sample for testing
        chunk_size : int, optional
            Size of chunks for memory-efficient loading
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        logger.info(f"Loading {variable_name} data from {csv_file.name}...")
        
        try:
            if chunk_size:
                # Load in chunks for memory efficiency
                logger.info(f"Loading data in chunks of {chunk_size:,} records")
                chunks = []
                total_rows = 0
                
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                    # Convert time column to datetime
                    if 'time' in chunk.columns:
                        chunk['time'] = pd.to_datetime(chunk['time'])
                    
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    if sample_size and total_rows >= sample_size:
                        logger.info(f"Reached sample size limit of {sample_size:,} records")
                        break
                
                df = pd.concat(chunks, ignore_index=True)
                
                if sample_size and len(df) > sample_size:
                    df = df.head(sample_size)
                    
            else:
                # Load entire file
                df = pd.read_csv(csv_file)
                
                # Convert time column to datetime
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                
                # Sample data if requested
                if sample_size and len(df) > sample_size:
                    logger.info(f"Sampling {sample_size:,} records from {len(df):,} total records")
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            logger.info(f"Loaded {len(df):,} records for {variable_name}")
            logger.info(f"Data columns: {list(df.columns)}")
            
            # Identify the data variable column
            data_cols = [col for col in df.columns if 'CaSR' in col]
            if data_cols:
                data_col = data_cols[0]
                logger.info(f"Data variable: {data_col}")
                
                # Check for null values
                null_count = df[data_col].isna().sum()
                logger.info(f"Null values in {data_col}: {null_count:,} ({null_count/len(df)*100:.1f}%)")
                
                # Show data range for non-null values
                non_null_data = df[data_col].dropna()
                if len(non_null_data) > 0:
                    logger.info(f"{data_col} range: {non_null_data.min():.3f} - {non_null_data.max():.3f}")
            
            # Show time range
            if 'time' in df.columns:
                time_range = f"{df['time'].min()} to {df['time'].max()}"
                logger.info(f"Time range: {time_range}")
            
            # Show elevation info
            elev_cols = [col for col in df.columns if 'elevation_' in col]
            if elev_cols:
                logger.info(f"Elevation columns: {elev_cols}")
                for col in elev_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        logger.info(f"{col} range: {df[col].min():.1f} - {df[col].max():.1f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
            raise
    
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
        tuple
            (merged_filtered_df, precip_col, swe_col)
        """
        logger.info("Filtering and merging data...")
        
        # Identify data columns
        precip_cols = [col for col in precip_df.columns if 'CaSR' in col and 'PR24' in col]
        swe_cols = [col for col in swe_df.columns if 'CaSR' in col and 'SWE' in col]
        
        if not precip_cols:
            raise ValueError("No precipitation data column found")
        if not swe_cols:
            raise ValueError("No SWE data column found")
            
        precip_col = precip_cols[0]
        swe_col = swe_cols[0]
        
        logger.info(f"Precipitation column: {precip_col}")
        logger.info(f"SWE column: {swe_col}")
        
        # Merge on common keys
        merge_keys = ['point_id', 'original_lon', 'original_lat', 'grid_lon', 'grid_lat', 'time']
        
        # Check if all merge keys exist in both dataframes
        missing_keys_precip = [key for key in merge_keys if key not in precip_df.columns]
        missing_keys_swe = [key for key in merge_keys if key not in swe_df.columns]
        
        if missing_keys_precip:
            logger.warning(f"Missing keys in precipitation data: {missing_keys_precip}")
        if missing_keys_swe:
            logger.warning(f"Missing keys in SWE data: {missing_keys_swe}")
        
        # Use only available merge keys
        available_keys = [key for key in merge_keys if key in precip_df.columns and key in swe_df.columns]
        logger.info(f"Merging on keys: {available_keys}")
        
        # Merge dataframes
        logger.info("Performing merge operation...")
        merged_df = pd.merge(
            precip_df,
            swe_df,
            on=available_keys,
            suffixes=('_precip', '_swe'),
            how='inner'
        )
        
        logger.info(f"Merged dataset size: {len(merged_df):,} records")
        
        # Handle duplicate elevation columns from merge
        elev_cols_to_clean = []
        for col in merged_df.columns:
            if col.startswith('elevation_') and col.endswith('_swe'):
                base_col = col.replace('_swe', '')
                precip_col_name = base_col + '_precip'
                if precip_col_name in merged_df.columns:
                    # Keep the precipitation version and rename it
                    merged_df[base_col] = merged_df[precip_col_name]
                    elev_cols_to_clean.extend([col, precip_col_name])
                else:
                    # Just rename the SWE version
                    merged_df[base_col] = merged_df[col]
                    elev_cols_to_clean.append(col)
        
        # Remove duplicate elevation columns
        if elev_cols_to_clean:
            merged_df = merged_df.drop(columns=elev_cols_to_clean)
            logger.info(f"Cleaned up duplicate elevation columns: {elev_cols_to_clean}")
        
        # Get the actual column names after merge (they might have suffixes)
        final_precip_col = precip_col if precip_col in merged_df.columns else precip_col + '_precip'
        final_swe_col = swe_col if swe_col in merged_df.columns else swe_col + '_swe'
        
        logger.info(f"Final precipitation column: {final_precip_col}")
        logger.info(f"Final SWE column: {final_swe_col}")
        
        # Check data before filtering
        logger.info(f"Total merged records: {len(merged_df):,}")
        logger.info(f"Records with null precipitation: {merged_df[final_precip_col].isna().sum():,}")
        logger.info(f"Records with null SWE: {merged_df[final_swe_col].isna().sum():,}")
        
        # Filter for non-null values in both variables
        filtered_df = merged_df[
            merged_df[final_precip_col].notna() & 
            merged_df[final_swe_col].notna()
        ].copy()
        
        logger.info(f"Records with non-null values in both variables: {len(filtered_df):,}")
        
        if len(filtered_df) == 0:
            logger.warning("No records remain after filtering for non-null values!")
            return filtered_df, final_precip_col, final_swe_col
        
        # Additional filtering for reasonable values (remove zeros and negative values if appropriate)
        logger.info("Applying additional data quality filters...")
        
        # For precipitation, remove negative values
        precip_negative = (filtered_df[final_precip_col] < 0).sum()
        if precip_negative > 0:
            logger.info(f"Removing {precip_negative:,} records with negative precipitation")
            filtered_df = filtered_df[filtered_df[final_precip_col] >= 0]
        
        # For SWE, remove negative values
        swe_negative = (filtered_df[final_swe_col] < 0).sum()
        if swe_negative > 0:
            logger.info(f"Removing {swe_negative:,} records with negative SWE")
            filtered_df = filtered_df[filtered_df[final_swe_col] >= 0]
        
        logger.info(f"Final filtered dataset size: {len(filtered_df):,} records")
        
        return filtered_df, final_precip_col, final_swe_col
    
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
        
        if len(filtered_df) == 0:
            logger.warning("No data available for elevation analysis")
            return None
        
        # Find elevation columns
        elev_cols = [col for col in filtered_df.columns if col.startswith('elevation_')]
        
        if not elev_cols:
            logger.warning("No elevation columns found for analysis")
            return None
        
        # Use mean elevation for analysis
        elev_col = 'elevation_mean' if 'elevation_mean' in elev_cols else elev_cols[0]
        logger.info(f"Using elevation column: {elev_col}")
        
        # Create elevation bins
        try:
            filtered_df = filtered_df.copy()
            filtered_df['elevation_bin'] = pd.cut(filtered_df[elev_col], bins=10, precision=0)
            
            # Calculate statistics by elevation bin
            stats_by_elevation = filtered_df.groupby('elevation_bin', observed=True).agg({
                precip_col: ['mean', 'std', 'count', 'min', 'max'],
                swe_col: ['mean', 'std', 'count', 'min', 'max'],
                'point_id': 'nunique'
            }).round(3)
            
            # Flatten column names
            stats_by_elevation.columns = [
                'precip_mean', 'precip_std', 'precip_count', 'precip_min', 'precip_max',
                'swe_mean', 'swe_std', 'swe_count', 'swe_min', 'swe_max',
                'unique_points'
            ]
            
            # Calculate correlation between variables
            correlation = filtered_df[[precip_col, swe_col]].corr().iloc[0, 1]
            logger.info(f"Correlation between precipitation and SWE: {correlation:.3f}")
            
            # Calculate overall statistics
            overall_stats = {
                'total_records': len(filtered_df),
                'unique_points': filtered_df['point_id'].nunique(),
                'time_span_days': (filtered_df['time'].max() - filtered_df['time'].min()).days,
                'elevation_range': f"{filtered_df[elev_col].min():.1f} - {filtered_df[elev_col].max():.1f}",
                'precip_correlation_with_elevation': filtered_df[[elev_col, precip_col]].corr().iloc[0, 1],
                'swe_correlation_with_elevation': filtered_df[[elev_col, swe_col]].corr().iloc[0, 1]
            }
            
            logger.info("Overall Statistics:")
            for key, value in overall_stats.items():
                logger.info(f"  {key}: {value}")
            
            return stats_by_elevation, overall_stats
            
        except Exception as e:
            logger.error(f"Error in elevation analysis: {e}")
            return None, None
    
    def save_results(self, filtered_df, stats_df, overall_stats=None, format='csv'):
        """
        Save filtered data and statistics.
        
        Parameters:
        -----------
        filtered_df : DataFrame
            Filtered data
        stats_df : DataFrame
            Statistics by elevation
        overall_stats : dict
            Overall statistics
        format : str
            Output format ('csv', 'parquet', 'both')
        """
        logger.info(f"Saving results to {self.output_dir}")
        
        if len(filtered_df) == 0:
            logger.warning("No data to save")
            return
        
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
                try:
                    # Convert interval index to string for parquet compatibility
                    stats_df_copy = stats_df.copy()
                    stats_df_copy.index = stats_df_copy.index.astype(str)
                    stats_parquet = self.output_dir / "elevation_statistics.parquet"
                    stats_df_copy.to_parquet(stats_parquet)
                    logger.info(f"Saved statistics to: {stats_parquet}")
                except Exception as e:
                    logger.warning(f"Could not save statistics to parquet format: {e}")
                    logger.info("Statistics saved in CSV format only")
        
        # Generate summary report
        self.generate_summary_report(filtered_df, stats_df, overall_stats)
    
    def generate_summary_report(self, filtered_df, stats_df, overall_stats=None):
        """Generate a summary report of the filtering results."""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'source_files': {
                'precipitation': str(self.precipitation_file.name) if hasattr(self, 'precipitation_file') else 'unknown',
                'swe': str(self.swe_file.name) if hasattr(self, 'swe_file') else 'unknown'
            },
            'filtered_records': len(filtered_df),
            'unique_points_with_data': filtered_df['point_id'].nunique() if len(filtered_df) > 0 else 0,
            'time_range': None
        }
        
        if len(filtered_df) > 0 and 'time' in filtered_df.columns:
            summary['time_range'] = {
                'start': filtered_df['time'].min().isoformat(),
                'end': filtered_df['time'].max().isoformat()
            }
        
        if overall_stats:
            summary['overall_statistics'] = overall_stats
        
        # Save summary
        summary_file = self.output_dir / "filtering_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("ELEVATION DATA FILTERING SUMMARY")
        print("="*70)
        print(f"Processing date: {summary['processing_date']}")
        print(f"Source files:")
        print(f"  Precipitation: {summary['source_files']['precipitation']}")
        print(f"  SWE: {summary['source_files']['swe']}")
        print(f"Filtered records (non-null precip & SWE): {summary['filtered_records']:,}")
        print(f"Unique points with valid data: {summary['unique_points_with_data']}")
        
        if summary['time_range']:
            print(f"Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        
        if overall_stats:
            print(f"\nOverall Statistics:")
            print(f"  Time span: {overall_stats['time_span_days']:,} days")
            print(f"  Elevation range: {overall_stats['elevation_range']}")
            print(f"  Precip-Elevation correlation: {overall_stats['precip_correlation_with_elevation']:.3f}")
            print(f"  SWE-Elevation correlation: {overall_stats['swe_correlation_with_elevation']:.3f}")
        
        if stats_df is not None and len(stats_df) > 0:
            print("\nElevation Bin Statistics:")
            print(stats_df.to_string())
        
        print("="*70)
    
    def process(self, sample_size=None, chunk_size=None, sample_points=None, sample_time=None):
        """
        Main processing function.
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of records to sample from each file
        chunk_size : int, optional
            Size of chunks for memory-efficient loading
        sample_points : int, optional
            Legacy parameter (ignored - for notebook compatibility)
        sample_time : int, optional
            Legacy parameter (ignored - for notebook compatibility)
        """
        # Handle legacy parameters from notebook
        if sample_points is not None or sample_time is not None:
            logger.info("Legacy parameters detected. Using flattened CSV approach.")
            if sample_time and not sample_size:
                sample_size = sample_time  # Use sample_time as sample_size for compatibility
        try:
            # Find CSV files
            self.precipitation_file, self.swe_file = self.find_elevation_csv_files()
            
            # Load precipitation data
            self.precipitation_df = self.load_csv_data(
                self.precipitation_file, 
                'precipitation',
                sample_size=sample_size,
                chunk_size=chunk_size
            )
            
            # Load SWE data
            self.swe_df = self.load_csv_data(
                self.swe_file, 
                'swe',
                sample_size=sample_size,
                chunk_size=chunk_size
            )
            
            if self.precipitation_df is None or self.swe_df is None:
                logger.error("Failed to load data")
                return False
            
            # Filter and merge data
            filtered_df, precip_col, swe_col = self.filter_and_merge_data(
                self.precipitation_df, 
                self.swe_df
            )
            
            if len(filtered_df) == 0:
                logger.error("No data remains after filtering")
                return False
            
            # Analyze elevation patterns
            result = self.analyze_elevation_patterns(filtered_df, precip_col, swe_col)
            if result is not None:
                stats_df, overall_stats = result
            else:
                stats_df, overall_stats = None, None
            
            # Save results
            self.save_results(filtered_df, stats_df, overall_stats, format='both')
            
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False


def main():
    """Main function to run the filtering."""
    parser = argparse.ArgumentParser(
        description='Filter and merge elevation data for non-null precipitation and SWE values'
    )
    parser.add_argument('--elevation-data-dir', default='data/output_data/elevation',
                       help='Directory containing flattened elevation CSV files')
    parser.add_argument('--output-dir', default='data/output_data/filtered_elevation',
                       help='Output directory for filtered data')
    parser.add_argument('--sample-size', type=int,
                       help='Number of records to sample from each file (for testing)')
    parser.add_argument('--chunk-size', type=int,
                       help='Size of chunks for memory-efficient loading')
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
            elevation_data_dir=args.elevation_data_dir,
            output_dir=args.output_dir
        )
        
        # Process data
        success = filter.process(
            sample_size=args.sample_size,
            chunk_size=args.chunk_size
        )
        
        if success:
            logger.info("Filtering completed successfully!")
            return 0
        else:
            logger.error("Filtering failed!")
            return 1
        
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
