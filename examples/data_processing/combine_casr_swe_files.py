#!/usr/bin/env python3
"""
Script to combine NetCDF files from the CaSR SWE dataset.

This script can combine NetCDF files in different ways:
1. Combine files across time periods (temporal concatenation)
2. Combine files across spatial regions (spatial merging)
3. Combine both temporal and spatial dimensions

The CaSR SWE dataset contains files organized by:
- Variable types: A_PR24_SFC (precipitation) and P_SWE_LAND (snow water equivalent)
- Spatial regions: Different rlon/rlat ranges
- Time periods: 4-year chunks from 1980-2023
"""

import os
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaSRFileCombiner:
    """Class to handle combining CaSR SWE NetCDF files."""
    
    def __init__(self, input_dir: str, output_dir: str = None):
        """
        Initialize the file combiner.
        
        Parameters
        ----------
        input_dir : str
            Directory containing the NetCDF files
        output_dir : str, optional
            Directory to save combined files. If None, uses input_dir parent + 'output_data'
        """
        self.input_dir = Path(input_dir)
        if output_dir is None:
            self.output_dir = self.input_dir.parent.parent / 'output_data' / 'combined_casr'
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse CaSR filename to extract metadata.
        
        Expected format: CaSR_v3.1_{VAR}_{SURFACE}_rlon{RLON_START}-{RLON_END}_rlat{RLAT_START}-{RLAT_END}_{YEAR_START}-{YEAR_END}.nc
        
        Parameters
        ----------
        filename : str
            NetCDF filename
            
        Returns
        -------
        dict
            Dictionary with parsed components
        """
        # Remove .nc extension and split by underscore
        parts = filename.replace('.nc', '').split('_')
        
        try:
            # Handle different filename formats
            if len(parts) >= 7:
                # Format: CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1980-1983
                # or: CaSR_v3.1_P_SWE_LAND_rlon211-245_rlat386-420_1980-1983
                if parts[2] == 'P' and parts[3] == 'SWE' and parts[4] == 'LAND':
                    # P_SWE_LAND format
                    variable = f"{parts[2]}_{parts[3]}_{parts[4]}"
                    rlon_part = parts[5]
                    rlat_part = parts[6]
                    time_part = parts[7]
                    surface = "LAND"
                elif parts[2] == 'A' and parts[3] == 'PR24' and parts[4] == 'SFC':
                    # A_PR24_SFC format
                    variable = f"{parts[2]}_{parts[3]}"
                    surface = parts[4]
                    rlon_part = parts[5]
                    rlat_part = parts[6]
                    time_part = parts[7]
                else:
                    logger.error(f"Unknown filename format: {filename}")
                    return {}
                
                return {
                    'version': parts[1],
                    'variable': variable,
                    'surface': surface,
                    'rlon_range': rlon_part,
                    'rlat_range': rlat_part,
                    'time_range': time_part,
                    'rlon_start': int(rlon_part.split('rlon')[1].split('-')[0]),
                    'rlon_end': int(rlon_part.split('-')[1]),
                    'rlat_start': int(rlat_part.split('rlat')[1].split('-')[0]),
                    'rlat_end': int(rlat_part.split('-')[1]),
                    'year_start': int(time_part.split('-')[0]),
                    'year_end': int(time_part.split('-')[1])
                }
            else:
                logger.error(f"Insufficient parts in filename {filename}: {len(parts)} parts")
                return {}
                
        except (IndexError, ValueError) as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            logger.debug(f"Filename parts: {parts}")
            return {}
    
    def get_file_groups(self) -> Dict[str, List[str]]:
        """
        Group files by variable and surface type.
        
        Returns
        -------
        dict
            Dictionary with groups of files
        """
        nc_files = list(self.input_dir.glob('*.nc'))
        groups = {}
        
        for file_path in nc_files:
            filename = file_path.name
            parsed = self.parse_filename(filename)
            
            if parsed:
                # Create a clean key without redundant surface info
                if parsed['variable'] == 'P_SWE_LAND':
                    key = parsed['variable']  # Already includes LAND
                else:
                    key = f"{parsed['variable']}_{parsed['surface']}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(str(file_path))
        
        logger.info(f"Found {len(groups)} file groups:")
        for key, files in groups.items():
            logger.info(f"  {key}: {len(files)} files")
        
        return groups
    
    def combine_temporal(self, file_list: List[str], output_filename: str) -> None:
        """
        Combine files along the time dimension.
        
        Parameters
        ----------
        file_list : list
            List of NetCDF file paths to combine
        output_filename : str
            Name of the output file
        """
        logger.info(f"Combining {len(file_list)} files temporally...")
        
        # Sort files by time range
        file_info = []
        for file_path in file_list:
            parsed = self.parse_filename(Path(file_path).name)
            if parsed:
                file_info.append((file_path, parsed['year_start']))
        
        file_info.sort(key=lambda x: x[1])
        sorted_files = [x[0] for x in file_info]
        
        # Open and combine datasets
        datasets = []
        for file_path in sorted_files:
            logger.info(f"  Loading: {Path(file_path).name}")
            ds = xr.open_dataset(file_path)
            datasets.append(ds)
        
        # Concatenate along time dimension
        logger.info("  Concatenating datasets...")
        combined = xr.concat(datasets, dim='time')
        
        # Sort by time to ensure proper ordering
        combined = combined.sortby('time')
        
        # Add metadata
        combined.attrs['title'] = f'Combined CaSR dataset - {output_filename}'
        combined.attrs['source'] = f'Combined from {len(file_list)} files'
        combined.attrs['creation_date'] = datetime.now().isoformat()
        combined.attrs['combined_files'] = [Path(f).name for f in sorted_files]
        
        # Save combined dataset
        output_path = self.output_dir / output_filename
        logger.info(f"  Saving to: {output_path}")
        combined.to_netcdf(output_path, engine='netcdf4')
        
        # Close datasets to free memory
        for ds in datasets:
            ds.close()
        combined.close()
        
        logger.info(f"  Successfully saved combined file: {output_path}")
    
    def combine_spatial_regions(self, file_list: List[str], output_filename: str) -> None:
        """
        Combine files across spatial regions for the same time period.
        
        Parameters
        ----------
        file_list : list
            List of NetCDF file paths to combine
        output_filename : str
            Name of the output file
        """
        logger.info(f"Combining {len(file_list)} files spatially...")
        
        # Group files by time period
        time_groups = {}
        for file_path in file_list:
            parsed = self.parse_filename(Path(file_path).name)
            if parsed:
                time_key = parsed['time_range']
                if time_key not in time_groups:
                    time_groups[time_key] = []
                time_groups[time_key].append((file_path, parsed))
        
        # Process each time period
        combined_datasets = []
        for time_key in sorted(time_groups.keys()):
            logger.info(f"  Processing time period: {time_key}")
            time_files = time_groups[time_key]
            
            # Sort by spatial coordinates
            time_files.sort(key=lambda x: (x[1]['rlon_start'], x[1]['rlat_start']))
            
            # Load datasets for this time period
            time_datasets = []
            for file_path, parsed in time_files:
                logger.info(f"    Loading: {Path(file_path).name}")
                ds = xr.open_dataset(file_path)
                time_datasets.append(ds)
            
            # Combine spatially using xarray's combine functionality
            logger.info(f"    Combining {len(time_datasets)} spatial regions...")
            try:
                # Try to combine by coordinates
                combined_time = xr.combine_by_coords(time_datasets)
            except Exception as e:
                logger.warning(f"    combine_by_coords failed: {e}")
                logger.info("    Trying manual concatenation...")
                # Fallback: concatenate along spatial dimensions
                combined_time = xr.concat(time_datasets, dim='rlon')
            
            combined_datasets.append(combined_time)
            
            # Close time datasets to free memory
            for ds in time_datasets:
                ds.close()
        
        # Combine all time periods
        logger.info("  Combining all time periods...")
        final_combined = xr.concat(combined_datasets, dim='time')
        final_combined = final_combined.sortby('time')
        
        # Add metadata
        final_combined.attrs['title'] = f'Spatially combined CaSR dataset - {output_filename}'
        final_combined.attrs['source'] = f'Combined from {len(file_list)} files'
        final_combined.attrs['creation_date'] = datetime.now().isoformat()
        final_combined.attrs['combined_files'] = [Path(f).name for f in file_list]
        
        # Save combined dataset
        output_path = self.output_dir / output_filename
        logger.info(f"  Saving to: {output_path}")
        final_combined.to_netcdf(output_path, engine='netcdf4')
        
        # Close datasets to free memory
        for ds in combined_datasets:
            ds.close()
        final_combined.close()
        
        logger.info(f"  Successfully saved spatially combined file: {output_path}")
    
    def combine_by_variable(self, combine_spatial: bool = True, combine_temporal: bool = True) -> None:
        """
        Combine files grouped by variable type.
        
        Parameters
        ----------
        combine_spatial : bool
            Whether to combine across spatial regions
        combine_temporal : bool
            Whether to combine across time periods
        """
        file_groups = self.get_file_groups()
        
        for group_name, file_list in file_groups.items():
            logger.info(f"\nProcessing group: {group_name}")
            
            if combine_spatial and combine_temporal:
                # Combine both spatial and temporal
                output_filename = f"CaSR_v3.1_{group_name}_combined_full.nc"
                self.combine_spatial_regions(file_list, output_filename)
            
            elif combine_temporal:
                # Group by spatial region and combine temporally
                spatial_groups = {}
                for file_path in file_list:
                    parsed = self.parse_filename(Path(file_path).name)
                    if parsed:
                        spatial_key = f"{parsed['rlon_range']}_{parsed['rlat_range']}"
                        if spatial_key not in spatial_groups:
                            spatial_groups[spatial_key] = []
                        spatial_groups[spatial_key].append(file_path)
                
                for spatial_key, spatial_files in spatial_groups.items():
                    output_filename = f"CaSR_v3.1_{group_name}_{spatial_key}_temporal_combined.nc"
                    self.combine_temporal(spatial_files, output_filename)
            
            elif combine_spatial:
                # Group by time period and combine spatially
                time_groups = {}
                for file_path in file_list:
                    parsed = self.parse_filename(Path(file_path).name)
                    if parsed:
                        time_key = parsed['time_range']
                        if time_key not in time_groups:
                            time_groups[time_key] = []
                        time_groups[time_key].append(file_path)
                
                for time_key, time_files in time_groups.items():
                    output_filename = f"CaSR_v3.1_{group_name}_{time_key}_spatial_combined.nc"
                    self.combine_spatial_regions(time_files, output_filename)
    
    def get_dataset_info(self) -> None:
        """Print information about the datasets in the input directory."""
        file_groups = self.get_file_groups()
        
        print("\n" + "="*60)
        print("CASR SWE DATASET INFORMATION")
        print("="*60)
        
        for group_name, file_list in file_groups.items():
            print(f"\nGroup: {group_name}")
            print(f"Number of files: {len(file_list)}")
            
            # Get time and spatial coverage
            time_ranges = []
            spatial_regions = []
            
            for file_path in file_list:
                parsed = self.parse_filename(Path(file_path).name)
                if parsed:
                    time_ranges.append((parsed['year_start'], parsed['year_end']))
                    spatial_regions.append((parsed['rlon_start'], parsed['rlon_end'], 
                                          parsed['rlat_start'], parsed['rlat_end']))
            
            if time_ranges:
                min_year = min(t[0] for t in time_ranges)
                max_year = max(t[1] for t in time_ranges)
                print(f"Time coverage: {min_year}-{max_year}")
            
            if spatial_regions:
                unique_regions = list(set(spatial_regions))
                print(f"Spatial regions: {len(unique_regions)}")
                for region in sorted(unique_regions):
                    print(f"  rlon {region[0]}-{region[1]}, rlat {region[2]}-{region[3]}")
            
            # Sample a file to get variable information
            if file_list:
                try:
                    sample_ds = xr.open_dataset(file_list[0])
                    print(f"Variables: {list(sample_ds.data_vars.keys())}")
                    print(f"Dimensions: {dict(sample_ds.sizes)}")
                    sample_ds.close()
                except Exception as e:
                    print(f"Error reading sample file: {e}")


def main():
    """Main function to run the file combiner."""
    parser = argparse.ArgumentParser(
        description="Combine CaSR SWE NetCDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all files (spatial and temporal)
  python combine_casr_swe_files.py --input data/input_data/CaSR_SWE --combine-all
  
  # Only combine temporally (keep spatial regions separate)
  python combine_casr_swe_files.py --input data/input_data/CaSR_SWE --temporal-only
  
  # Only combine spatially (keep time periods separate)
  python combine_casr_swe_files.py --input data/input_data/CaSR_SWE --spatial-only
  
  # Get information about the dataset
  python combine_casr_swe_files.py --input data/input_data/CaSR_SWE --info-only
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/input_data/CaSR_SWE',
        help='Input directory containing NetCDF files (default: data/input_data/CaSR_SWE)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for combined files (default: data/output_data/combined_casr)'
    )
    
    parser.add_argument(
        '--combine-all',
        action='store_true',
        help='Combine files across both spatial and temporal dimensions'
    )
    
    parser.add_argument(
        '--temporal-only',
        action='store_true',
        help='Only combine files across time (keep spatial regions separate)'
    )
    
    parser.add_argument(
        '--spatial-only',
        action='store_true',
        help='Only combine files across space (keep time periods separate)'
    )
    
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='Only display information about the dataset without combining'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize combiner
    combiner = CaSRFileCombiner(args.input, args.output)
    
    if args.info_only:
        combiner.get_dataset_info()
        return
    
    # Determine combination strategy
    if args.combine_all:
        combiner.combine_by_variable(combine_spatial=True, combine_temporal=True)
    elif args.temporal_only:
        combiner.combine_by_variable(combine_spatial=False, combine_temporal=True)
    elif args.spatial_only:
        combiner.combine_by_variable(combine_spatial=True, combine_temporal=False)
    else:
        # Default: show info and ask user
        combiner.get_dataset_info()
        print("\nNo combination option specified. Use --help to see available options.")
        print("Use --combine-all, --temporal-only, --spatial-only, or --info-only")


if __name__ == "__main__":
    main()
