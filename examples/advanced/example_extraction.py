#!/usr/bin/env python3
"""
Example usage of the elevation data extraction script.

This script demonstrates how to use the ElevationDataExtractor class
to extract CaSR data at elevation point locations.
"""

from examples.data_processing.extract_elevation_data_optimized import OptimizedElevationDataExtractor as ElevationDataExtractor
import logging

# Enable logging to see progress
logging.basicConfig(level=logging.INFO)

def main():
    """Example usage of the elevation data extractor."""
    
    print("Elevation Data Extraction Example")
    print("=" * 40)
    
    # Initialize the extractor
    extractor = ElevationDataExtractor(
        elevation_dir="data/input_data/Elevation",
        combined_casr_dir="data/output_data/combined_casr",
        output_dir="data/output_data/elevation"
    )
    
    # Example 1: Extract data from temporal files only
    print("\nExample 1: Processing temporal files only...")
    results_temporal = extractor.process_all_files(file_types=['temporal'])
    
    if results_temporal:
        extractor.save_results(results_temporal, format='csv')
        print(f"Extracted data from {len(results_temporal)} temporal files")
    
    # Example 2: Extract data from full combined files only
    print("\nExample 2: Processing full combined files only...")
    results_full = extractor.process_all_files(file_types=['full'])
    
    if results_full:
        extractor.save_results(results_full, format='csv')
        print(f"Extracted data from {len(results_full)} full files")
    
    # Example 3: Process all files and save in multiple formats
    print("\nExample 3: Processing all files with multiple output formats...")
    results_all = extractor.process_all_files(file_types=['temporal', 'full'])
    
    if results_all:
        # Save in both CSV and Parquet formats
        extractor.save_results(results_all, format='both')
        
        # Generate summary report
        extractor.generate_summary_report(results_all)
        
        print(f"Processed {len(results_all)} files total")
    
    print("\nExtraction examples completed!")

if __name__ == "__main__":
    main()
