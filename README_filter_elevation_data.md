# Filter and Merge Elevation Data Script

This script filters and merges elevation data based on non-null precipitation (CaSR_v3.1_A_PR24_SFC) and snow water equivalent (CaSR_v3.1_P_SWE_LAND) values.

## Overview

The `filter_merge_elevation_data.py` script performs the following operations:

1. **Loads elevation data** from shapefiles
2. **Extracts CaSR data** (precipitation and SWE) at elevation point locations
3. **Filters for non-null values** in both precipitation and SWE variables
4. **Merges the data** and provides statistical analysis by elevation bins
5. **Saves filtered results** in both CSV and Parquet formats

## Features

- **Sampling capability**: Can sample a subset of elevation points and time steps for testing
- **Automatic coordinate matching**: Handles different coordinate systems (regular lat/lon, rotated grids)
- **Elevation-based analysis**: Groups data by elevation bins and calculates statistics
- **Correlation analysis**: Calculates correlation between precipitation and SWE
- **Multiple output formats**: Supports CSV and Parquet output formats
- **Comprehensive reporting**: Generates summary statistics and JSON reports

## Requirements

The script requires the following Python packages:
- numpy
- pandas
- xarray
- geopandas
- shapely
- netCDF4

Install requirements:
```bash
pip install -r requirements_extraction.txt
```

## Usage

### Basic Usage

Run with default settings (100 sample points, 10 time steps):
```bash
python filter_merge_elevation_data.py
```

### Command Line Options

```bash
python filter_merge_elevation_data.py [OPTIONS]

Options:
  --elevation-dir PATH     Directory containing elevation shapefiles 
                          (default: data/input_data/Elevation)
  
  --casr-dir PATH         Directory containing CaSR NetCDF files
                          (default: data/output_data/combined_casr)
  
  --output-dir PATH       Output directory for filtered data
                          (default: data/output_data/filtered_elevation)
  
  --sample-points INT     Number of elevation points to sample
                          (default: 100)
  
  --sample-time INT       Number of time steps to sample
                          (default: 10)
  
  --format {csv,parquet,both}  Output format (default: both)
  
  -v, --verbose           Enable verbose logging
```

### Examples

1. **Process with 500 sample points and 50 time steps:**
   ```bash
   python filter_merge_elevation_data.py --sample-points 500 --sample-time 50
   ```

2. **Use custom directories:**
   ```bash
   python filter_merge_elevation_data.py \
     --elevation-dir /path/to/elevation/data \
     --casr-dir /path/to/casr/files \
     --output-dir /path/to/output
   ```

3. **Save only CSV format with verbose output:**
   ```bash
   python filter_merge_elevation_data.py --format csv --verbose
   ```

## Input Data Requirements

### Elevation Data
- Format: Shapefile (.shp)
- Location: `elevation-dir` directory
- Required: Point or polygon geometries with elevation attributes
- Elevation columns: Any column containing 'elev' in the name, or columns named 'min', 'max', 'mean', 'median'

### CaSR Data
- Format: NetCDF (.nc) files
- Location: `casr-dir` directory
- Required files:
  - Precipitation files: Must contain "A_PR24_SFC" in filename
  - SWE files: Must contain "P_SWE_LAND" in filename
- The script will automatically detect and use the first file of each type

## Output Files

The script generates the following output files in the specified output directory:

1. **filtered_elevation_data.csv/parquet**
   - Contains all data points with non-null precipitation and SWE values
   - Includes: point_id, lon, lat, time (if applicable), precipitation, SWE, elevation values

2. **elevation_statistics.csv/parquet**
   - Statistics grouped by elevation bins
   - Includes: mean, standard deviation, and count for both precipitation and SWE
   - Number of unique points per elevation bin

3. **filtering_summary.json**
   - Processing metadata
   - Number of points processed
   - Number of filtered records
   - Time range of data
   - Processing date

## How It Works

1. **Data Loading**: The script loads elevation points from a shapefile and identifies available CaSR files.

2. **Data Extraction**: For each elevation point, it finds the nearest grid cell in the CaSR data and extracts the time series.

3. **Filtering**: The extracted data is filtered to keep only records where both precipitation and SWE have non-null values.

4. **Merging**: Precipitation and SWE data are merged based on point location and time.

5. **Analysis**: The filtered data is analyzed by elevation bins to understand elevation-dependent patterns.

6. **Output**: Results are saved in the requested format(s) with comprehensive statistics.

## Performance Considerations

- **Sampling**: Use the `--sample-points` and `--sample-time` options to work with smaller datasets during testing
- **Memory usage**: Large datasets may require significant memory. Consider processing in batches if needed
- **Processing time**: Extraction time scales with the number of points and time steps

## Example Output

After running the script, you'll see a summary like:

```
============================================================
ELEVATION DATA FILTERING SUMMARY
============================================================
Elevation points loaded: 100
Filtered records (non-null precip & SWE): 850
Unique points with valid data: 85
Time range: 2020-01-01T00:00:00 to 2020-12-31T00:00:00

Elevation Statistics:
                     precip_mean  precip_std  precip_count  swe_mean  swe_std  swe_count  unique_points
elevation_bin                                                                                           
(500.0, 750.0]           12.5        5.2           120        45.3     15.2       120             12
(750.0, 1000.0]          15.8        6.1           180        62.1     18.5       180             18
(1000.0, 1250.0]         18.2        7.3           230        78.4     22.1       230             23
...
============================================================
```

## Troubleshooting

1. **No shapefile found**: Ensure the elevation directory contains at least one .shp file

2. **No CaSR files found**: Check that the CaSR directory contains NetCDF files with "A_PR24_SFC" and "P_SWE_LAND" in their names

3. **Coordinate system issues**: The script automatically handles coordinate transformations, but ensure your elevation data is in a standard CRS

4. **Memory errors**: Reduce the `--sample-points` and `--sample-time` parameters for large datasets

5. **No data extracted**: Check that the elevation points overlap with the CaSR data coverage area

## Integration with Other Scripts

This script is designed to work with the CaSR SWE file combination workflow:

1. First, use `combine_casr_swe_files.py` to combine CaSR files
2. Then, use `filter_merge_elevation_data.py` to extract and filter data at elevation points
3. Finally, use the filtered data for drought analysis or other applications

## Technical Details

### Coordinate Handling
- Supports both regular lat/lon and rotated coordinate systems
- Automatically converts between coordinate systems as needed
- Uses nearest neighbor interpolation for grid point selection

### Data Filtering
- Filters out NaN values in both precipitation and SWE
- Maintains temporal alignment between variables
- Preserves elevation metadata throughout the process

### Statistical Analysis
- Creates 10 elevation bins for analysis
- Calculates mean, standard deviation, and count for each bin
- Computes correlation between precipitation and SWE

## Related Files

- `combine_casr_swe_files.py`: Combines CaSR NetCDF files
- `extract_elevation_data_optimized.py`: Optimized elevation data extraction
- `notebooks/workflows/5_CaSR_SWE_file_combination_workflow.ipynb`: Jupyter notebook workflow

## License

This script is part of the Snow Drought Index project and follows the same license terms.
