# CaSR Data Analysis Notebook Usage Guide

## Overview

This guide explains how to use the Jupyter notebook (`CaSR_Data_Analysis_Notebook.ipynb`) for comprehensive analysis of Canadian Snow and Sea Ice Service Reanalysis (CaSR) precipitation data.

## Files Created

### Analysis Scripts
- **`analyze_casr_swe_data_improved.py`** - Enhanced analysis script with coordinate analysis
- **`CaSR_Data_Analysis_Notebook.ipynb`** - Interactive Jupyter notebook
- **`README_Notebook_Usage.md`** - This usage guide

### Generated Outputs
- **`casr_precipitation_analysis_report.txt`** - Comprehensive analysis report
- **`output_plots/`** - Directory containing 13 visualization plots
- **`CaSR_Data_Analysis_Summary.md`** - Summary documentation

## Prerequisites

### Required Python Libraries
```bash
pip install numpy pandas xarray matplotlib seaborn jupyter
```

### Data Requirements
- CaSR NetCDF file: `data/input_data/CaSR_SWE/CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1980-1983.nc`
- File should be ~8.79 MB containing 4 years of hourly precipitation data

## How to Use the Notebook

### 1. Launch Jupyter Notebook
```bash
jupyter notebook CaSR_Data_Analysis_Notebook.ipynb
```

### 2. Notebook Structure

The notebook is organized into 12 sections:

#### **Section 1: Setup and Imports**
- Imports required libraries
- Sets up plotting environment
- Imports the custom `CaSRDataAnalyzer` class

#### **Section 2: Initialize Data Analyzer**
- Checks file existence and size
- Initializes the analyzer object

#### **Section 3: Load and Examine Dataset**
- Loads the NetCDF dataset
- Displays basic dataset information

#### **Section 4: Temporal Analysis**
- Analyzes time coverage (1980-1983)
- Shows temporal distribution statistics

#### **Section 5: Spatial and Coordinate Analysis**
- Comprehensive coordinate system analysis
- Rotated pole grid mapping
- Grid properties and transformation metrics

#### **Section 6: Precipitation Data Analysis**
- Precipitation statistics and categorization
- Data coverage assessment

#### **Section 7: Interactive Data Exploration**
- Direct dataset access for custom analysis
- Coordinate range examination

#### **Section 8: Custom Visualization Examples**
- Interactive precipitation mapping
- Rotated vs geographic coordinate visualization

#### **Section 9: Time Series Analysis**
- Hourly, daily, and monthly aggregations
- Temporal pattern analysis

#### **Section 10: Coordinate System Visualization**
- Coordinate transformation visualization
- Grid structure analysis

#### **Section 11: Generate Complete Analysis Report**
- Creates all 13 visualization plots
- Exports comprehensive analysis report

#### **Section 12: Cleanup**
- Closes dataset and provides summary

## Key Features Demonstrated

### 1. **Comprehensive Data Analysis**
```python
# Initialize and load data
analyzer = CaSRDataAnalyzer(file_path)
analyzer.load_data()

# Perform all analyses
analyzer.analyze_temporal_coverage()
analyzer.analyze_spatial_coverage()  # Includes coordinate analysis
analyzer.analyze_precipitation_data()
```

### 2. **Coordinate System Analysis**
- **Rotated Pole Grid Mapping**: Analysis of grid transformation parameters
- **Coordinate Transformation**: Gradients and grid cell area calculations
- **Grid Properties**: Regularity assessment and spacing analysis
- **Quality Assessment**: Missing values, bounds checking, monotonicity

### 3. **Interactive Visualizations**
```python
# Custom precipitation mapping
time_slice = precip_data.isel(time=1000)
precip_mm = time_slice * 1000  # Convert to mm

# Plot on both coordinate systems
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# ... plotting code
```

### 4. **Time Series Analysis**
```python
# Multi-scale temporal analysis
spatial_mean = precip_data.mean(dim=['rlat', 'rlon'], skipna=True) * 1000
daily_precip = precip_series.resample('D').sum()
monthly_precip = precip_series.resample('M').sum()
```

## Analysis Results

### Coordinate Analysis Results
- **Grid Type**: Rotated latitude-longitude with pole at (31.758°N, 87.597°E)
- **Grid Resolution**: ~0.09° (~10 km) with slight longitude irregularities
- **Cell Areas**: 88.54 ± 1.14 km² (uniform distribution)
- **Coordinate Quality**: Excellent (no missing values, proper bounds)

### Precipitation Analysis Results
- **Data Coverage**: 4.2% of total grid points (1.79M valid measurements)
- **Mean Precipitation**: 2.41 mm (24-hour accumulation)
- **Maximum Precipitation**: 124.82 mm
- **Categories**: 22.3% no precip, 42.8% light, 28.2% moderate, 6.8% heavy

### Generated Visualizations (13 plots)
1. **Precipitation Analysis (8 plots)**:
   - Time series (hourly/daily)
   - Spatial maps and mean distribution
   - Statistical distributions
   - Data availability and coverage
   - Seasonal patterns

2. **Coordinate Analysis (5 plots)**:
   - Coordinate grids and transformations
   - Coordinate gradients
   - Grid cell areas
   - Grid spacing analysis
   - Coordinate system comparison

## Customization Options

### 1. **Modify Analysis Parameters**
```python
# Change output directory
analyzer.create_enhanced_visualizations(output_dir="custom_plots")

# Custom report filename
analyzer.export_comprehensive_report(output_file="custom_report.txt")
```

### 2. **Add Custom Analysis**
```python
# Access raw dataset for custom analysis
dataset = analyzer.dataset
precip_data = dataset['CaSR_v3.1_A_PR24_SFC']

# Custom calculations
custom_stats = precip_data.where(precip_data > 0.01).mean()
```

### 3. **Extend Visualizations**
```python
# Create custom plots using the loaded data
fig, ax = plt.subplots(figsize=(10, 8))
# ... custom plotting code
```

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure the NetCDF file exists in the correct path
   - Check file permissions

2. **Import Errors**
   - Install missing libraries: `pip install xarray netcdf4`
   - Ensure the analysis script is in the same directory

3. **Memory Issues**
   - The dataset is ~43M data points; ensure sufficient RAM
   - Consider analyzing subsets for very large datasets

4. **Plotting Issues**
   - Use `%matplotlib inline` for Jupyter notebooks
   - Ensure matplotlib backend is properly configured

### Performance Tips

1. **For Large Datasets**:
   ```python
   # Use chunking for very large files
   dataset = xr.open_dataset(file_path, chunks={'time': 1000})
   ```

2. **Memory Management**:
   ```python
   # Close dataset when done
   analyzer.close()
   ```

## Advanced Usage

### 1. **Batch Processing Multiple Files**
```python
file_list = [
    "CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1980-1983.nc",
    "CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_1984-1987.nc",
    # ... more files
]

for file_path in file_list:
    analyzer = CaSRDataAnalyzer(file_path)
    # ... analysis code
```

### 2. **Custom Analysis Functions**
```python
def analyze_extreme_events(precip_data, threshold=0.05):
    """Custom function to analyze extreme precipitation events"""
    extreme_events = precip_data.where(precip_data > threshold)
    return extreme_events.count(), extreme_events.mean()
```

## Output Files

### When you run the notebook, it generates:
- **`notebook_output_plots/`** - 13 visualization plots
- **`notebook_analysis_report.txt`** - Comprehensive analysis report
- **Console output** - Interactive analysis results and statistics

## Next Steps

1. **Explore Different Time Periods**: Modify the notebook to analyze different CaSR files
2. **Add Custom Analysis**: Extend the notebook with domain-specific analysis
3. **Compare Datasets**: Use the framework to compare different CaSR products
4. **Automate Processing**: Convert notebook cells to scripts for batch processing

## Support

For questions or issues:
1. Check the comprehensive analysis report for detailed results
2. Review the generated visualizations for data insights
3. Examine the console output for analysis statistics
4. Refer to the original analysis script for implementation details

The notebook provides a complete framework for CaSR data analysis and can be easily adapted for other NetCDF climate datasets.
