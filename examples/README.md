# Snow Drought Index Examples

This directory contains example scripts demonstrating various aspects of the Snow Drought Index package. The examples are organized into categories to help you find the most relevant scripts for your needs.

## Directory Structure

```
examples/
├── README.md                    # This file
├── basic/                       # Basic usage examples
├── data_analysis/              # Data analysis examples
├── data_processing/            # Data processing examples
├── advanced/                   # Advanced usage examples
└── utils/                      # Utility scripts
```

## Basic Examples (`basic/`)

These examples demonstrate fundamental usage of the package and basic setup:

- **`download_sample_data.py`** - Creates synthetic sample SWE and precipitation data for testing
- **`test_drought_indices_real_data.py`** - Demonstrates drought indices calculation with real data
- **`create_cdsapirc.py`** - Simple utility to create CDS API configuration file for data access

**Start here if you're new to the package!**

## Data Analysis Examples (`data_analysis/`)

These scripts show how to analyze various types of snow and climate data:

- **`analyze_casr_swe_data.py`** - Comprehensive analysis of Canadian Snow and Sea Ice Service Reanalysis (CaSR) SWE data
- **`analyze_casr_swe_data_improved.py`** - Enhanced version of the CaSR analysis with additional features
- **`analyze_elevation_data.py`** - Analysis of elevation data and its relationship to snow patterns
- **`detailed_elevation_analysis.py`** - In-depth elevation data analysis with visualization
- **`examine_coordinates.py`** - Script to examine and validate coordinate systems in datasets

## Data Processing Examples (`data_processing/`)

These examples demonstrate data preprocessing, combination, and optimization techniques:

- **`combine_casr_swe_files.py`** - Comprehensive script to combine NetCDF files from CaSR dataset across time and space
- **`extract_elevation_data_optimized.py`** - Optimized extraction of elevation data at specific point locations
- **`filter_merge_elevation_data.py`** - Filter and merge elevation data from multiple sources
- **`optimized_chunking_approach.py`** - Demonstrates optimized data chunking for large datasets

## Advanced Examples (`advanced/`)

These scripts showcase advanced features and optimization techniques:

- **`usage_example_optimized.py`** - Example of optimized artificial gap filling function with performance comparisons
- **`example_extraction.py`** - Advanced example of elevation data extraction using the ElevationDataExtractor class

## Utility Scripts (`utils/`)

*Currently empty - reserved for future utility examples*

## Getting Started

1. **First-time users**: Start with the `basic/` examples
2. **Data analysis**: Explore `data_analysis/` for analysis workflows
3. **Data preprocessing**: Check `data_processing/` for data preparation scripts
4. **Performance optimization**: Look at `advanced/` examples for optimized implementations

## Prerequisites

Before running the examples, ensure you have:

1. Installed the snowdroughtindex package:
   ```bash
   pip install -e .
   ```

2. Required dependencies (see `requirements.txt` in the project root)

3. Sample data (run `basic/download_sample_data.py` to create test data)

## Running Examples

Most examples can be run directly from the command line:

```bash
# Basic example
python examples/basic/download_sample_data.py

# Data analysis example
python examples/data_analysis/analyze_casr_swe_data.py

# Data processing example
python examples/data_processing/combine_casr_swe_files.py --help
```

## Data Requirements

Some examples require specific data files:

- **CaSR data examples**: Require NetCDF files from the Canadian Snow and Sea Ice Service Reanalysis
- **Elevation examples**: Require elevation point data and combined CaSR files
- **Real data examples**: May require actual meteorological datasets

Check individual script documentation for specific data requirements.

## Output

Examples typically create output in one or more of these locations:

- `output/` - General output directory
- `output_plots/` - Visualization outputs
- `sample_data/` - Generated sample datasets
- `data/output_data/` - Processed data outputs

## Contributing

When adding new examples:

1. Place them in the appropriate category directory
2. Include clear documentation and comments
3. Add a description to this README
4. Ensure examples are self-contained where possible
5. Include error handling and informative output

## Support

For questions about specific examples:

1. Check the script's internal documentation
2. Review the main package documentation
3. Open an issue on the project repository

## Example Categories Explained

### Basic vs Advanced
- **Basic**: Straightforward usage, minimal configuration, good for learning
- **Advanced**: Complex workflows, optimization techniques, production-ready code

### Data Analysis vs Data Processing
- **Data Analysis**: Focus on understanding and visualizing data
- **Data Processing**: Focus on transforming, combining, and preparing data

Choose the category that best matches your current needs!
