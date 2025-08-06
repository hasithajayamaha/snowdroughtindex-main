# Snow Drought Index

A Python package for analyzing and classifying snow drought conditions using various methodologies and indices.

## Overview

This package provides tools for analyzing snow water equivalent (SWE) data, calculating snow drought indices, and classifying drought conditions. It implements several methodologies including:

- Standardized Snow Water Equivalent Index (SSWEI) based on Huning & AghaKouchak methodology
- Snow drought classification according to Heldmyer et al. (2024)
- SWE/P ratio analysis for drought classification
- Gap filling of SWE data using quantile mapping
- DEM and shapefile integration for elevation-based analysis
- Comprehensive workflow comparison tools

## Features

- **Data Preprocessing**: 
  - Gap filling of SWE data using quantile mapping
  - Artificial gap filling evaluation
  - Linear interpolation for small data gaps
  - Data extraction and filtering based on geographical boundaries

- **Snow Drought Analysis**:
  - Calculation of Standardized Snow Water Equivalent Index (SSWEI)
  - Classification of drought conditions based on SSWEI values
  - SWE/P ratio analysis for drought classification
  - Elevation-based snow drought analysis

- **Visualization**:
  - Time series plots of SWE and drought indices
  - Spatial visualization of stations and basins
  - Drought classification visualization

## Installation

### Quick Setup (Recommended)

```bash
git clone https://github.com/username/snowdroughtindex.git
cd snowdroughtindex
python setup.py
```

This will automatically create a virtual environment, install all dependencies, and set up the package in development mode.

### Manual Installation

```bash
git clone https://github.com/username/snowdroughtindex.git
cd snowdroughtindex
pip install -e .
```

For detailed installation instructions, troubleshooting, and dependency information, see the [Installation Guide](docs/source/user_guide/installation.rst).

## Usage

### Data Preparation

```python
import xarray as xr
from notebooks.functions import extract_stations_in_basin, qm_gap_filling

# Load SWE data
SWE_data = xr.open_dataset('path/to/swe_data.nc')

# Extract stations within a basin
SWE_stations, basin_buffer = extract_stations_in_basin(stations, basins, basin_id, buffer_km=0)

# Perform gap filling
SWE_gapfilled, flags, donor_stations = qm_gap_filling(SWE_data, window_days=7, 
                                                     min_obs_corr=3, min_obs_cdf=10, 
                                                     min_corr=0.6)
```

### SSWEI Calculation

```python
import pandas as pd
from scipy.stats import norm
from scipy.integrate import trapz

# Integrate SWE over the season
def integrate_season(group):
    group = group.sort_values(by='date')
    days_since_start = (group['date'] - group['date'].min()).dt.days
    total_swe_integration = trapz(group['mean_SWE'], days_since_start)
    return pd.Series({'total_SWE_integration': total_swe_integration})

# Calculate Gringorten probabilities
def gringorten_probabilities(values):
    sorted_values = np.sort(values)
    ranks = np.argsort(np.argsort(values)) + 1
    n = len(values)
    probabilities = (ranks - 0.44) / (n + 0.12)
    return probabilities

# Compute SWEI
def compute_swei(probabilities):
    return norm.ppf(probabilities)

# Apply to seasonal data
Integrated_data = season_data.groupby('season_year').apply(integrate_season).reset_index()
Integrated_data['Gringorten_probabilities'] = gringorten_probabilities(Integrated_data['total_SWE_integration'])
Integrated_data['SWEI'] = compute_swei(Integrated_data['Gringorten_probabilities'])
```

### Drought Classification

```python
def classify_drought(swei):
    if swei <= -2.0:
        return "Exceptional Drought"
    elif -2.0 < swei <= -1.5:
        return "Extreme Drought"
    elif -1.5 < swei <= -1.0:
        return "Severe Drought"
    elif -1.0 < swei <= -0.5:
        return "Moderate Drought"
    elif -0.5 < swei <= 0.5:
        return "Near Normal"
    elif 0.5 < swei <= 1.0:
        return "Abnormally Wet"
    elif 1.0 < swei <= 1.5:
        return "Moderately Wet"
    elif 1.5 < swei <= 2.0:
        return "Very Wet"
    else:
        return "Extremely Wet"

Integrated_data['Drought_Classification'] = Integrated_data['SWEI'].apply(classify_drought)
```

## Workflows

The package includes several documented workflows to guide users through common analysis tasks:

1. **Data Preparation**: Loading and preprocessing SWE data
2. **Gap Filling**: Filling missing data in SWE time series
3. **SSWEI Calculation**: Computing the Standardized Snow Water Equivalent Index
4. **Drought Classification**: Classifying drought conditions based on SSWEI
5. **Heldmyer Classification**: Implementing the Heldmyer et al. (2024) classification methodology
6. **SCS Analysis**: Analyzing snow-to-precipitation ratios
7. **Case Studies**: Applying methodologies to specific regions
8. **DEM and Shapefile Integration**: Incorporating elevation data into analysis
9. **Workflow Comparison**: Comparing different analysis approaches

Detailed documentation for each workflow is available in the [documentation](docs/source/user_guide/workflows.rst).

## Case Studies

The package includes several case studies demonstrating the application of these methodologies:

1. **Bow River at Banff**: Analysis of snow drought conditions in the Bow River basin from 1980-2023
2. **Elevation-based Analysis**: Comparison of snow drought conditions at different elevation bands

## Functions

The package includes a comprehensive set of functions for data analysis, forecasting, and visualization. Key functions include:

- `artificial_gap_filling`: Creates random artificial gaps in the original dataset for each month & station, and runs the gap filling function to assess its performance
- `qm_gap_filling`: Performs gap filling for missing observations using quantile mapping
- `calculate_stations_doy_corr`: Calculates stations' correlations for each day of the year
- `circular_stats`: Calculates circular statistics for streamflow peaks
- `KGE_Tang2021`: Calculates the modified Kling-Gupta Efficiency (KGE") and its components
- `principal_component_analysis`: Transforms stations observations into principal components
- `regime_classification`: Performs regime classification using circular statistics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Walimunige Nadie Senali Rupasinghe

## Examples

The package includes a comprehensive set of example scripts organized by category in the `examples/` directory:

- **`examples/basic/`** - Basic usage examples and getting started scripts
- **`examples/data_analysis/`** - Data analysis workflows and visualization examples
- **`examples/data_processing/`** - Data preprocessing, combination, and optimization examples
- **`examples/advanced/`** - Advanced usage patterns and optimization techniques

See the [Examples README](examples/README.md) for detailed descriptions and usage instructions for each example script.

## Documentation

Comprehensive documentation is available at [docs/](docs/). The documentation includes:

- Installation instructions
- Quickstart guide
- Detailed workflow guides
- API reference
- Methodology explanations
- Examples and tutorials

## References

- Huning, L. S., & AghaKouchak, A. (2020). Global snow drought hot spots and characteristics. Proceedings of the National Academy of Sciences, 117(33), 19753-19759.
- Heldmyer, A. J., Livneh, B., Molotch, N. P., & Harpold, A. A. (2024). Classifying snow drought types: A new approach to understanding snow drought mechanisms. Journal of Hydrometeorology.
- Tang, G., Clark, M. P., & Papalexiou, S. M. (2021). SC-earth: A station-based serially complete earth dataset from 1950 to 2019. Journal of Climate, 34(16), 6493-6511.
