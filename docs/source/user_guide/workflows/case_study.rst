Case Study Workflow
==================

This guide explains the case study workflow for the Snow Drought Index package.

Purpose
-------

The case study workflow provides a comprehensive approach to analyzing snow drought conditions for specific case studies. It consolidates functionality from multiple original case study notebooks and demonstrates how to use the refactored Snow Drought Index package with its class-based implementation. This workflow involves:

1. Setting up configuration parameters
2. Loading and preprocessing SWE data
3. Calculating SSWEI values
4. Analyzing drought conditions
5. Visualizing results for specific case studies

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Prepared SWE data for your case study area
- Completed the :doc:`data preparation workflow <data_preparation>` if needed
- Completed the :doc:`gap filling workflow <gap_filling>` if your data has gaps

Workflow Steps
-------------

Step 1: Import Required Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, import the necessary libraries and modules:

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import geopandas as gpd
    from shapely.geometry import Point
    from scipy.integrate import trapz
    from scipy.stats import norm
    import warnings
    
    # Import class-based implementations
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.drought_analysis import DroughtAnalysis
    from snowdroughtindex.core.configuration import Configuration
    from snowdroughtindex.utils import visualization

Step 2: Configuration Setup
^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the configuration parameters for your case study:

.. code-block:: python

    # Create a configuration object with custom parameters
    config = Configuration(
        # Data parameters
        data_path='path/to/swe_data.nc',
        shapefile_path='path/to/basin_shapefile.shp',
        
        # Season parameters
        start_month=11,  # November
        start_day=1,
        end_month=4,     # April
        end_day=30,
        min_swe=15,      # Minimum SWE to consider as the start of the snow season
        
        # Gap filling parameters
        window_days=15,
        min_obs_corr=10,
        min_obs_cdf=5,
        min_corr=0.7,
        
        # Drought classification parameters
        drought_thresholds={
            "exceptional": -1.5,
            "extreme": -1.0,
            "severe": -0.5,
            "normal_lower": -0.5,
            "normal_upper": 0.5,
            "abnormally_wet": 0.5,
            "moderately_wet": 1.0,
            "very_wet": 1.5
        }
    )
    
    # Display configuration
    print(config)

Step 3: Load and Preprocess Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load and preprocess the SWE data using the SWEDataset class:

.. code-block:: python

    # Create a SWEDataset object
    swe_dataset = SWEDataset(config)
    
    # Load data
    swe_dataset.load_data()
    
    # Preprocess data
    swe_dataset.preprocess()
    
    # Fill gaps if needed
    swe_dataset.fill_gaps()
    
    # Display dataset information
    print(swe_dataset)
    
    # Plot daily mean SWE
    swe_dataset.plot_daily_mean_swe()

Step 4: Calculate SSWEI
^^^^^^^^^^^^^^^^^^^^^

Calculate the SSWEI values using the SSWEI class:

.. code-block:: python

    # Create an SSWEI object
    sswei_obj = SSWEI(swe_dataset, config)
    
    # Calculate SSWEI
    sswei_obj.calculate_sswei()
    
    # Display SSWEI results
    print(sswei_obj.sswei_results.head())
    
    # Plot SSWEI values
    sswei_obj.plot_sswei()

Step 5: Analyze Drought Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze drought conditions using the DroughtAnalysis class:

.. code-block:: python

    # Create a DroughtAnalysis object
    drought_analysis = DroughtAnalysis(sswei_obj, config)
    
    # Calculate drought characteristics
    drought_analysis.calculate_drought_characteristics()
    
    # Analyze drought trends
    drought_analysis.analyze_drought_trends(window_size=10)
    
    # Display drought characteristics
    print(drought_analysis.drought_characteristics)
    
    # Plot drought characteristics
    drought_analysis.plot_drought_characteristics()
    
    # Plot drought trends
    drought_analysis.plot_drought_trends()

Step 6: Case Study Specific Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform case study specific analysis, such as focusing on a particular drought event or region:

.. code-block:: python

    # Filter for a specific drought event
    drought_event = drought_analysis.drought_characteristics[
        drought_analysis.drought_characteristics['severity'] > 2.0
    ]
    
    # Get SSWEI values for the drought event years
    event_years = drought_event['start_year'].tolist() + drought_event['end_year'].tolist()
    event_sswei = sswei_obj.sswei_results[sswei_obj.sswei_results['season_year'].isin(event_years)]
    
    # Plot SSWEI values for the drought event
    plt.figure(figsize=(12, 6))
    plt.bar(event_sswei['season_year'], event_sswei['SWEI'])
    plt.axhline(y=-0.5, color='r', linestyle='--', label='Drought Threshold')
    plt.xlabel('Year')
    plt.ylabel('SSWEI')
    plt.title('SSWEI Values for Drought Event')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Analyze seasonal SWE patterns during the drought event
    for year in event_years:
        seasonal_data = swe_dataset.get_seasonal_data(year)
        plt.figure(figsize=(12, 6))
        plt.plot(seasonal_data['date'], seasonal_data['mean_SWE'], label=f'Year {year}')
        plt.xlabel('Date')
        plt.ylabel('Mean SWE (mm)')
        plt.title(f'Seasonal SWE Pattern for Year {year}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

Step 7: Compare with Other Drought Indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare SSWEI results with other drought indices if available:

.. code-block:: python

    # Load other drought indices (e.g., SPI, PDSI)
    other_indices = pd.read_csv('path/to/other_indices.csv')
    
    # Merge with SSWEI results
    comparison = pd.merge(
        sswei_obj.sswei_results[['season_year', 'SWEI', 'Drought_Classification']],
        other_indices,
        on='season_year'
    )
    
    # Calculate correlations
    correlation_matrix = comparison[['SWEI', 'SPI', 'PDSI']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(comparison['season_year'], comparison['SWEI'], label='SSWEI')
    plt.plot(comparison['season_year'], comparison['SPI'], label='SPI')
    plt.plot(comparison['season_year'], comparison['PDSI'], label='PDSI')
    plt.xlabel('Year')
    plt.ylabel('Index Value')
    plt.title('Comparison of Drought Indices')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

Step 8: Save Results
^^^^^^^^^^^^^^^^^

Save the results for future reference:

.. code-block:: python

    # Create output directory if it doesn't exist
    output_dir = 'path/to/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SSWEI results
    sswei_obj.sswei_results.to_csv(f'{output_dir}/sswei_results.csv', index=False)
    
    # Save drought characteristics
    drought_analysis.drought_characteristics.to_csv(
        f'{output_dir}/drought_characteristics.csv', 
        index=False
    )
    
    # Save drought trends
    drought_analysis.drought_trends.to_csv(
        f'{output_dir}/drought_trends.csv', 
        index=False
    )
    
    # Save configuration
    config.save(f'{output_dir}/config.yaml')

Key Classes
----------

The case study workflow uses the following key classes from the Snow Drought Index package:

- ``Configuration``: Manages configuration parameters for the analysis
- ``SWEDataset``: Handles loading, preprocessing, and gap filling of SWE data
- ``SSWEI``: Calculates SSWEI values and classifies drought conditions
- ``DroughtAnalysis``: Analyzes drought characteristics and trends

These classes provide a more object-oriented approach to snow drought analysis, making it easier to manage state and encapsulate related functionality.

Case Study Examples
-----------------

The Snow Drought Index package can be applied to various case studies, such as:

- **Regional Analysis**: Analyzing snow drought conditions in specific regions or watersheds
- **Historical Drought Events**: Investigating historical drought events and their characteristics
- **Climate Change Impact**: Assessing the impact of climate change on snow drought frequency and severity
- **Elevation-Based Analysis**: Comparing snow drought conditions across different elevation bands
- **Seasonal Variability**: Analyzing seasonal variability in snow accumulation and melt patterns

For each case study, you can customize the configuration parameters and analysis approach to suit your specific research questions.

Example Notebook
---------------

For a complete example of the case study workflow, refer to the 
`case_study_workflow.ipynb <https://github.com/yourusername/snowdroughtindex/blob/main/notebooks/workflows/case_study_workflow.ipynb>`_ 
notebook in the package repository.

Next Steps
---------

After completing the case study workflow, you can:

- Apply the workflow to different regions or time periods
- Extend the analysis with additional metrics or visualization techniques
- Integrate with other environmental data sources for more comprehensive analysis
- Develop custom analysis methods for specific research questions
