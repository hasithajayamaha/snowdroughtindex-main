SCS Analysis Workflow
===================

This guide explains the Snow Cover Seasonality (SCS) analysis workflow for the Snow Drought Index package.

Purpose
-------

The SCS analysis workflow provides a method for analyzing snow drought types based on the relationship between Snow Water Equivalent (SWE) and precipitation. It involves:

1. Loading SWE and precipitation data
2. Filtering data points within a basin
3. Calculating SWE/P ratios
4. Classifying snow drought types using K-means clustering
5. Visualizing the results

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Prepared SWE data in NetCDF format
- Prepared precipitation data in CSV format
- Prepared coordinates data in CSV format
- Prepared basin shapefile (optional, for spatial filtering)

Workflow Steps
-------------

Step 1: Load Data
^^^^^^^^^^^^^^^

First, load the SWE and precipitation data for analysis:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr
    import geopandas as gpd
    from shapely.geometry import Point
    import seaborn as sns
    
    from snowdroughtindex.analysis import scs_analysis
    from snowdroughtindex.core import data_preparation, dataset
    from snowdroughtindex.utils import visualization
    
    # Define data paths
    swe_path = 'path/to/swe_data.nc'
    precip_path = 'path/to/precip_data.csv'
    coordinates_path = 'path/to/coordinates_data.csv'
    shapefile_path = 'path/to/basin_shapefile.shp'
    
    # Load SWE data
    swe_data = xr.open_dataset(swe_path)
    
    # Load precipitation data
    precip_data = pd.read_csv(precip_path)
    
    # Load coordinates data
    coordinates_data = pd.read_csv(coordinates_path)

Step 2: Calculate Daily Mean SWE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the daily mean SWE for the basin:

.. code-block:: python

    # Calculate daily mean SWE
    daily_mean_swe = scs_analysis.calculate_daily_mean_swe(swe_data)
    
    # Plot the daily mean SWE
    plt.figure(figsize=(12, 6))
    plt.plot(daily_mean_swe['Date'], daily_mean_swe['mean_SWE'])
    plt.xlabel('Date')
    plt.ylabel('Mean SWE (mm)')
    plt.title('Daily Mean SWE')
    plt.grid(True, alpha=0.3)
    plt.show()

Step 3: Filter Data Points Within Basin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter the precipitation data points that fall within the basin of interest:

.. code-block:: python

    # Filter points within the shapefile
    points_within = scs_analysis.filter_points_within_shapefile(
        coordinates_data,
        shapefile_path,
        station_name=None  # Update with your station name if needed
    )
    
    # Plot the basin and points
    shapefile = gpd.read_file(shapefile_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    shapefile.plot(ax=ax, color='lightblue')
    points_within.plot(ax=ax, color='red', marker='o', markersize=50)
    plt.title("Basin and Precipitation Data Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
    # Get the list of station IDs within the basin
    station_ids = points_within['subid'].astype(str).tolist()

Step 4: Calculate Basin Mean Precipitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the mean precipitation across the selected stations within the basin:

.. code-block:: python

    # Calculate basin mean precipitation
    mean_precip = scs_analysis.calculate_basin_mean_precipitation(precip_data, station_ids)
    
    # Plot the basin mean precipitation
    plt.figure(figsize=(12, 6))
    plt.plot(mean_precip['Date'], mean_precip['mean_precip'])
    plt.xlabel('Date')
    plt.ylabel('Mean Precipitation (mm)')
    plt.title('Basin Mean Precipitation')
    plt.grid(True, alpha=0.3)
    plt.show()

Step 5: Merge SWE and Precipitation Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Merge the SWE and precipitation data on common dates:

.. code-block:: python

    # Merge SWE and precipitation data
    merged_data = scs_analysis.merge_swe_precip_data(daily_mean_swe, mean_precip)
    
    # Plot the merged data
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot SWE on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mean SWE (mm)', color=color)
    ax1.plot(merged_data['Date'], merged_data['mean_SWE'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for precipitation
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Precipitation (mm)', color=color)
    ax2.plot(merged_data['Date'], merged_data['mean_precip'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('SWE and Precipitation Time Series')
    fig.tight_layout()
    plt.show()

Step 6: Filter for Snow Season
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter the data for the snow season (November to May by default):

.. code-block:: python

    # Filter for snow season
    snow_season_data = scs_analysis.filter_snow_season(
        merged_data,
        start_month=11,
        start_day=1,
        end_month=5,
        end_day=1
    )
    
    # Calculate seasonal means
    seasonal_means = scs_analysis.calculate_seasonal_means(snow_season_data)

Step 7: Filter for Complete Snow Seasons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter for complete snow seasons based on a SWE threshold:

.. code-block:: python

    # Filter for complete snow seasons
    complete_seasons = scs_analysis.filter_complete_seasons(
        merged_data,
        swe_threshold=15,  # Adjust based on your data
        start_month=11,
        start_day=1,
        end_month=5,
        end_day=1
    )
    
    # Calculate SWE/P ratio and cumulative precipitation
    complete_seasons = scs_analysis.calculate_swe_p_ratio(complete_seasons)

Step 8: Calculate Seasonal Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate seasonal metrics including max SWE, mean SWE/P ratio, and mean cumulative precipitation:

.. code-block:: python

    # Calculate seasonal metrics
    seasonal_metrics = scs_analysis.calculate_seasonal_metrics(complete_seasons)
    
    # Plot seasonal mean SWE vs precipitation
    fig = scs_analysis.plot_seasonal_swe_precip(seasonal_metrics)
    plt.show()

Step 9: Standardize Metrics for Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standardize the metrics for clustering and filter out any outliers:

.. code-block:: python

    # Standardize metrics for clustering
    standardized_metrics = scs_analysis.standardize_metrics(seasonal_metrics, ratio_max=1.0)

Step 10: Classify Snow Drought Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classify snow drought types using K-means clustering:

.. code-block:: python

    # Classify snow drought types
    classified_data, cluster_centers, cluster_labels = scs_analysis.classify_snow_drought(
        standardized_metrics,
        n_clusters=3,
        random_state=0
    )
    
    # Define cluster colors
    cluster_colors = {
        'Warm': 'red',
        'Dry': 'blue',
        'Warm & Dry': 'purple',
        'Normal': 'grey'
    }
    
    # Plot snow drought classification
    fig = scs_analysis.plot_snow_drought_classification(classified_data, cluster_colors)
    plt.show()

Step 11: Calculate and Visualize Anomalies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate and visualize various anomalies to better understand the snow drought patterns:

.. code-block:: python

    # Calculate ratio anomaly
    classified_data['ratio_anomaly'] = classified_data['SWE_P_ratio'] - classified_data['SWE_P_ratio'].mean()
    
    # Calculate peak SWE anomaly
    classified_data['peak_SWE_anomaly'] = classified_data['SWEmax'] - classified_data['SWEmax'].mean()
    
    # Plot cumulative precipitation anomaly
    fig1 = scs_analysis.plot_drought_time_series(
        classified_data,
        'cum_P_anom',
        cluster_colors=cluster_colors
    )
    plt.title('Cumulative Precipitation Anomaly')
    plt.show()
    
    # Plot SWE/P ratio anomaly
    fig2 = scs_analysis.plot_drought_time_series(
        classified_data,
        'ratio_anomaly',
        cluster_colors=cluster_colors
    )
    plt.title('SWE/P Ratio Anomaly')
    plt.show()
    
    # Plot peak SWE anomaly
    fig3 = scs_analysis.plot_drought_time_series(
        classified_data,
        'peak_SWE_anomaly',
        cluster_colors=cluster_colors
    )
    plt.title('Peak SWE Anomaly')
    plt.show()

Step 12: Run Complete SCS Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can run the complete SCS analysis workflow using the `run_scs_analysis` function:

.. code-block:: python

    # Run complete SCS analysis
    results = scs_analysis.run_scs_analysis(
        daily_mean_swe,
        precip_data,
        station_ids,
        swe_threshold=15,
        n_clusters=3
    )

Step 13: Save Results
^^^^^^^^^^^^^^^^^^^

Save the results for future reference:

.. code-block:: python

    # Create output directory if it doesn't exist
    import os
    output_dir = 'path/to/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classified data
    classified_data.to_csv(f'{output_dir}/scs_classified_data.csv', index=False)
    
    # Save seasonal metrics
    seasonal_metrics.to_csv(f'{output_dir}/scs_seasonal_metrics.csv', index=False)

Key Functions
------------

The SCS analysis workflow uses the following key functions from the ``scs_analysis`` module:

- ``calculate_daily_mean_swe()`` for calculating daily mean SWE for a basin
- ``filter_points_within_shapefile()`` for filtering data points within a basin boundary
- ``calculate_basin_mean_precipitation()`` for calculating mean precipitation across selected stations
- ``merge_swe_precip_data()`` for merging SWE and precipitation data
- ``filter_snow_season()`` for filtering data for the snow season
- ``filter_complete_seasons()`` for filtering for complete snow seasons
- ``calculate_swe_p_ratio()`` for calculating SWE/P ratio and cumulative precipitation
- ``calculate_seasonal_metrics()`` for calculating seasonal metrics
- ``standardize_metrics()`` for standardizing metrics for clustering
- ``classify_snow_drought()`` for classifying snow drought types
- ``plot_snow_drought_classification()`` for visualizing snow drought classification
- ``plot_drought_time_series()`` for visualizing drought time series
- ``run_scs_analysis()`` for running the complete SCS analysis workflow

Snow Drought Types
----------------

The SCS analysis workflow classifies snow drought types based on the relationship between SWE/P ratio and cumulative precipitation:

- **Warm Drought**: Low SWE/P ratio with normal or above-normal precipitation. This indicates that precipitation is falling as rain rather than snow due to warm temperatures.
- **Dry Drought**: Low SWE with low precipitation. This indicates a lack of precipitation overall.
- **Warm & Dry Drought**: Low SWE/P ratio with low precipitation. This indicates both warm temperatures and a lack of precipitation.
- **Normal**: Normal SWE/P ratio and normal precipitation. This indicates no drought conditions.

These classifications help identify the underlying causes of snow drought conditions, which can inform water resource management and climate adaptation strategies.

Example Notebook
---------------

For a complete example of the SCS analysis workflow, refer to the 
`scs_analysis_workflow.ipynb <https://github.com/yourusername/snowdroughtindex/blob/main/notebooks/workflows/scs_analysis_workflow.ipynb>`_ 
notebook in the package repository.

Next Steps
---------

After completing the SCS analysis workflow, you can proceed to:

- :doc:`Case study workflow <case_study>` for applying the SCS analysis to specific case studies
- Combine SCS analysis with SSWEI results to gain a more comprehensive understanding of snow drought conditions
