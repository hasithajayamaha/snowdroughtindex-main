SCS Analysis Notebook Migration
===========================

This guide provides detailed instructions for migrating code from the original ``SCS_analysis.ipynb`` notebook to the refactored package structure.

Original Notebook Overview
------------------------

The ``SCS_analysis.ipynb`` notebook contains code for:

1. Loading gap-filled SWE and precipitation data
2. Filtering data by basin using shapefiles and geopandas
3. Calculating daily and seasonal means for SWE and precipitation
4. Computing ratios and cumulative values
5. Performing spatial filtering and clustering (KMeans)
6. Statistical analysis including correlations and regressions
7. Visualizing spatial and temporal patterns

Equivalent Functionality in the Refactored Package
-----------------------------------------------

The functionality from ``SCS_analysis.ipynb`` has been distributed across several modules in the refactored package:

.. list-table::
   :header-rows: 1

   * - Original Functionality
     - New Module/Class
     - Function/Method
   * - Data loading and preprocessing
     - ``snowdroughtindex.core.dataset.SWEDataset``
     - ``__init__``, ``load_data``, ``preprocess``
   * - Basin filtering
     - ``snowdroughtindex.analysis.scs_analysis``
     - ``filter_by_basin``, ``load_basin_boundaries``
   * - Calculating means and ratios
     - ``snowdroughtindex.analysis.scs_analysis``
     - ``calculate_swe_precip_ratio``, ``calculate_cumulative_values``
   * - Spatial analysis
     - ``snowdroughtindex.analysis.scs_analysis``
     - ``perform_spatial_clustering``, ``analyze_elevation_bands``
   * - Statistical analysis
     - ``snowdroughtindex.utils.statistics``
     - ``calculate_correlation``, ``perform_regression``
   * - Visualization
     - ``snowdroughtindex.utils.visualization``
     - ``plot_spatial_distribution``, ``plot_timeseries_comparison``

Step-by-Step Migration Guide
-------------------------

### 1. Data Loading and Basin Filtering

Original code:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Load SWE data
    swe_ds = xr.open_dataset('data/swe_data.nc')
    
    # Load precipitation data
    precip_df = pd.read_csv('data/precipitation.csv')
    
    # Load basin boundaries
    basin_gdf = gpd.read_file('data/basin_boundaries.shp')
    
    # Create station points
    stations = pd.read_csv('data/stations.csv')
    geometry = [Point(xy) for xy in zip(stations['longitude'], stations['latitude'])]
    station_gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs="EPSG:4326")
    
    # Filter stations by basin
    basin_name = 'Colorado River Basin'
    basin_geom = basin_gdf[basin_gdf['name'] == basin_name].geometry.iloc[0]
    basin_stations = station_gdf[station_gdf.geometry.within(basin_geom)]
    
    # Filter SWE data by basin stations
    basin_swe = swe_ds.sel(station=basin_stations['station_id'].values)

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.analysis.scs_analysis import filter_by_basin, load_basin_boundaries
    
    # Load SWE data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Load precipitation data
    precip_dataset = SWEDataset('data/precipitation.csv', data_type='precipitation')
    
    # Load basin boundaries and filter stations
    basin_boundaries = load_basin_boundaries('data/basin_boundaries.shp')
    basin_name = 'Colorado River Basin'
    
    # Filter SWE data by basin
    basin_swe_dataset = filter_by_basin(swe_dataset, basin_boundaries, basin_name)
    basin_precip_dataset = filter_by_basin(precip_dataset, basin_boundaries, basin_name)

### 2. Calculating Means, Ratios, and Cumulative Values

Original code:

.. code-block:: python

    # Calculate daily mean SWE
    daily_swe = basin_swe.groupby('time.dayofyear').mean()
    
    # Calculate seasonal mean SWE
    seasonal_swe = basin_swe.sel(time=slice('2000-10-01', '2001-06-30')).mean(dim='time')
    
    # Calculate cumulative precipitation
    precip_df['cumulative'] = precip_df.groupby('station_id')['precipitation'].cumsum()
    
    # Calculate SWE to precipitation ratio
    # Assuming precip_df has been processed to match basin_swe stations and time periods
    swe_precip_ratio = seasonal_swe.values / precip_df.groupby('station_id')['cumulative'].max().values

Migrated code:

.. code-block:: python

    from snowdroughtindex.analysis.scs_analysis import calculate_swe_precip_ratio, calculate_cumulative_values
    
    # Calculate daily and seasonal means using SWEDataset methods
    daily_swe = basin_swe_dataset.calculate_daily_mean()
    seasonal_swe = basin_swe_dataset.calculate_seasonal_mean(start_date='10-01', end_date='06-30')
    
    # Calculate cumulative precipitation
    cumulative_precip = calculate_cumulative_values(basin_precip_dataset.data)
    
    # Calculate SWE to precipitation ratio
    swe_precip_ratio = calculate_swe_precip_ratio(basin_swe_dataset.data, basin_precip_dataset.data)

### 3. Spatial Analysis and Clustering

Original code:

.. code-block:: python

    from sklearn.cluster import KMeans
    
    # Extract station coordinates
    coords = basin_stations[['latitude', 'longitude', 'elevation']].values
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
    basin_stations['cluster'] = kmeans.labels_
    
    # Analyze by elevation bands
    elevation_bands = [
        (0, 1000),
        (1000, 2000),
        (2000, float('inf'))
    ]
    
    for min_elev, max_elev in elevation_bands:
        band_stations = basin_stations[(basin_stations['elevation'] >= min_elev) & 
                                      (basin_stations['elevation'] < max_elev)]
        band_swe = basin_swe.sel(station=band_stations['station_id'].values)
        # Analysis for this elevation band...

Migrated code:

.. code-block:: python

    from snowdroughtindex.analysis.scs_analysis import perform_spatial_clustering, analyze_elevation_bands
    
    # Perform spatial clustering
    clustered_stations = perform_spatial_clustering(basin_swe_dataset, n_clusters=3)
    
    # Analyze by elevation bands
    elevation_bands = [
        (0, 1000),
        (1000, 2000),
        (2000, float('inf'))
    ]
    
    elevation_analysis = analyze_elevation_bands(basin_swe_dataset, elevation_bands)

### 4. Statistical Analysis

Original code:

.. code-block:: python

    import numpy as np
    from scipy import stats
    
    # Calculate correlation between SWE and elevation
    elevations = basin_stations['elevation'].values
    max_swe = basin_swe.max(dim='time').values
    
    correlation, p_value = stats.pearsonr(elevations, max_swe)
    print(f"Correlation: {correlation:.2f}, p-value: {p_value:.4f}")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(elevations, max_swe)
    regression_line = slope * elevations + intercept

Migrated code:

.. code-block:: python

    from snowdroughtindex.utils.statistics import calculate_correlation, perform_regression
    
    # Calculate correlation between SWE and elevation
    correlation_result = calculate_correlation(
        basin_swe_dataset.get_station_metadata('elevation'),
        basin_swe_dataset.data.max(dim='time')
    )
    
    print(f"Correlation: {correlation_result['correlation']:.2f}, p-value: {correlation_result['p_value']:.4f}")
    
    # Perform linear regression
    regression_result = perform_regression(
        basin_swe_dataset.get_station_metadata('elevation'),
        basin_swe_dataset.data.max(dim='time')
    )
    
    regression_line = regression_result['slope'] * basin_swe_dataset.get_station_metadata('elevation') + regression_result['intercept']

### 5. Visualization

Original code:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Plot spatial distribution of SWE
    plt.figure(figsize=(10, 8))
    basin_gdf.plot(ax=plt.gca(), color='lightgray', edgecolor='black')
    scatter = plt.scatter(
        basin_stations['longitude'], 
        basin_stations['latitude'],
        c=seasonal_swe.values,
        cmap='Blues',
        s=50,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Seasonal Mean SWE (mm)')
    plt.title('Spatial Distribution of Seasonal Mean SWE')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
    # Plot SWE vs. elevation
    plt.figure(figsize=(10, 6))
    plt.scatter(elevations, max_swe, alpha=0.7)
    plt.plot(elevations, regression_line, 'r-')
    plt.title('Maximum SWE vs. Elevation')
    plt.xlabel('Elevation (m)')
    plt.ylabel('Maximum SWE (mm)')
    plt.text(
        0.05, 0.95, 
        f"r = {r_value:.2f}, p = {p_value:.4f}\ny = {slope:.2f}x + {intercept:.2f}", 
        transform=plt.gca().transAxes
    )
    plt.show()

Migrated code:

.. code-block:: python

    from snowdroughtindex.utils.visualization import plot_spatial_distribution, plot_regression

    # Plot spatial distribution of SWE
    plot_spatial_distribution(
        basin_swe_dataset,
        basin_boundaries,
        value_field='seasonal_mean',
        title='Spatial Distribution of Seasonal Mean SWE'
    )
    
    # Plot SWE vs. elevation with regression line
    plot_regression(
        basin_swe_dataset.get_station_metadata('elevation'),
        basin_swe_dataset.data.max(dim='time'),
        regression_result,
        xlabel='Elevation (m)',
        ylabel='Maximum SWE (mm)',
        title='Maximum SWE vs. Elevation'
    )

Complete Migration Example
-----------------------

Here's a complete example showing how to migrate a typical workflow from the original ``SCS_analysis.ipynb`` notebook to the refactored package:

Original workflow:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point
    from sklearn.cluster import KMeans
    from scipy import stats
    
    # Load data
    swe_ds = xr.open_dataset('data/swe_data.nc')
    precip_df = pd.read_csv('data/precipitation.csv')
    basin_gdf = gpd.read_file('data/basin_boundaries.shp')
    stations = pd.read_csv('data/stations.csv')
    
    # Create station points
    geometry = [Point(xy) for xy in zip(stations['longitude'], stations['latitude'])]
    station_gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs="EPSG:4326")
    
    # Filter stations by basin
    basin_name = 'Colorado River Basin'
    basin_geom = basin_gdf[basin_gdf['name'] == basin_name].geometry.iloc[0]
    basin_stations = station_gdf[station_gdf.geometry.within(basin_geom)]
    
    # Filter SWE data by basin stations
    basin_swe = swe_ds.sel(station=basin_stations['station_id'].values)
    
    # Calculate seasonal mean SWE
    seasonal_swe = basin_swe.sel(time=slice('2000-10-01', '2001-06-30')).mean(dim='time')
    
    # Analyze by elevation bands
    elevation_bands = [(0, 1000), (1000, 2000), (2000, float('inf'))]
    
    for min_elev, max_elev in elevation_bands:
        band_stations = basin_stations[(basin_stations['elevation'] >= min_elev) & 
                                      (basin_stations['elevation'] < max_elev)]
        band_swe = basin_swe.sel(station=band_stations['station_id'].values)
        print(f"Elevation band {min_elev}-{max_elev}m: {len(band_stations)} stations")
        print(f"Mean SWE: {band_swe.mean().values:.2f} mm")
    
    # Calculate correlation between SWE and elevation
    elevations = basin_stations['elevation'].values
    max_swe = basin_swe.max(dim='time').values
    
    correlation, p_value = stats.pearsonr(elevations, max_swe)
    print(f"Correlation: {correlation:.2f}, p-value: {p_value:.4f}")
    
    # Plot spatial distribution of SWE
    plt.figure(figsize=(10, 8))
    basin_gdf.plot(ax=plt.gca(), color='lightgray', edgecolor='black')
    scatter = plt.scatter(
        basin_stations['longitude'], 
        basin_stations['latitude'],
        c=seasonal_swe.values,
        cmap='Blues',
        s=50,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Seasonal Mean SWE (mm)')
    plt.title('Spatial Distribution of Seasonal Mean SWE')
    plt.show()

Migrated workflow:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.analysis.scs_analysis import (
        filter_by_basin, 
        load_basin_boundaries,
        analyze_elevation_bands
    )
    from snowdroughtindex.utils.statistics import calculate_correlation
    from snowdroughtindex.utils.visualization import plot_spatial_distribution
    
    # Load SWE data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Load basin boundaries and filter stations
    basin_boundaries = load_basin_boundaries('data/basin_boundaries.shp')
    basin_name = 'Colorado River Basin'
    
    # Filter SWE data by basin
    basin_swe_dataset = filter_by_basin(swe_dataset, basin_boundaries, basin_name)
    
    # Calculate seasonal mean
    seasonal_mean = basin_swe_dataset.calculate_seasonal_mean(start_date='10-01', end_date='06-30')
    
    # Analyze by elevation bands
    elevation_bands = [(0, 1000), (1000, 2000), (2000, float('inf'))]
    elevation_analysis = analyze_elevation_bands(basin_swe_dataset, elevation_bands)
    
    for band, analysis in elevation_analysis.items():
        print(f"Elevation band {band}: {analysis['station_count']} stations")
        print(f"Mean SWE: {analysis['mean_swe']:.2f} mm")
    
    # Calculate correlation between SWE and elevation
    correlation_result = calculate_correlation(
        basin_swe_dataset.get_station_metadata('elevation'),
        basin_swe_dataset.data.max(dim='time')
    )
    
    print(f"Correlation: {correlation_result['correlation']:.2f}, p-value: {correlation_result['p_value']:.4f}")
    
    # Plot spatial distribution of SWE
    plot_spatial_distribution(
        basin_swe_dataset,
        basin_boundaries,
        value_field='seasonal_mean',
        title='Spatial Distribution of Seasonal Mean SWE'
    )

Using the DroughtAnalysis Class
----------------------------

For more advanced analysis, you can use the ``DroughtAnalysis`` class, which provides methods for analyzing drought conditions across elevation bands and time periods:

.. code-block:: python

    from snowdroughtindex.core.drought_analysis import DroughtAnalysis
    
    # Create a DroughtAnalysis object
    drought_analysis = DroughtAnalysis(basin_swe_dataset)
    
    # Analyze drought conditions by elevation band
    elevation_drought = drought_analysis.analyze_by_elevation(
        elevation_bands=[(0, 1000), (1000, 2000), (2000, float('inf'))],
        start_date='10-01',
        end_date='06-30'
    )
    
    # Analyze drought conditions over time
    temporal_drought = drought_analysis.analyze_temporal_trends(
        start_year=1980,
        end_year=2020,
        window_size=10
    )
    
    # Visualize results
    drought_analysis.plot_elevation_analysis(elevation_drought)
    drought_analysis.plot_temporal_analysis(temporal_drought)

Configuration Options
------------------

The refactored package includes a configuration system that allows you to customize parameters for SCS analysis:

.. code-block:: python

    from snowdroughtindex.core.configuration import Configuration
    
    # Create a custom configuration
    config = Configuration()
    config.set_scs_analysis_params(
        clustering_variables=['latitude', 'longitude', 'elevation'],
        n_clusters=4,
        random_state=42
    )
    
    # Use the configuration with SCS analysis functions
    from snowdroughtindex.analysis.scs_analysis import perform_spatial_clustering
    
    clustered_stations = perform_spatial_clustering(
        basin_swe_dataset,
        config=config
    )

Advanced Usage
-----------

For advanced usage scenarios, such as custom statistical analyses or specialized visualization, refer to the API documentation:

- :doc:`/api/analysis`
- :doc:`/api/utils`

You can also check the example notebooks for more complex workflows:

- :doc:`/user_guide/workflows/scs_analysis`
