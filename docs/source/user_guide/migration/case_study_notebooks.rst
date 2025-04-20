Case Study Notebooks Migration
==========================

This guide provides detailed instructions for migrating code from the original case study notebooks (``case_study.ipynb``, ``case_study_classification.ipynb``, ``case_study_SSWEI.ipynb``, and ``CaSR_Land_case_study.ipynb``) to the refactored package structure.

Original Notebooks Overview
------------------------

The case study notebooks contain code for:

1. Loading and preprocessing SWE data for specific regions or time periods
2. Calculating SSWEI for case studies
3. Classifying drought conditions for specific events
4. Analyzing drought patterns in different regions
5. Visualizing case study results
6. Comparing drought conditions across different regions and time periods

Equivalent Functionality in the Refactored Package
-----------------------------------------------

The functionality from the case study notebooks has been distributed across several modules in the refactored package:

.. list-table::
   :header-rows: 1

   * - Original Functionality
     - New Module/Class
     - Function/Method
   * - Data loading and preprocessing
     - ``snowdroughtindex.core.dataset.SWEDataset``
     - ``__init__``, ``load_data``, ``preprocess``
   * - Case study data extraction
     - ``snowdroughtindex.analysis.case_studies``
     - ``load_case_study``, ``extract_region``
   * - SSWEI calculation
     - ``snowdroughtindex.core.sswei_class.SSWEI``
     - ``calculate``
   * - Drought classification
     - ``snowdroughtindex.core.drought_classification``
     - ``classify_drought``
   * - Case study analysis
     - ``snowdroughtindex.analysis.case_studies``
     - ``analyze_drought_event``, ``compare_regions``
   * - Visualization
     - ``snowdroughtindex.utils.visualization``
     - ``plot_case_study``, ``plot_drought_comparison``

Step-by-Step Migration Guide
-------------------------

### 1. Data Loading and Region Extraction

Original code:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    import numpy as np
    
    # Load data
    ds = xr.open_dataset('data/swe_data.nc')
    
    # Extract region of interest
    region_name = 'Sierra Nevada'
    stations = pd.read_csv('data/stations.csv')
    region_stations = stations[stations['region'] == region_name]
    region_ds = ds.sel(station=region_stations['station_id'].values)
    
    # Extract time period of interest
    drought_year = 2015
    start_date = f"{drought_year-1}-10-01"
    end_date = f"{drought_year}-06-30"
    drought_ds = region_ds.sel(time=slice(start_date, end_date))

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.analysis.case_studies import load_case_study, extract_region
    
    # Load data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Extract region of interest
    region_name = 'Sierra Nevada'
    region_dataset = extract_region(swe_dataset, region_name)
    
    # Load specific case study
    drought_year = 2015
    case_study_dataset = load_case_study(
        region_dataset,
        year=drought_year,
        start_month=10,
        start_day=1,
        end_month=6,
        end_day=30
    )

### 2. SSWEI Calculation for Case Studies

Original code:

.. code-block:: python

    # Calculate SSWEI for the case study
    def integrate_season(ds, start_date='10-01', end_date='06-30'):
        # Implementation details...
        return integrated_swe
    
    def calculate_sswei(integrated_swe):
        # Implementation details...
        return sswei
    
    # Calculate for the drought year
    integrated_swe = integrate_season(drought_ds)
    drought_sswei = calculate_sswei(integrated_swe)
    
    # Calculate for the reference period
    reference_period = ds.sel(time=slice('1981-10-01', '2010-06-30'))
    reference_integrated = integrate_season(reference_period)
    reference_sswei = calculate_sswei(reference_integrated)

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Calculate SSWEI for the case study using SSWEI class
    sswei_obj = SSWEI(case_study_dataset)
    sswei_obj.calculate(start_date='10-01', end_date='06-30')
    drought_sswei = sswei_obj.sswei
    
    # Calculate SSWEI for the reference period
    reference_dataset = SWEDataset('data/swe_data.nc')
    reference_dataset = extract_region(reference_dataset, region_name)
    reference_dataset.filter_time(start_date='1981-10-01', end_date='2010-06-30')
    
    reference_sswei_obj = SSWEI(reference_dataset)
    reference_sswei_obj.calculate(start_date='10-01', end_date='06-30')
    reference_sswei = reference_sswei_obj.sswei

### 3. Drought Classification for Case Studies

Original code:

.. code-block:: python

    # Classify drought conditions
    def classify_drought(sswei, thresholds=None):
        # Implementation details...
        return drought_class
    
    drought_class = classify_drought(drought_sswei)
    
    # Count drought categories
    categories = ['Extreme Dry', 'Dry', 'Normal', 'Wet', 'Extreme Wet']
    category_counts = {cat: np.sum(drought_class == i) for i, cat in enumerate(categories)}
    
    # Calculate percentage of stations in drought
    drought_percentage = (category_counts['Extreme Dry'] + category_counts['Dry']) / len(drought_class) * 100
    print(f"Percentage of stations in drought: {drought_percentage:.1f}%")

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.drought_classification import classify_drought, calculate_drought_statistics
    
    # Using SSWEI class for classification
    drought_class = sswei_obj.classify_drought()
    
    # Or using the function directly
    drought_class = classify_drought(drought_sswei)
    
    # Calculate drought statistics
    drought_stats = calculate_drought_statistics(drought_class)
    
    print(f"Percentage of stations in drought: {drought_stats['drought_percentage']:.1f}%")
    print(f"Category counts: {drought_stats['category_counts']}")

### 4. Case Study Analysis

Original code:

.. code-block:: python

    # Analyze drought patterns by elevation
    stations['elevation_band'] = pd.cut(
        stations['elevation'],
        bins=[0, 1000, 2000, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    for band in ['Low', 'Medium', 'High']:
        band_stations = stations[stations['elevation_band'] == band]
        band_drought = drought_class.sel(station=band_stations['station_id'].values)
        band_percentage = np.sum(band_drought <= 1) / len(band_drought) * 100
        print(f"{band} elevation band drought percentage: {band_percentage:.1f}%")
    
    # Compare with historical droughts
    historical_droughts = [1977, 1988, 2002, 2012]
    for year in historical_droughts:
        start_date = f"{year-1}-10-01"
        end_date = f"{year}-06-30"
        hist_ds = region_ds.sel(time=slice(start_date, end_date))
        hist_integrated = integrate_season(hist_ds)
        hist_sswei = calculate_sswei(hist_integrated)
        hist_class = classify_drought(hist_sswei)
        hist_percentage = np.sum(hist_class <= 1) / len(hist_class) * 100
        print(f"{year} drought percentage: {hist_percentage:.1f}%")

Migrated code:

.. code-block:: python

    from snowdroughtindex.analysis.case_studies import analyze_drought_event, compare_historical_droughts
    from snowdroughtindex.core.drought_analysis import DroughtAnalysis
    
    # Analyze drought patterns by elevation using DroughtAnalysis class
    drought_analysis = DroughtAnalysis(case_study_dataset)
    elevation_analysis = drought_analysis.analyze_by_elevation(
        elevation_bands=[(0, 1000), (1000, 2000), (2000, float('inf'))],
        band_labels=['Low', 'Medium', 'High']
    )
    
    for band, stats in elevation_analysis.items():
        print(f"{band} elevation band drought percentage: {stats['drought_percentage']:.1f}%")
    
    # Compare with historical droughts
    historical_droughts = [1977, 1988, 2002, 2012]
    historical_comparison = compare_historical_droughts(
        region_dataset,
        drought_year,
        historical_droughts,
        start_month=10,
        start_day=1,
        end_month=6,
        end_day=30
    )
    
    for year, stats in historical_comparison.items():
        print(f"{year} drought percentage: {stats['drought_percentage']:.1f}%")

### 5. Visualization of Case Studies

Original code:

.. code-block:: python

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Plot spatial distribution of drought classification
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    
    scatter = plt.scatter(
        region_stations['longitude'],
        region_stations['latitude'],
        c=drought_class.values,
        cmap='RdYlBu',
        vmin=0,
        vmax=4,
        transform=ccrs.PlateCarree()
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Extreme Dry', 'Dry', 'Normal', 'Wet', 'Extreme Wet'])
    
    plt.title(f'Drought Classification for {region_name} ({drought_year})')
    plt.show()
    
    # Plot time series of SWE for the drought year
    plt.figure(figsize=(12, 6))
    drought_ds.mean(dim='station').plot(label=f'{drought_year}')
    
    # Add climatology for comparison
    climatology = ds.sel(time=slice('1981-10-01', '2010-06-30'))
    climatology = climatology.groupby('time.dayofyear').mean()
    climatology.plot(label='1981-2010 Mean')
    
    plt.title(f'SWE Time Series for {region_name}')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.legend()
    plt.show()

Migrated code:

.. code-block:: python

    from snowdroughtindex.utils.visualization import (
        plot_drought_classification_map,
        plot_swe_comparison,
        plot_drought_severity
    )
    
    # Plot spatial distribution of drought classification
    plot_drought_classification_map(
        case_study_dataset,
        drought_class,
        title=f'Drought Classification for {region_name} ({drought_year})'
    )
    
    # Plot time series of SWE for the drought year compared to climatology
    plot_swe_comparison(
        case_study_dataset,
        reference_dataset,
        title=f'SWE Time Series for {region_name}',
        labels=[f'{drought_year}', '1981-2010 Mean']
    )
    
    # Plot drought severity
    plot_drought_severity(
        drought_class,
        title=f'Drought Severity for {region_name} ({drought_year})'
    )

Complete Migration Example
-----------------------

Here's a complete example showing how to migrate a typical workflow from the original case study notebooks to the refactored package:

Original workflow:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data
    ds = xr.open_dataset('data/swe_data.nc')
    stations = pd.read_csv('data/stations.csv')
    
    # Extract region and time period
    region_name = 'Sierra Nevada'
    region_stations = stations[stations['region'] == region_name]
    region_ds = ds.sel(station=region_stations['station_id'].values)
    
    drought_year = 2015
    start_date = f"{drought_year-1}-10-01"
    end_date = f"{drought_year}-06-30"
    drought_ds = region_ds.sel(time=slice(start_date, end_date))
    
    # Calculate SSWEI
    def integrate_season(ds, start_date='10-01', end_date='06-30'):
        # Implementation details...
        return integrated_swe
    
    def calculate_sswei(integrated_swe):
        # Implementation details...
        return sswei
    
    integrated_swe = integrate_season(drought_ds)
    drought_sswei = calculate_sswei(integrated_swe)
    
    # Classify drought
    def classify_drought(sswei, thresholds=None):
        # Implementation details...
        return drought_class
    
    drought_class = classify_drought(drought_sswei)
    
    # Calculate drought statistics
    categories = ['Extreme Dry', 'Dry', 'Normal', 'Wet', 'Extreme Wet']
    category_counts = {cat: np.sum(drought_class == i) for i, cat in enumerate(categories)}
    drought_percentage = (category_counts['Extreme Dry'] + category_counts['Dry']) / len(drought_class) * 100
    
    print(f"Drought Year: {drought_year}")
    print(f"Region: {region_name}")
    print(f"Percentage of stations in drought: {drought_percentage:.1f}%")
    print("Category counts:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} ({count/len(drought_class)*100:.1f}%)")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.scatter(
        region_stations['longitude'],
        region_stations['latitude'],
        c=drought_class.values,
        cmap='RdYlBu',
        vmin=0,
        vmax=4
    )
    plt.colorbar(label='Drought Classification')
    plt.title(f'Drought Classification for {region_name} ({drought_year})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

Migrated workflow:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.analysis.case_studies import load_case_study, extract_region
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.drought_classification import calculate_drought_statistics
    from snowdroughtindex.utils.visualization import plot_drought_classification_map
    
    # Load data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Extract region of interest
    region_name = 'Sierra Nevada'
    region_dataset = extract_region(swe_dataset, region_name)
    
    # Load specific case study
    drought_year = 2015
    case_study_dataset = load_case_study(
        region_dataset,
        year=drought_year,
        start_month=10,
        start_day=1,
        end_month=6,
        end_day=30
    )
    
    # Calculate SSWEI using SSWEI class
    sswei_obj = SSWEI(case_study_dataset)
    sswei_obj.calculate(start_date='10-01', end_date='06-30')
    
    # Classify drought
    drought_class = sswei_obj.classify_drought()
    
    # Calculate drought statistics
    drought_stats = calculate_drought_statistics(drought_class)
    
    print(f"Drought Year: {drought_year}")
    print(f"Region: {region_name}")
    print(f"Percentage of stations in drought: {drought_stats['drought_percentage']:.1f}%")
    print("Category counts:")
    for cat, count in drought_stats['category_counts'].items():
        print(f"  {cat}: {count} ({count/len(drought_class)*100:.1f}%)")
    
    # Plot results
    plot_drought_classification_map(
        case_study_dataset,
        drought_class,
        title=f'Drought Classification for {region_name} ({drought_year})'
    )

Using the DroughtAnalysis Class for Case Studies
--------------------------------------------

For more advanced case study analysis, you can use the ``DroughtAnalysis`` class:

.. code-block:: python

    from snowdroughtindex.core.drought_analysis import DroughtAnalysis
    
    # Create a DroughtAnalysis object
    drought_analysis = DroughtAnalysis(case_study_dataset)
    
    # Analyze drought conditions by elevation band
    elevation_analysis = drought_analysis.analyze_by_elevation(
        elevation_bands=[(0, 1000), (1000, 2000), (2000, float('inf'))],
        band_labels=['Low', 'Medium', 'High']
    )
    
    # Analyze drought conditions by region
    region_analysis = drought_analysis.analyze_by_region(
        regions=['Sierra Nevada', 'Rocky Mountains', 'Cascades']
    )
    
    # Analyze drought conditions over time
    temporal_analysis = drought_analysis.analyze_temporal_trends(
        start_year=1980,
        end_year=2020,
        window_size=10
    )
    
    # Visualize results
    drought_analysis.plot_elevation_analysis(elevation_analysis)
    drought_analysis.plot_region_analysis(region_analysis)
    drought_analysis.plot_temporal_analysis(temporal_analysis)

Comparing Multiple Case Studies
----------------------------

To compare multiple case studies, you can use the ``compare_case_studies`` function:

.. code-block:: python

    from snowdroughtindex.analysis.case_studies import compare_case_studies
    
    # Define case studies to compare
    case_studies = [
        {'year': 2015, 'region': 'Sierra Nevada'},
        {'year': 2012, 'region': 'Sierra Nevada'},
        {'year': 2015, 'region': 'Rocky Mountains'},
        {'year': 2012, 'region': 'Rocky Mountains'}
    ]
    
    # Compare case studies
    comparison = compare_case_studies(
        swe_dataset,
        case_studies,
        start_month=10,
        start_day=1,
        end_month=6,
        end_day=30
    )
    
    # Print comparison results
    for case, stats in comparison.items():
        print(f"Case Study: {case}")
        print(f"  Drought Percentage: {stats['drought_percentage']:.1f}%")
        print(f"  Mean SSWEI: {stats['mean_sswei']:.2f}")
    
    # Visualize comparison
    from snowdroughtindex.utils.visualization import plot_case_study_comparison
    
    plot_case_study_comparison(comparison)

Configuration Options
------------------

The refactored package includes a configuration system that allows you to customize parameters for case study analysis:

.. code-block:: python

    from snowdroughtindex.core.configuration import Configuration
    
    # Create a custom configuration
    config = Configuration()
    config.set_case_study_params(
        reference_period=(1981, 2010),
        drought_thresholds={
            'extreme_wet': 1.5,
            'wet': 0.5,
            'normal': -0.5,
            'dry': -1.5,
            'extreme_dry': float('-inf')
        }
    )
    
    # Use the configuration with SSWEI class
    sswei_obj = SSWEI(case_study_dataset, config=config)
    sswei_obj.calculate()
    drought_class = sswei_obj.classify_drought()

Advanced Usage
-----------

For advanced usage scenarios, such as custom case study analysis or specialized visualization, refer to the API documentation:

- :doc:`/api/analysis`
- :doc:`/api/core`
- :doc:`/api/utils`

You can also check the example notebooks for more complex workflows:

- :doc:`/user_guide/workflows/case_study`
