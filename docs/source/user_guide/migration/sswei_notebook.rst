SSWEI Notebook Migration
=====================

This guide provides detailed instructions for migrating code from the original ``SSWEI.ipynb`` notebook to the refactored package structure.

Original Notebook Overview
------------------------

The ``SSWEI.ipynb`` notebook contains code for:

1. Loading and preprocessing SWE data from NetCDF files using xarray and pandas
2. Calculating daily and seasonal means
3. Handling zero values and integrating SWE over the season
4. Computing Gringorten probabilities
5. Transforming to SSWEI using the normal distribution
6. Classifying drought conditions
7. Visualizing SSWEI trends and drought classification thresholds

Equivalent Functionality in the Refactored Package
-----------------------------------------------

The functionality from ``SSWEI.ipynb`` has been distributed across several modules in the refactored package:

.. list-table::
   :header-rows: 1

   * - Original Functionality
     - New Module/Class
     - Function/Method
   * - Data loading and preprocessing
     - ``snowdroughtindex.core.dataset.SWEDataset``
     - ``__init__``, ``load_data``, ``preprocess``
   * - Calculating daily/seasonal means
     - ``snowdroughtindex.core.data_preparation``
     - ``calculate_daily_mean``, ``calculate_seasonal_mean``
   * - SWE integration
     - ``snowdroughtindex.core.sswei``
     - ``integrate_season``
   * - Probability transformation
     - ``snowdroughtindex.core.sswei``
     - ``calculate_gringorten_probability``, ``transform_to_normal``
   * - SSWEI calculation
     - ``snowdroughtindex.core.sswei_class.SSWEI``
     - ``calculate``
   * - Drought classification
     - ``snowdroughtindex.core.drought_classification``
     - ``classify_drought``
   * - Visualization
     - ``snowdroughtindex.utils.visualization``
     - ``plot_sswei_timeseries``, ``plot_drought_classification``

Step-by-Step Migration Guide
-------------------------

### 1. Data Loading and Preprocessing

Original code:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    import numpy as np
    
    # Load data
    ds = xr.open_dataset('data/swe_data.nc')
    
    # Filter stations if needed
    stations = pd.read_csv('data/stations.csv')
    filtered_stations = stations[stations['elevation'] > 1000]
    ds = ds.sel(station=filtered_stations['station_id'].values)

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    
    # Load data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Filter stations if needed
    swe_dataset.filter_stations(elevation_min=1000)

### 2. Calculating Daily and Seasonal Means

Original code:

.. code-block:: python

    # Calculate daily mean
    daily_mean = ds.groupby('time.dayofyear').mean()
    
    # Calculate seasonal mean
    seasonal_mean = ds.sel(time=slice('2000-10-01', '2001-06-30')).mean(dim='time')

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.data_preparation import calculate_daily_mean, calculate_seasonal_mean
    
    # Using functions
    daily_mean = calculate_daily_mean(swe_dataset.data)
    seasonal_mean = calculate_seasonal_mean(swe_dataset.data, 
                                           start_date='10-01', 
                                           end_date='06-30')
    
    # Or using SWEDataset methods
    daily_mean = swe_dataset.calculate_daily_mean()
    seasonal_mean = swe_dataset.calculate_seasonal_mean(start_date='10-01', 
                                                       end_date='06-30')

### 3. SWE Integration and SSWEI Calculation

Original code:

.. code-block:: python

    # Integrate SWE over the season
    def integrate_season(ds, start_date='10-01', end_date='06-30'):
        # Implementation details...
        return integrated_swe
    
    integrated_swe = integrate_season(ds, start_date='10-01', end_date='06-30')
    
    # Calculate SSWEI
    def calculate_sswei(integrated_swe):
        # Implementation details...
        return sswei
    
    sswei = calculate_sswei(integrated_swe)

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Using SSWEI class
    sswei_obj = SSWEI(swe_dataset)
    sswei_obj.calculate(start_date='10-01', end_date='06-30')
    
    # Access the calculated SSWEI values
    sswei_values = sswei_obj.sswei
    
    # Alternatively, using individual functions
    from snowdroughtindex.core.sswei import integrate_season, calculate_sswei
    
    integrated_swe = integrate_season(swe_dataset.data, 
                                     start_date='10-01', 
                                     end_date='06-30')
    sswei = calculate_sswei(integrated_swe)

### 4. Drought Classification

Original code:

.. code-block:: python

    # Classify drought
    def classify_drought(sswei, thresholds=None):
        # Implementation details...
        return drought_class
    
    drought_class = classify_drought(sswei)

Migrated code:

.. code-block:: python

    # Using SSWEI class
    drought_class = sswei_obj.classify_drought()
    
    # Or using the function directly
    from snowdroughtindex.core.drought_classification import classify_drought
    
    drought_class = classify_drought(sswei_values)

### 5. Visualization

Original code:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Plot SSWEI time series
    plt.figure(figsize=(12, 6))
    plt.plot(sswei.time, sswei.values)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('SSWEI Time Series')
    plt.xlabel('Time')
    plt.ylabel('SSWEI')
    plt.show()
    
    # Plot drought classification
    plt.figure(figsize=(12, 6))
    plt.scatter(drought_class.time, drought_class.values, c=drought_class.values, cmap='RdYlBu')
    plt.colorbar(label='Drought Class')
    plt.title('Drought Classification')
    plt.xlabel('Time')
    plt.ylabel('Station')
    plt.show()

Migrated code:

.. code-block:: python

    from snowdroughtindex.utils.visualization import plot_sswei_timeseries, plot_drought_classification
    
    # Plot SSWEI time series
    plot_sswei_timeseries(sswei_obj.sswei)
    
    # Plot drought classification
    plot_drought_classification(drought_class)

Complete Migration Example
-----------------------

Here's a complete example showing how to migrate a typical workflow from the original ``SSWEI.ipynb`` notebook to the refactored package:

Original workflow:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data
    ds = xr.open_dataset('data/swe_data.nc')
    
    # Calculate seasonal mean
    seasonal_mean = ds.groupby('time.dayofyear').mean()
    
    # Integrate SWE over the season
    def integrate_season(ds, start_date='10-01', end_date='06-30'):
        # Implementation details...
        return integrated_swe
    
    integrated_swe = integrate_season(ds, start_date='10-01', end_date='06-30')
    
    # Calculate SSWEI
    def calculate_sswei(integrated_swe):
        # Implementation details...
        return sswei
    
    sswei = calculate_sswei(integrated_swe)
    
    # Classify drought
    def classify_drought(sswei, thresholds=None):
        # Implementation details...
        return drought_class
    
    drought_class = classify_drought(sswei)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(sswei.time, sswei.values)
    plt.title('SSWEI Time Series')
    plt.show()

Migrated workflow:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.utils.visualization import plot_sswei_timeseries, plot_drought_classification
    
    # Load data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Calculate SSWEI using SSWEI class
    sswei_obj = SSWEI(swe_dataset)
    sswei_obj.calculate(start_date='10-01', end_date='06-30')
    
    # Classify drought
    drought_class = sswei_obj.classify_drought()
    
    # Plot results
    plot_sswei_timeseries(sswei_obj.sswei)
    plot_drought_classification(drought_class)

Configuration Options
------------------

The refactored package includes a configuration system that allows you to customize parameters for SSWEI calculation and drought classification:

.. code-block:: python

    from snowdroughtindex.core.configuration import Configuration
    
    # Create a custom configuration
    config = Configuration()
    config.set_sswei_params(start_date='11-01', end_date='05-31')
    config.set_drought_thresholds({
        'extreme_wet': 1.5,
        'wet': 0.5,
        'normal': -0.5,
        'dry': -1.5,
        'extreme_dry': float('-inf')
    })
    
    # Use the configuration with SSWEI class
    sswei_obj = SSWEI(swe_dataset, config=config)
    sswei_obj.calculate()
    drought_class = sswei_obj.classify_drought()

Advanced Usage
-----------

For advanced usage scenarios, such as custom probability transformations or specialized visualization, refer to the API documentation:

- :doc:`/api/core`
- :doc:`/api/utils`

You can also check the example notebooks for more complex workflows:

- :doc:`/user_guide/workflows/sswei_calculation`
- :doc:`/user_guide/workflows/drought_classification`
