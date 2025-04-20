Class-Based Implementation
========================

This guide explains how to use the class-based implementation of the Snow Drought Index package.

Introduction
-----------

The Snow Drought Index package provides a class-based implementation that encapsulates related functionality and improves state management. The main classes are:

1. **SWEDataset**: Handles loading, preprocessing, and gap filling of SWE data.
2. **SSWEI**: Calculates the Standardized Snow Water Equivalent Index and classifies drought conditions.
3. **DroughtAnalysis**: Analyzes drought conditions across elevation bands and time periods.
4. **Configuration**: Manages parameters for gap filling, SSWEI calculation, and visualization.

Using these classes can simplify your code and make it more maintainable.

Configuration Class
-----------------

The ``Configuration`` class manages parameters for various operations in the package. It provides default values and allows you to customize parameters for your specific needs.

.. code-block:: python

    from snowdroughtindex.core.configuration import Configuration
    
    # Create a configuration object with default parameters
    config = Configuration()
    
    # Customize gap filling parameters
    config.set_gap_filling_params(
        method='linear',
        min_neighbors=3,
        max_distance=100,
        min_correlation=0.7
    )
    
    # Customize SSWEI calculation parameters
    config.set_sswei_params(
        start_month=11,
        start_day=1,
        end_month=4,
        end_day=30,
        min_swe_threshold=15,
        probability_method='gringorten'
    )
    
    # Customize visualization parameters
    config.set_visualization_params(
        figsize=(10, 6),
        cmap='viridis',
        dpi=300,
        save_format='png'
    )
    
    # Access parameters
    gap_filling_params = config.get_gap_filling_params()
    sswei_params = config.get_sswei_params()
    visualization_params = config.get_visualization_params()
    
    # Save configuration to a file
    config.save('config.yaml')
    
    # Load configuration from a file
    config = Configuration.load('config.yaml')

SWEDataset Class
--------------

The ``SWEDataset`` class handles loading, preprocessing, and gap filling of SWE data. It provides methods for data manipulation and analysis.

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.configuration import Configuration
    
    # Create a configuration object
    config = Configuration()
    
    # Create a SWEDataset object
    dataset = SWEDataset('path/to/swe_data.nc', config=config)
    
    # Load and preprocess data
    dataset.load_data()
    dataset.preprocess()
    
    # Filter data for a specific basin
    dataset.filter_by_basin('path/to/basin_shapefile.shp', basin_name='Basin Name')
    
    # Assess data availability
    availability = dataset.assess_data_availability()
    dataset.plot_data_availability(availability)
    
    # Fill gaps in the data
    dataset.fill_gaps(method='linear')
    
    # Evaluate gap filling performance
    performance = dataset.evaluate_gap_filling(
        test_fraction=0.2,
        random_seed=42
    )
    dataset.plot_gap_filling_performance(performance)
    
    # Extract data for analysis
    daily_mean = dataset.get_daily_mean()
    seasonal_mean = dataset.get_seasonal_mean()
    
    # Save processed data
    dataset.save('processed_data.nc')

SSWEI Class
---------

The ``SSWEI`` class calculates the Standardized Snow Water Equivalent Index and classifies drought conditions.

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.configuration import Configuration
    
    # Create a configuration object
    config = Configuration()
    
    # Create a SWEDataset object
    dataset = SWEDataset('path/to/swe_data.nc', config=config)
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Create an SSWEI object
    sswei = SSWEI(dataset, config=config)
    
    # Calculate SSWEI
    sswei.calculate()
    
    # Access SSWEI results
    integrated_swe = sswei.get_integrated_swe()
    probabilities = sswei.get_probabilities()
    sswei_values = sswei.get_sswei_values()
    
    # Classify drought conditions
    sswei.classify_drought()
    drought_classes = sswei.get_drought_classes()
    
    # Visualize results
    sswei.plot_sswei_timeseries()
    sswei.plot_drought_classification()
    
    # Save results
    sswei.save('sswei_results.csv')

DroughtAnalysis Class
------------------

The ``DroughtAnalysis`` class analyzes drought conditions across elevation bands and time periods.

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.drought_analysis import DroughtAnalysis
    from snowdroughtindex.core.configuration import Configuration
    
    # Create a configuration object
    config = Configuration()
    
    # Create a SWEDataset object
    dataset = SWEDataset('path/to/swe_data.nc', config=config)
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Create an SSWEI object
    sswei = SSWEI(dataset, config=config)
    sswei.calculate()
    sswei.classify_drought()
    
    # Create a DroughtAnalysis object
    analysis = DroughtAnalysis(sswei, config=config)
    
    # Analyze drought conditions by elevation bands
    analysis.analyze_elevation_bands(
        elevation_breaks=[1000, 1500, 2000, 2500, 3000],
        elevation_data='path/to/elevation_data.nc'
    )
    
    # Analyze drought trends
    analysis.analyze_trends(
        start_year=1980,
        end_year=2020,
        period_length=10
    )
    
    # Analyze drought frequency
    analysis.analyze_frequency()
    
    # Analyze drought duration
    analysis.analyze_duration()
    
    # Analyze drought severity
    analysis.analyze_severity()
    
    # Visualize results
    analysis.plot_elevation_analysis()
    analysis.plot_trend_analysis()
    analysis.plot_frequency_analysis()
    analysis.plot_duration_analysis()
    analysis.plot_severity_analysis()
    
    # Export results
    analysis.export_results('output_directory')

Complete Workflow Example
-----------------------

Here's a complete example that demonstrates how to use the class-based implementation for a typical workflow:

.. code-block:: python

    import matplotlib.pyplot as plt
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.drought_analysis import DroughtAnalysis
    from snowdroughtindex.core.configuration import Configuration
    
    # Create a configuration object with custom parameters
    config = Configuration()
    config.set_gap_filling_params(method='linear', min_neighbors=3)
    config.set_sswei_params(start_month=11, start_day=1, end_month=4, end_day=30)
    
    # Create a SWEDataset object
    dataset = SWEDataset('path/to/swe_data.nc', config=config)
    
    # Load and preprocess data
    dataset.load_data()
    dataset.preprocess()
    
    # Filter data for a specific basin
    dataset.filter_by_basin('path/to/basin_shapefile.shp', basin_name='Basin Name')
    
    # Assess data availability
    availability = dataset.assess_data_availability()
    plt.figure(figsize=(10, 6))
    dataset.plot_data_availability(availability)
    plt.title('Data Availability')
    plt.tight_layout()
    plt.show()
    
    # Fill gaps in the data
    dataset.fill_gaps()
    
    # Create an SSWEI object
    sswei = SSWEI(dataset)
    
    # Calculate SSWEI
    sswei.calculate()
    
    # Classify drought conditions
    sswei.classify_drought()
    
    # Visualize SSWEI results
    plt.figure(figsize=(10, 6))
    sswei.plot_sswei_timeseries()
    plt.title('SSWEI Time Series')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sswei.plot_drought_classification()
    plt.title('Drought Classification')
    plt.tight_layout()
    plt.show()
    
    # Create a DroughtAnalysis object
    analysis = DroughtAnalysis(sswei)
    
    # Analyze drought conditions by elevation bands
    analysis.analyze_elevation_bands(
        elevation_breaks=[1000, 1500, 2000, 2500, 3000],
        elevation_data='path/to/elevation_data.nc'
    )
    
    # Visualize elevation analysis
    plt.figure(figsize=(10, 6))
    analysis.plot_elevation_analysis()
    plt.title('Drought Conditions by Elevation Band')
    plt.tight_layout()
    plt.show()
    
    # Analyze drought trends
    analysis.analyze_trends(start_year=1980, end_year=2020, period_length=10)
    
    # Visualize trend analysis
    plt.figure(figsize=(10, 6))
    analysis.plot_trend_analysis()
    plt.title('Drought Trends')
    plt.tight_layout()
    plt.show()
    
    # Export results
    analysis.export_results('output_directory')

Advanced Usage
------------

The class-based implementation also supports advanced usage scenarios:

Custom Drought Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the drought classification thresholds:

.. code-block:: python

    # Create an SSWEI object
    sswei = SSWEI(dataset)
    
    # Calculate SSWEI
    sswei.calculate()
    
    # Customize drought classification thresholds
    custom_thresholds = {
        'Exceptional Drought': -2.5,
        'Extreme Drought': -2.0,
        'Severe Drought': -1.5,
        'Moderate Drought': -1.0,
        'Near Normal': 0.0,
        'Moderately Wet': 1.0,
        'Very Wet': 1.5,
        'Extremely Wet': 2.0
    }
    
    # Classify drought with custom thresholds
    sswei.classify_drought(thresholds=custom_thresholds)

Custom Visualization
^^^^^^^^^^^^^^^^^

You can customize the visualization of results:

.. code-block:: python

    # Create a DroughtAnalysis object
    analysis = DroughtAnalysis(sswei)
    
    # Analyze drought conditions
    analysis.analyze_elevation_bands()
    
    # Customize visualization
    plt.figure(figsize=(12, 8))
    analysis.plot_elevation_analysis(
        cmap='RdYlBu',
        title='Custom Drought Analysis by Elevation',
        xlabel='Year',
        ylabel='Elevation Band',
        legend_title='Drought Class',
        grid=True,
        colorbar=True
    )
    plt.tight_layout()
    plt.show()

Parallel Processing
^^^^^^^^^^^^^^^^

For large datasets, you can enable parallel processing:

.. code-block:: python

    # Create a SWEDataset object with parallel processing enabled
    dataset = SWEDataset('path/to/swe_data.nc', parallel=True, n_jobs=-1)
    
    # Fill gaps with parallel processing
    dataset.fill_gaps(parallel=True, n_jobs=-1)
    
    # Create an SSWEI object with parallel processing enabled
    sswei = SSWEI(dataset, parallel=True, n_jobs=-1)
    
    # Calculate SSWEI with parallel processing
    sswei.calculate(parallel=True, n_jobs=-1)

Conclusion
---------

The class-based implementation of the Snow Drought Index package provides a more organized and maintainable way to analyze snow drought conditions. By encapsulating related functionality into classes, it simplifies complex workflows and improves code readability.

For more examples and detailed API documentation, refer to:

- :doc:`API Reference <../api/core>`
- :doc:`Example Notebooks <../user_guide/examples>`
- :doc:`Workflow Guides <../user_guide/workflows>`
