Advanced Usage
=============

This guide covers advanced usage scenarios and techniques for the Snow Drought Index package, building upon the concepts introduced in the quickstart and class-based implementation guides.

Introduction
-----------

The Snow Drought Index package provides advanced capabilities for customizing analyses, extending functionality, and integrating with other tools and workflows. This guide covers:

1. Custom analysis workflows
2. Extending the package with custom functions
3. Integration with other data sources and tools
4. Advanced configuration options
5. Command-line interface usage
6. Batch processing and automation
7. Working with custom drought classification schemes
8. Advanced visualization techniques
9. Spatial analysis and interpolation
10. Temporal trend analysis

Custom Analysis Workflows
-----------------------

Creating Custom Workflows
^^^^^^^^^^^^^^^^^^^^^^^

You can create custom analysis workflows by combining the package's components in new ways:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.utils.visualization import plot_custom_analysis
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a custom analysis function
    def analyze_drought_frequency_by_decade(dataset, start_year, end_year):
        # Create SSWEI object
        sswei_obj = SSWEI(dataset)
        sswei_obj.calculate()
        sswei_obj.classify_drought()
        
        # Get drought classes
        drought_classes = sswei_obj.get_drought_classes()
        
        # Analyze by decade
        decades = range(start_year, end_year, 10)
        results = {}
        
        for decade in decades:
            decade_end = min(decade + 9, end_year)
            decade_mask = (drought_classes.index.year >= decade) & (drought_classes.index.year <= decade_end)
            decade_data = drought_classes[decade_mask]
            
            # Count drought occurrences by class
            drought_counts = decade_data.value_counts().to_dict()
            results[f"{decade}-{decade_end}"] = drought_counts
        
        return results
    
    # Use the custom analysis function
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Run custom analysis
    frequency_by_decade = analyze_drought_frequency_by_decade(dataset, 1980, 2020)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    decades = list(frequency_by_decade.keys())
    drought_classes = ['Exceptional Drought', 'Extreme Drought', 'Severe Drought', 'Moderate Drought']
    
    for drought_class in drought_classes:
        values = [frequency_by_decade[decade].get(drought_class, 0) for decade in decades]
        plt.plot(decades, values, marker='o', label=drought_class)
    
    plt.title('Drought Frequency by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

Combining Multiple Data Sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can combine multiple data sources for more comprehensive analyses:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    import xarray as xr
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load SWE data
    swe_dataset = SWEDataset('path/to/swe_data.nc')
    swe_dataset.load_data()
    swe_dataset.preprocess()
    swe_dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(swe_dataset)
    sswei_obj.calculate()
    sswei_obj.classify_drought()
    drought_classes = sswei_obj.get_drought_classes()
    
    # Load precipitation data
    precip_data = xr.open_dataset('path/to/precipitation_data.nc')
    
    # Extract precipitation time series for a specific location
    lat, lon = 40.0, -120.0
    precip_ts = precip_data.sel(lat=lat, lon=lon, method='nearest').precipitation.to_pandas()
    
    # Combine with drought classification
    combined_data = pd.DataFrame({
        'Drought Class': drought_classes,
        'Precipitation': precip_ts
    })
    
    # Analyze relationship between precipitation and drought class
    drought_precip = combined_data.groupby('Drought Class')['Precipitation'].mean()
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    drought_precip.plot(kind='bar')
    plt.title('Average Precipitation by Drought Class')
    plt.xlabel('Drought Class')
    plt.ylabel('Average Precipitation (mm)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

Extending the Package
-------------------

Creating Custom Functions
^^^^^^^^^^^^^^^^^^^^^^^

You can extend the package with custom functions:

.. code-block:: python

    from snowdroughtindex.core.sswei import compute_swei
    import numpy as np
    from scipy import stats
    
    # Create a custom SSWEI calculation function
    def compute_custom_swei(probabilities, distribution='gamma'):
        """
        Compute SSWEI using a custom probability distribution.
        
        Parameters
        ----------
        probabilities : array-like
            Probabilities calculated from integrated SWE.
        distribution : str, optional
            Probability distribution to use ('gamma', 'weibull', or 'normal').
            Default is 'gamma'.
            
        Returns
        -------
        array-like
            SSWEI values.
        """
        if distribution == 'gamma':
            # Fit gamma distribution parameters
            shape, loc, scale = stats.gamma.fit(probabilities)
            # Transform to standard normal
            swei = stats.gamma.ppf(probabilities, shape, loc=loc, scale=scale)
        elif distribution == 'weibull':
            # Fit Weibull distribution parameters
            shape, loc, scale = stats.weibull_min.fit(probabilities)
            # Transform to standard normal
            swei = stats.weibull_min.ppf(probabilities, shape, loc=loc, scale=scale)
        elif distribution == 'normal':
            # Use standard normal distribution (default behavior)
            swei = stats.norm.ppf(probabilities)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        return swei
    
    # Use the custom function
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Calculate SSWEI using different distributions
    swei_normal = compute_custom_swei(probabilities, distribution='normal')
    swei_gamma = compute_custom_swei(probabilities, distribution='gamma')
    swei_weibull = compute_custom_swei(probabilities, distribution='weibull')
    
    # Compare results
    print("Normal distribution:", swei_normal)
    print("Gamma distribution:", swei_gamma)
    print("Weibull distribution:", swei_weibull)

Creating Custom Classes
^^^^^^^^^^^^^^^^^^^^^

You can create custom classes that extend the package's functionality:

.. code-block:: python

    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.dataset import SWEDataset
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    class ExtendedSSWEI(SSWEI):
        """
        Extended SSWEI class with additional functionality.
        """
        
        def __init__(self, dataset, config=None):
            super().__init__(dataset, config)
            self.drought_duration = None
            self.drought_severity = None
        
        def analyze_drought_duration(self):
            """
            Analyze drought duration.
            
            Returns
            -------
            pandas.DataFrame
                Drought duration statistics.
            """
            if self.drought_classes is None:
                self.classify_drought()
            
            # Identify drought periods
            is_drought = self.drought_classes.isin(['Moderate Drought', 'Severe Drought', 
                                                   'Extreme Drought', 'Exceptional Drought'])
            
            # Calculate drought duration
            drought_periods = []
            current_period = {'start': None, 'end': None, 'duration': 0}
            
            for date, is_drought_day in is_drought.items():
                if is_drought_day:
                    if current_period['start'] is None:
                        current_period['start'] = date
                else:
                    if current_period['start'] is not None:
                        current_period['end'] = date
                        current_period['duration'] = (current_period['end'] - current_period['start']).days
                        drought_periods.append(current_period.copy())
                        current_period = {'start': None, 'end': None, 'duration': 0}
            
            # Handle ongoing drought at the end of the time series
            if current_period['start'] is not None:
                current_period['end'] = is_drought.index[-1]
                current_period['duration'] = (current_period['end'] - current_period['start']).days
                drought_periods.append(current_period)
            
            # Create DataFrame
            self.drought_duration = pd.DataFrame(drought_periods)
            
            return self.drought_duration
        
        def analyze_drought_severity(self):
            """
            Analyze drought severity.
            
            Returns
            -------
            pandas.DataFrame
                Drought severity statistics.
            """
            if self.sswei_values is None:
                self.calculate()
            
            if self.drought_classes is None:
                self.classify_drought()
            
            # Calculate drought severity (magnitude and duration)
            drought_severity = []
            
            for drought_period in self.analyze_drought_duration().itertuples():
                start_date = drought_period.start
                end_date = drought_period.end
                
                # Get SSWEI values during the drought period
                period_mask = (self.sswei_values.index >= start_date) & (self.sswei_values.index <= end_date)
                period_sswei = self.sswei_values[period_mask]
                
                # Calculate severity metrics
                severity = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': drought_period.duration,
                    'mean_sswei': period_sswei.mean(),
                    'min_sswei': period_sswei.min(),
                    'cumulative_deficit': period_sswei[period_sswei < 0].sum()
                }
                
                drought_severity.append(severity)
            
            # Create DataFrame
            self.drought_severity = pd.DataFrame(drought_severity)
            
            return self.drought_severity
        
        def plot_drought_duration_histogram(self):
            """
            Plot histogram of drought durations.
            """
            if self.drought_duration is None:
                self.analyze_drought_duration()
            
            plt.figure(figsize=(10, 6))
            plt.hist(self.drought_duration['duration'], bins=20, edgecolor='black')
            plt.title('Drought Duration Histogram')
            plt.xlabel('Duration (days)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        def plot_severity_vs_duration(self):
            """
            Plot drought severity vs. duration.
            """
            if self.drought_severity is None:
                self.analyze_drought_severity()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.drought_severity['duration_days'], 
                       self.drought_severity['cumulative_deficit'].abs(),
                       alpha=0.7)
            plt.title('Drought Severity vs. Duration')
            plt.xlabel('Duration (days)')
            plt.ylabel('Cumulative Deficit (absolute value)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
    
    # Use the extended class
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Create extended SSWEI object
    extended_sswei = ExtendedSSWEI(dataset)
    extended_sswei.calculate()
    extended_sswei.classify_drought()
    
    # Use new functionality
    drought_duration = extended_sswei.analyze_drought_duration()
    drought_severity = extended_sswei.analyze_drought_severity()
    
    # Visualize results
    extended_sswei.plot_drought_duration_histogram()
    extended_sswei.plot_severity_vs_duration()

Integration with Other Tools
--------------------------

Integration with Climate Data Operators (CDO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can integrate with CDO for advanced data processing:

.. code-block:: python

    import os
    import subprocess
    from snowdroughtindex.core.dataset import SWEDataset
    
    def process_with_cdo(input_file, output_file, operation):
        """
        Process a NetCDF file using CDO.
        
        Parameters
        ----------
        input_file : str
            Path to input NetCDF file.
        output_file : str
            Path to output NetCDF file.
        operation : str
            CDO operation to perform.
            
        Returns
        -------
        str
            Path to output file.
        """
        cmd = f"cdo {operation} {input_file} {output_file}"
        subprocess.run(cmd, shell=True, check=True)
        return output_file
    
    # Example usage
    input_file = 'path/to/swe_data.nc'
    
    # Calculate seasonal mean using CDO
    seasonal_mean_file = process_with_cdo(
        input_file, 
        'seasonal_mean.nc', 
        'seasmean'
    )
    
    # Calculate spatial mean using CDO
    spatial_mean_file = process_with_cdo(
        input_file, 
        'spatial_mean.nc', 
        'fldmean'
    )
    
    # Load processed data
    seasonal_mean_dataset = SWEDataset(seasonal_mean_file)
    seasonal_mean_dataset.load_data()

Integration with GIS Tools
^^^^^^^^^^^^^^^^^^^^^^^^

You can integrate with GIS tools for spatial analysis:

.. code-block:: python

    import geopandas as gpd
    import xarray as xr
    import matplotlib.pyplot as plt
    from snowdroughtindex.core.sswei_class import SSWEI
    from snowdroughtindex.core.dataset import SWEDataset
    
    # Load SWE data
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(dataset)
    sswei_obj.calculate()
    sswei_obj.classify_drought()
    
    # Get SSWEI values for a specific year
    year = 2015
    sswei_values = sswei_obj.get_sswei_values()
    sswei_year = sswei_values[sswei_values.index.year == year]
    
    # Convert to spatial dataset
    sswei_spatial = dataset.data.sel(time=f"{year}").copy()
    sswei_spatial['sswei'] = xr.DataArray(
        sswei_year.values,
        dims=['time'],
        coords={'time': sswei_year.index}
    )
    
    # Export to GeoTIFF for GIS analysis
    sswei_spatial['sswei'].mean('time').to_netcdf('sswei_spatial.nc')
    
    # Use subprocess to convert NetCDF to GeoTIFF
    import subprocess
    subprocess.run(
        "gdal_translate -of GTiff sswei_spatial.nc sswei_spatial.tif",
        shell=True, check=True
    )
    
    # Load basin shapefile
    basins = gpd.read_file('path/to/basin_shapefile.shp')
    
    # Plot SSWEI with basin boundaries
    fig, ax = plt.subplots(figsize=(12, 8))
    sswei_spatial['sswei'].mean('time').plot(ax=ax, cmap='RdBu_r')
    basins.boundary.plot(ax=ax, color='black', linewidth=1)
    plt.title(f'SSWEI Spatial Distribution ({year})')
    plt.tight_layout()
    plt.show()

Advanced Configuration Options
----------------------------

Using YAML Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use YAML configuration files for more flexible configuration:

.. code-block:: python

    from snowdroughtindex.core.configuration import Configuration
    import yaml
    
    # Create a configuration object
    config = Configuration()
    
    # Set parameters
    config.set_gap_filling_params(method='linear', min_neighbors=3)
    config.set_sswei_params(start_month=11, start_day=1, end_month=4, end_day=30)
    config.set_visualization_params(figsize=(10, 6), cmap='viridis')
    
    # Save configuration to YAML file
    with open('config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration object from dictionary
    loaded_config = Configuration.from_dict(config_dict)
    
    # Use the loaded configuration
    from snowdroughtindex.core.dataset import SWEDataset
    dataset = SWEDataset('path/to/swe_data.nc', config=loaded_config)

Example YAML configuration file:

.. code-block:: yaml

    gap_filling:
      method: linear
      min_neighbors: 3
      max_distance: 100
      min_correlation: 0.7
    
    sswei:
      start_month: 11
      start_day: 1
      end_month: 4
      end_day: 30
      min_swe_threshold: 15
      probability_method: gringorten
    
    visualization:
      figsize:
        - 10
        - 6
      cmap: viridis
      dpi: 300
      save_format: png
    
    performance:
      parallel: true
      n_jobs: 4
      lazy_loading: true
      chunks:
        time: 100
        lat: 50
        lon: 50
      memory_efficient: true
      enable_caching: true
      cache_dir: ./cache

Environment Variables
^^^^^^^^^^^^^^^^^^

You can use environment variables to override configuration settings:

.. code-block:: python

    import os
    from snowdroughtindex.core.configuration import Configuration
    
    # Set environment variables
    os.environ['SNOWDROUGHT_GAP_FILLING_METHOD'] = 'linear'
    os.environ['SNOWDROUGHT_SSWEI_START_MONTH'] = '11'
    os.environ['SNOWDROUGHT_SSWEI_END_MONTH'] = '4'
    os.environ['SNOWDROUGHT_PARALLEL'] = 'true'
    os.environ['SNOWDROUGHT_N_JOBS'] = '4'
    
    # Create configuration with environment variable support
    config = Configuration.from_env()
    
    # Use the configuration
    from snowdroughtindex.core.dataset import SWEDataset
    dataset = SWEDataset('path/to/swe_data.nc', config=config)

Command-Line Interface
--------------------

Basic CLI Usage
^^^^^^^^^^^^^

The package provides a command-line interface for common operations:

.. code-block:: bash

    # Calculate SSWEI
    python -m snowdroughtindex.cli calculate-sswei \
        --input-file path/to/swe_data.nc \
        --output-file sswei_results.csv \
        --start-month 11 \
        --start-day 1 \
        --end-month 4 \
        --end-day 30
    
    # Classify drought
    python -m snowdroughtindex.cli classify-drought \
        --input-file sswei_results.csv \
        --output-file drought_classes.csv
    
    # Generate plots
    python -m snowdroughtindex.cli plot-sswei \
        --input-file sswei_results.csv \
        --output-file sswei_plot.png \
        --title "SSWEI Time Series" \
        --figsize 10 6
    
    # Run a complete workflow
    python -m snowdroughtindex.cli run-workflow \
        --input-file path/to/swe_data.nc \
        --output-dir results \
        --config-file config.yaml

Advanced CLI Options
^^^^^^^^^^^^^^^^^

The CLI supports advanced options for customization:

.. code-block:: bash

    # Fill gaps with custom parameters
    python -m snowdroughtindex.cli fill-gaps \
        --input-file path/to/swe_data.nc \
        --output-file filled_data.nc \
        --method linear \
        --min-neighbors 3 \
        --max-distance 100 \
        --min-correlation 0.7 \
        --parallel \
        --n-jobs 4
    
    # Calculate SSWEI with custom parameters
    python -m snowdroughtindex.cli calculate-sswei \
        --input-file filled_data.nc \
        --output-file sswei_results.csv \
        --start-month 11 \
        --start-day 1 \
        --end-month 4 \
        --end-day 30 \
        --min-swe-threshold 15 \
        --probability-method gringorten \
        --distribution normal
    
    # Classify drought with custom thresholds
    python -m snowdroughtindex.cli classify-drought \
        --input-file sswei_results.csv \
        --output-file drought_classes.csv \
        --thresholds exceptional=-2.5 extreme=-2.0 severe=-1.5 moderate=-1.0 \
                     normal=0.0 moderately_wet=1.0 very_wet=1.5 extremely_wet=2.0

Batch Processing and Automation
-----------------------------

Processing Multiple Files
^^^^^^^^^^^^^^^^^^^^^^^

You can process multiple files in batch:

.. code-block:: python

    import os
    import glob
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    import pandas as pd
    
    def process_file(file_path, output_dir):
        """
        Process a single SWE data file.
        
        Parameters
        ----------
        file_path : str
            Path to SWE data file.
        output_dir : str
            Directory to save results.
            
        Returns
        -------
        dict
            Processing results.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load and process data
        dataset = SWEDataset(file_path)
        dataset.load_data()
        dataset.preprocess()
        dataset.fill_gaps()
        
        # Calculate SSWEI
        sswei_obj = SSWEI(dataset)
        sswei_obj.calculate()
        sswei_obj.classify_drought()
        
        # Get results
        sswei_values = sswei_obj.get_sswei_values()
        drought_classes = sswei_obj.get_drought_classes()
        
        # Save results
        sswei_values.to_csv(os.path.join(output_dir, f"{filename}_sswei.csv"))
        drought_classes.to_csv(os.path.join(output_dir, f"{filename}_drought.csv"))
        
        # Create summary
        summary = {
            'file': file_path,
            'n_stations': len(dataset.data.station),
            'start_date': sswei_values.index.min(),
            'end_date': sswei_values.index.max(),
            'mean_sswei': sswei_values.mean(),
            'min_sswei': sswei_values.min(),
            'max_sswei': sswei_values.max()
        }
        
        return summary
    
    # Process multiple files
    input_files = glob.glob('path/to/swe_data/*.nc')
    output_dir = 'results'
    
    # Process each file
    summaries = []
    for file_path in input_files:
        summary = process_file(file_path, output_dir)
        summaries.append(summary)
    
    # Create summary report
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    print(f"Processed {len(input_files)} files. Results saved to {output_dir}")

Automation with Airflow
^^^^^^^^^^^^^^^^^^^^^

You can automate workflows with Apache Airflow:

.. code-block:: python

    # airflow_dag.py
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    
    # Import package functions
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Define default arguments
    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
    
    # Create DAG
    dag = DAG(
        'snow_drought_analysis',
        default_args=default_args,
        description='Snow Drought Analysis Workflow',
        schedule_interval=timedelta(days=1),
    )
    
    # Define tasks
    def load_and_preprocess(**kwargs):
        file_path = kwargs['file_path']
        dataset = SWEDataset(file_path)
        dataset.load_data()
        dataset.preprocess()
        return dataset
    
    def fill_gaps(**kwargs):
        ti = kwargs['ti']
        dataset = ti.xcom_pull(task_ids='load_and_preprocess')
        dataset.fill_gaps()
        return dataset
    
    def calculate_sswei(**kwargs):
        ti = kwargs['ti']
        dataset = ti.xcom_pull(task_ids='fill_gaps')
        sswei_obj = SSWEI(dataset)
        sswei_obj.calculate()
        return sswei_obj
    
    def classify_drought(**kwargs):
        ti = kwargs['ti']
        sswei_obj = ti.xcom_pull(task_ids='calculate_sswei')
        sswei_obj.classify_drought()
        
        # Save results
        output_file = kwargs['output_file']
        drought_classes = sswei_obj.get_drought_classes()
        drought_classes.to_csv(output_file)
        
        return output_file
    
    # Create tasks
    load_task = PythonOperator(
        task_id='load_and_preprocess',
        python_callable=load_and_preprocess,
        op_kwargs={'file_path': 'path/to/swe_data.nc'},
        dag=dag,
    )
    
    fill_gaps_task = PythonOperator(
        task_id='fill_gaps',
        python_callable=fill_gaps,
        provide_context=True,
        dag=dag,
    )
    
    calculate_sswei_task = PythonOperator(
        task_id='calculate_sswei',
        python_callable=calculate_sswei,
        provide_context=True,
        dag=dag,
    )
    
    classify_drought_task = PythonOperator(
        task_id='classify_drought',
        python_callable=classify_drought,
        op_kwargs={'output_file': 'path/to/output/drought_classes.csv'},
        provide_context=True,
        dag=dag,
    )
    
    # Define task dependencies
    load_task >> fill_gaps_task >> calculate_sswei_task >> classify_drought_task

Custom Drought Classification
---------------------------

Creating Custom Classification Schemes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create custom drought classification schemes:

.. code-block:: python

    from snowdroughtindex.core.drought_classification import classify_drought
    from snowdroughtindex.core.sswei_class import SSWEI
    import numpy as np
    import pandas as pd
    
    # Define a custom classification function
    def custom_classify_drought(sswei_values, thresholds=None):
        """
        Classify drought conditions based on SSWEI values using custom thresholds.
        
        Parameters
        ----------
        sswei_values : array-like
            SSWEI values.
        thresholds : dict, optional
            Custom thresholds for drought classification.
            
        Returns
        -------
        pandas.Series
            Drought classes.
        """
        if thresholds is None:
            thresholds = {
                'Exceptional Drought': -2.5,
                'Extreme Drought': -2.0,
                'Severe Drought': -1.5,
                'Moderate Drought': -1.0,
                'Near Normal': 0.0,
                'Moderately Wet': 1.0,
                'Very Wet': 1.5,
                'Extremely Wet': 2.0
            }
        
        # Sort thresholds by value
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
        
        # Initialize with the lowest class
        classes = np.full_like(sswei_values, sorted_thresholds[0][0], dtype=object)
        
        # Classify based on thresholds
        for i in range(1, len(sorted_thresholds)):
            class_name = sorted_thresholds[i][0]
            threshold = sorted_thresholds[i][1]
            classes[sswei_values >= threshold] = class_name
        
        # Convert to pandas Series if input is a Series
        if isinstance(sswei_values, pd.Series):
            classes = pd.Series(classes, index=sswei_values.index)
        
        return classes
    
    # Use the custom classification function
    from snowdroughtindex.core.dataset import SWEDataset
    
    # Load and process data
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(dataset)
    sswei_obj.calculate()
    sswei_values = sswei_obj.get_sswei_values()
    
    # Define custom thresholds
    custom_thresholds = {
        'Extreme Deficit': -3.0,
        'Severe Deficit': -2.0,
        'Moderate Deficit': -1.0,
        'Normal': 0.0,
        'Moderate Surplus': 1.0,
        'Severe Surplus': 2.0,
        'Extreme Surplus': 3.0
    }
    
    # Classify drought using custom thresholds
    custom_drought_classes = custom_classify_drought(sswei_values, thresholds=custom_thresholds)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    custom_drought_classes.value_counts().sort_index().plot(kind='bar')
    plt.title('Drought Classification with Custom Thresholds')
    plt.xlabel('Drought Class')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

Advanced Visualization Techniques
-------------------------------

Custom Visualization Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create custom visualization functions for specific analysis needs:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap
    
    def plot_drought_heatmap(drought_classes, years=None, months=None, cmap=None, ax=None):
        """
        Plot a heatmap of drought classes over time.
        
        Parameters
        ----------
        drought_classes : pandas.Series
            Drought classes.
        years : list, optional
            Years to include in the plot.
        months : list, optional
            Months to include in the plot.
        cmap : matplotlib.colors.Colormap, optional
            Colormap to use.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
            
        Returns
        -------
        matplotlib.axes.Axes
            Axes with the plot.
        """
        # Create a mapping of drought classes to numeric values
        class_mapping = {
            'Exceptional Drought': -4,
            'Extreme Drought': -3,
            'Severe Drought': -2,
            'Moderate Drought': -1,
            'Near Normal': 0,
            'Moderately Wet': 1,
            'Very Wet': 2,
            'Extremely Wet': 3
        }
        
        # Convert drought classes to numeric values
        numeric_classes = drought_classes.map(class_mapping)
        
        # Create a DataFrame with years as rows and months as columns
        if years is None:
            years = sorted(set(drought_classes.index.year))
        
        if months is None:
            months = range(1, 13)
        
        # Create an empty matrix
        matrix = np.full((len(years), len(months)), np.nan)
        
        # Fill the matrix with drought class values
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                mask = (numeric_classes.index.year == year) & (numeric_classes.index.month == month)
                if mask.any():
                    matrix[i, j] = numeric_classes[mask].mean()
        
        # Create a custom colormap
        if cmap is None:
            colors = ['#730000', '#E60000', '#FFAA00', '#FCD37F', '#FFFFFF', '#B3DBFF', '#6699FF', '#0000FF']
            cmap = ListedColormap(colors)
        
        # Create a figure if ax is not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-4, vmax=3)
        
        # Set ticks and labels
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([pd.Timestamp(2000, month, 1).strftime('%b') for month in months])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([-4, -3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(['Exceptional Drought', 'Extreme Drought', 'Severe Drought', 'Moderate Drought',
                            'Near Normal', 'Moderately Wet', 'Very Wet', 'Extremely Wet'])
        
        # Add grid
        ax.grid(False)
        
        # Add title and labels
        ax.set_title('Drought Conditions Over Time')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        return ax
    
    # Use the custom visualization function
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Load and process data
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(dataset)
    sswei_obj.calculate()
    sswei_obj.classify_drought()
    drought_classes = sswei_obj.get_drought_classes()
    
    # Plot drought heatmap
    plt.figure(figsize=(12, 8))
    plot_drought_heatmap(drought_classes, years=range(1980, 2021), months=range(1, 13))
    plt.tight_layout()
    plt.show()

Interactive Visualization
^^^^^^^^^^^^^^^^^^^^^^

You can create interactive visualizations using libraries like Plotly:

.. code-block:: python

    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    def create_interactive_sswei_plot(sswei_values, drought_classes):
        """
        Create an interactive plot of SSWEI values and drought classes.
        
        Parameters
        ----------
        sswei_values : pandas.Series
            SSWEI values.
        drought_classes : pandas.Series
            Drought classes.
            
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure.
        """
        # Create a DataFrame with SSWEI values and drought classes
        df = pd.DataFrame({
            'SSWEI': sswei_values,
            'Drought Class': drought_classes,
            'Date': sswei_values.index
        })
        
        # Create a color mapping for drought classes
        color_mapping = {
            'Exceptional Drought': '#730000',
            'Extreme Drought': '#E60000',
            'Severe Drought': '#FFAA00',
            'Moderate Drought': '#FCD37F',
            'Near Normal': '#FFFFFF',
            'Moderately Wet': '#B3DBFF',
            'Very Wet': '#6699FF',
            'Extremely Wet': '#0000FF'
        }
        
        # Create the figure
        fig = go.Figure()
        
        # Add SSWEI line
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SSWEI'],
            mode='lines',
            name='SSWEI',
            line=dict(color='black', width=2)
        ))
        
        # Add drought class markers
        for drought_class, color in color_mapping.items():
            mask = df['Drought Class'] == drought_class
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, 'Date'],
                    y=df.loc[mask, 'SSWEI'],
                    mode='markers',
                    name=drought_class,
                    marker=dict(color=color, size=8)
                ))
        
        # Add threshold lines
        thresholds = {
            'Exceptional Drought': -2.5,
            'Extreme Drought': -2.0,
            'Severe Drought': -1.5,
            'Moderate Drought': -1.0,
            'Near Normal': 0.0,
            'Moderately Wet': 1.0,
            'Very Wet': 1.5,
            'Extremely Wet': 2.0
        }
        
        for drought_class, threshold in thresholds.items():
            fig.add_shape(
                type='line',
                x0=df['Date'].min(),
                y0=threshold,
                x1=df['Date'].max(),
                y1=threshold,
                line=dict(color='gray', width=1, dash='dash'),
                name=f"{drought_class} Threshold"
            )
        
        # Update layout
        fig.update_layout(
            title='SSWEI Time Series with Drought Classification',
            xaxis_title='Date',
            yaxis_title='SSWEI',
            legend_title='Drought Class',
            hovermode='closest',
            height=600,
            width=1000
        )
        
        return fig
    
    # Use the interactive visualization function
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Load and process data
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(dataset)
    sswei_obj.calculate()
    sswei_obj.classify_drought()
    sswei_values = sswei_obj.get_sswei_values()
    drought_classes = sswei_obj.get_drought_classes()
    
    # Create interactive plot
    fig = create_interactive_sswei_plot(sswei_values, drought_classes)
    fig.show()

Spatial Analysis and Interpolation
--------------------------------

Spatial Interpolation
^^^^^^^^^^^^^^^^^^^

You can perform spatial interpolation to create continuous surfaces from point data:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import geopandas as gpd
    from shapely.geometry import Point
    
    def interpolate_sswei(sswei_values, stations, resolution=100):
        """
        Interpolate SSWEI values to create a continuous surface.
        
        Parameters
        ----------
        sswei_values : pandas.Series
            SSWEI values.
        stations : pandas.DataFrame
            Station information with latitude and longitude.
        resolution : int, optional
            Resolution of the interpolation grid.
            
        Returns
        -------
        tuple
            Tuple containing (x_grid, y_grid, z_grid) for plotting.
        """
        # Extract coordinates and values
        lons = stations['longitude'].values
        lats = stations['latitude'].values
        values = sswei_values.values
        
        # Create a grid for interpolation
        lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5
        lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
        
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(lon_min, lon_max, resolution),
            np.linspace(lat_min, lat_max, resolution)
        )
        
        # Interpolate values
        points = np.column_stack((lons, lats))
        grid_values = griddata(points, values, (lon_grid, lat_grid), method='cubic')
        
        return lon_grid, lat_grid, grid_values
    
    # Use the spatial interpolation function
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Load and process data
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(dataset)
    sswei_obj.calculate()
    sswei_values = sswei_obj.get_sswei_values()
    
    # Get station information
    stations = pd.DataFrame({
        'station_id': dataset.data.station.values,
        'latitude': dataset.data.lat.values,
        'longitude': dataset.data.lon.values
    })
    
    # Interpolate SSWEI values for a specific date
    date = '2015-01-01'
    sswei_date = sswei_values.loc[date]
    
    # Perform interpolation
    lon_grid, lat_grid, grid_values = interpolate_sswei(sswei_date, stations)
    
    # Plot interpolated surface
    plt.figure(figsize=(12, 8))
    
    # Create a contour plot
    contour = plt.contourf(lon_grid, lat_grid, grid_values, cmap='RdBu_r', levels=20)
    
    # Add station points
    plt.scatter(stations['longitude'], stations['latitude'], c='black', s=10)
    
    # Add colorbar
    plt.colorbar(contour, label='SSWEI')
    
    # Add title and labels
    plt.title(f'Interpolated SSWEI Values ({date})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add basemap
    try:
        import contextily as ctx
        ctx.add_basemap(plt.gca(), crs='EPSG:4326')
    except ImportError:
        print("Install contextily for basemaps")
    
    plt.tight_layout()
    plt.show()

Temporal Trend Analysis
---------------------

Trend Detection
^^^^^^^^^^^^^

You can perform trend detection to identify long-term changes in drought conditions:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def analyze_drought_trends(sswei_values, window_size=10):
        """
        Analyze trends in SSWEI values.
        
        Parameters
        ----------
        sswei_values : pandas.Series
            SSWEI values.
        window_size : int, optional
            Window size for rolling statistics.
            
        Returns
        -------
        pandas.DataFrame
            Trend analysis results.
        """
        # Resample to annual values
        annual_sswei = sswei_values.resample('A').mean()
        
        # Calculate rolling statistics
        rolling_mean = annual_sswei.rolling(window=window_size).mean()
        rolling_std = annual_sswei.rolling(window=window_size).std()
        
        # Calculate trend using linear regression
        years = np.arange(len(annual_sswei))
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, annual_sswei)
        
        # Calculate trend line
        trend_line = intercept + slope * years
        
        # Create results DataFrame
        results = pd.DataFrame({
            'SSWEI': annual_sswei,
            'Rolling Mean': rolling_mean,
            'Rolling Std': rolling_std,
            'Trend Line': trend_line
        })
        
        # Add trend statistics
        trend_stats = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }
        
        return results, trend_stats
    
    # Use the trend analysis function
    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Load and process data
    dataset = SWEDataset('path/to/swe_data.nc')
    dataset.load_data()
    dataset.preprocess()
    dataset.fill_gaps()
    
    # Calculate SSWEI
    sswei_obj = SSWEI(dataset)
    sswei_obj.calculate()
    sswei_values = sswei_obj.get_sswei_values()
    
    # Analyze trends
    trend_results, trend_stats = analyze_drought_trends(sswei_values, window_size=10)
    
    # Plot trend analysis
    plt.figure(figsize=(12, 8))
    
    # Plot annual SSWEI values
    plt.plot(trend_results.index, trend_results['SSWEI'], 'o-', color='gray', alpha=0.7, label='Annual SSWEI')
    
    # Plot rolling mean
    plt.plot(trend_results.index, trend_results['Rolling Mean'], 'r-', linewidth=2, label=f'{10}-Year Rolling Mean')
    
    # Plot trend line
    plt.plot(trend_results.index, trend_results['Trend Line'], 'b--', linewidth=2, label='Linear Trend')
    
    # Add confidence interval for rolling mean
    plt.fill_between(
        trend_results.index,
        trend_results['Rolling Mean'] - trend_results['Rolling Std'],
        trend_results['Rolling Mean'] + trend_results['Rolling Std'],
        color='r', alpha=0.2, label=f'{10}-Year Standard Deviation'
    )
    
    # Add trend statistics
    plt.text(
        0.05, 0.05,
        f"Trend: {trend_stats['slope']:.4f} per year (p={trend_stats['p_value']:.4f})",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Add title and labels
    plt.title('SSWEI Trend Analysis')
    plt.xlabel('Year')
    plt.ylabel('SSWEI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Conclusion
---------

This advanced usage guide has covered a wide range of techniques for extending and customizing the Snow Drought Index package. By leveraging these advanced capabilities, you can create more sophisticated analyses, integrate with other tools and data sources, and automate complex workflows.

For more information, refer to:

- :doc:`API Reference <../api/core>`
- :doc:`Class-Based Implementation Guide <../user_guide/class_based_implementation>`
- :doc:`Performance Optimization Guide <../user_guide/performance_optimization>`
- :doc:`Example Notebooks <../user_guide/examples>`
- :doc:`Workflow Guides <../user_guide/workflows>`
