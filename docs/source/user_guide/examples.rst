Examples
========

This section provides examples of how to use the Snow Drought Index package for various analyses.

Basic SSWEI Calculation
----------------------

This example shows how to calculate the Standardized Snow Water Equivalent Index (SSWEI) for a dataset:

.. code-block:: python

   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   from snowdroughtindex.core.data_preparation import load_swe_data
   from snowdroughtindex.core.sswei import integrate_season, gringorten_probabilities, compute_swei
   from snowdroughtindex.core.drought_classification import classify_drought
   from snowdroughtindex.utils.visualization import plot_sswei_timeseries
   
   # Load SWE data
   swe_data = load_swe_data('path/to/swe_data.nc')
   
   # Extract daily mean SWE for the basin
   daily_mean = swe_data.groupby('time')['SWE'].mean().reset_index()
   daily_mean.columns = ['date', 'mean_SWE']
   
   # Define season parameters
   start_month, start_day = 11, 1  # Start in November
   end_month, end_day = 4, 30      # End in April
   
   # Find the first date with 15 mm SWE each year to set as season start
   daily_mean['season_year'] = daily_mean['date'].apply(lambda x: x.year if x.month >= start_month else x.year - 1)
   season_starts = daily_mean[daily_mean['mean_SWE'] >= 15].groupby('season_year')['date'].min()
   
   # Filter seasons based on season start and ensure they run through to April 30th of the next year
   filtered_seasons = []
   
   for year, start_date in season_starts.items():
       if start_date.month < start_month:
           continue  # Skip incomplete seasons at the beginning
       
       end_date = pd.Timestamp(year + 1, end_month, end_day)
       season_data = daily_mean[(daily_mean['date'] >= start_date) & (daily_mean['date'] <= end_date)]
       
       # Check if season has data from start_date to end_date
       if not season_data.empty and season_data['date'].max() >= end_date - pd.Timedelta(days=1):
           filtered_seasons.append(season_data)
   
   # Combine all complete seasons
   season_data = pd.concat(filtered_seasons, ignore_index=True)
   
   # Integrate SWE over each season
   integrated_data = season_data.groupby('season_year').apply(integrate_season).reset_index()
   
   # Calculate Gringorten probabilities
   integrated_data['Gringorten_probabilities'] = gringorten_probabilities(integrated_data['total_SWE_integration'])
   
   # Compute SSWEI
   integrated_data['SSWEI'] = compute_swei(integrated_data['Gringorten_probabilities'])
   
   # Classify drought conditions
   integrated_data['Drought_Classification'] = integrated_data['SSWEI'].apply(classify_drought)
   
   # Plot SSWEI trends
   plt.figure(figsize=(10, 6))
   plt.plot(integrated_data['season_year'], integrated_data['SSWEI'], marker='o', label='SSWEI', color='black')
   
   # Add thresholds for drought classifications
   plt.axhline(-2.0, color='r', linestyle='--', label='Exceptional Drought Threshold')
   plt.axhline(-1.5, color='orange', linestyle='--', label='Extreme Drought Threshold')
   plt.axhline(-1.0, color='yellow', linestyle='--', label='Severe Drought Threshold')
   plt.axhline(-0.5, color='gray', linestyle='--', label='Near Normal Threshold')
   
   plt.title('SSWEI Trends by Season Year')
   plt.xlabel('Season Year')
   plt.ylabel('Standardized SSWEI')
   plt.legend()
   plt.grid()
   plt.show()

Using the SWEDataset Class
-------------------------

This example shows how to use the SWEDataset class for more complex analyses:

.. code-block:: python

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
   
   # Fill gaps in the data
   dataset.fill_gaps()
   
   # Create an SSWEI object
   sswei = SSWEI(dataset)
   
   # Calculate SSWEI
   sswei.calculate()
   
   # Classify drought conditions
   sswei.classify_drought()
   
   # Create a DroughtAnalysis object
   analysis = DroughtAnalysis(sswei)
   
   # Analyze drought conditions by elevation bands
   analysis.analyze_elevation_bands()
   
   # Analyze drought trends
   analysis.analyze_trends()
   
   # Visualize results
   analysis.plot_drought_trends()
   analysis.plot_elevation_analysis()
   
   # Export results
   analysis.export_results('output_directory')

Snow-Climate Sensitivity Analysis
--------------------------------

This example shows how to perform a snow-climate sensitivity analysis:

.. code-block:: python

   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import geopandas as gpd
   from sklearn.cluster import KMeans
   
   from snowdroughtindex.core.data_preparation import load_swe_data, filter_by_basin
   from snowdroughtindex.analysis.scs_analysis import calculate_swe_precip_ratio, cluster_snow_drought
   from snowdroughtindex.utils.visualization import plot_scatter, plot_clusters
   
   # Load SWE and precipitation data
   swe_data = load_swe_data('path/to/swe_data.nc')
   precip_data = pd.read_csv('path/to/precip_data.csv')
   
   # Filter data for a specific basin
   basin_shapefile = 'path/to/basin_shapefile.shp'
   basin_data = filter_by_basin(swe_data, basin_shapefile, basin_name='Basin Name')
   
   # Calculate daily and seasonal means
   daily_mean_swe = basin_data.groupby('time')['SWE'].mean().reset_index()
   daily_mean_precip = precip_data.groupby('Date')['Precipitation'].mean().reset_index()
   
   # Merge SWE and precipitation data
   merged_data = pd.merge(daily_mean_swe, daily_mean_precip, left_on='time', right_on='Date', how='inner')
   
   # Calculate SWE/P ratio and cumulative precipitation
   merged_data = calculate_swe_precip_ratio(merged_data)
   
   # Standardize data for clustering
   merged_data['cum_P_anom_z'] = (merged_data['cum_P_anom'] - merged_data['cum_P_anom'].mean()) / merged_data['cum_P_anom'].std()
   merged_data['SWE_P_ratio_z'] = (merged_data['SWE_P_ratio'] - merged_data['SWE_P_ratio'].mean()) / merged_data['SWE_P_ratio'].std()
   
   # Perform clustering
   clusters, centers = cluster_snow_drought(merged_data, ['SWE_P_ratio_z', 'cum_P_anom_z'], n_clusters=3)
   merged_data['cluster'] = clusters
   
   # Assign cluster names
   cluster_names = {0: 'Dry', 1: 'Warm', 2: 'Warm & Dry'}
   merged_data['cluster_name'] = merged_data['cluster'].map(cluster_names)
   
   # Plot results
   plot_clusters(merged_data, x='cum_P_anom', y='SWE_P_ratio', color_col='cluster_name')
