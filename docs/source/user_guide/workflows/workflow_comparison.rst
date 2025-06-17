Full Workflow Comparison
=====================

This guide explains how to compare different snow drought classification methods using the Snow Drought Index package, specifically comparing the Standardized Snow Water Equivalent Index (SWEI) with the Heldmyer et al. drought classification approach.

Overview
--------

The Full Workflow Comparison allows you to:

1. Process and analyze SWE and precipitation data for a watershed
2. Calculate the Standardized SWEI for drought classification
3. Apply the Heldmyer et al. classification method using K-means clustering
4. Compare results between different classification approaches
5. Visualize drought patterns and classifications

This workflow is particularly useful for:

- Comparing different drought classification methodologies
- Conducting comprehensive watershed-scale drought analysis
- Identifying different types of snow drought (warm, dry, or warm & dry)
- Visualizing temporal patterns in snow drought occurrence

Prerequisites
------------

- Snow Water Equivalent (SWE) data for your study area
- Precipitation data for the same area and time period
- Watershed boundary shapefile
- Python packages: numpy, pandas, xarray, geopandas, matplotlib, scipy, scikit-learn

Workflow Steps
-------------

1. Data Loading and Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load the required datasets and prepare them for analysis:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from scipy.integrate import trapz
    from scipy.stats import norm
    
    # Import snowdroughtindex package
    from snowdroughtindex.core import data_preparation, gap_filling, sswei
    from snowdroughtindex.utils import visualization
    
    # Load data
    SWE_path = 'path/to/SWE_data.csv'
    P_path = 'path/to/precipitation_data.nc'
    basin_shapefile = 'path/to/basin_shapefile.shp'
    
    # Load SWE data
    SWE = pd.read_csv(SWE_path)
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(SWE['lon'], SWE['lat'])]
    SWE_gdf = gpd.GeoDataFrame(SWE, crs={'init': 'epsg:4326'}, geometry=geometry)
    
    # Load precipitation data
    P = xr.open_dataset(P_path)
    P = P.to_dataframe().reset_index()
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(P['lon'], P['lat'])]
    P_data = P.drop(['lon', 'lat'], axis=1)
    P_data = gpd.GeoDataFrame(P_data, crs={'init': 'epsg:4326'}, geometry=geometry)
    
    # Load basin shapefile
    basin = gpd.read_file(basin_shapefile)

2. Visualize Data Points
~~~~~~~~~~~~~~~~~~~~~~

Visualize the spatial distribution of data points within your watershed:

.. code-block:: python

    # Plot data points on the shapefile
    fig, ax = plt.subplots(figsize=(10, 10))    
    basin.plot(ax=ax, color='lightgrey', edgecolor='black')
    SWE_gdf.plot(ax=ax, color='blue', markersize=10, label='SWE')
    P_data.plot(ax=ax, color='red', markersize=5, label='Precipitation')
    ax.set_title('Grid Cells within the Basin')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.savefig('basin_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

3. Calculate Climatological Mean SWE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the climatological mean SWE for your reference period:

.. code-block:: python

    # Calculate basin mean SWE
    SWE_basin_mean = SWE.groupby('time').mean().reset_index()
    
    # Convert time to datetime
    SWE_basin_mean['time'] = pd.to_datetime(SWE_basin_mean['time'])
    
    # Get peak SWE for each year in reference period (e.g., 1981-2010)
    peak_SWE = SWE_basin_mean[SWE_basin_mean['time'].dt.year.isin(range(1981, 2011))].groupby(
        SWE_basin_mean['time'].dt.year)['SWE'].max().reset_index()
    
    # Calculate average peak SWE
    average_peak_SWE = peak_SWE['peak_SWE'].mean()
    
    # Define minimum SWE threshold (5% of average peak SWE)
    min_SWE = 0.05 * average_peak_SWE

4. Prepare Data for Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the data for snow drought analysis:

.. code-block:: python

    # Add coordinate IDs to SWE and precipitation data
    SWE['coordinate_id'] = SWE.groupby(['geometry']).ngroup() + 1
    SWE = SWE[['time', 'coordinate_id', 'SWE']]
    
    P_data = P_data[['time', 'coordinate_id', 'daily_precipitation']]
    
    # Merge SWE and precipitation data
    SWE_P = pd.merge(SWE, P_data, on=['time', 'coordinate_id'], how='inner')
    
    # Define water years
    SWE_P['time'] = pd.to_datetime(SWE_P['time'])
    water_year_grid = SWE_P[(SWE_P['time'].dt.month >= 10) | (SWE_P['time'].dt.month <= 9)]
    water_year_grid['season_year'] = water_year_grid['time'].dt.year
    water_year_grid['season_year'] = water_year_grid['season_year'].where(
        water_year_grid['time'].dt.month >= 10, 
        water_year_grid['season_year'] - 1
    )
    
    # Calculate daily SWE change
    SWE_P_rearranged = water_year_grid[['time', 'coordinate_id', 'SWE', 'daily_precipitation', 'season_year']]
    SWE_P_rearranged.columns = ['time', 'coordinate_id', 'SWE', 'P', 'season_year']
    
    SWE_P_rearranged['daily_SWE_change'] = SWE_P_rearranged.groupby(['coordinate_id', 'season_year'])['SWE'].diff().shift(-1)
    SWE_P_rearranged['daily_SWE_change'] = SWE_P_rearranged['daily_SWE_change'].fillna(0)
    SWE_P_rearranged.loc[SWE_P_rearranged['daily_SWE_change'] < 0, 'daily_SWE_change'] = 0

5. Extract Snow Accumulation Period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract the snow accumulation period (from onset to peak SWE) for each year:

.. code-block:: python

    # Initialize an empty DataFrame
    onset_to_peak = pd.DataFrame()
    
    # Iterate through each coordinate_id and season year
    for coordinate_id in SWE_P_rearranged['coordinate_id'].unique():
        coord_data = SWE_P_rearranged[SWE_P_rearranged['coordinate_id'] == coordinate_id]
        
        for year in coord_data['season_year'].unique():
            season_data = coord_data[coord_data['season_year'] == year]
            
            # Find onset date (when SWE exceeds minimum threshold)
            onset_date = season_data[season_data['SWE'] >= min_SWE]['time'].min()
            
            # Find peak date
            peak_date = season_data[season_data['SWE'] == season_data['SWE'].max()]['time'].values[0]
            
            # Select data from onset to peak
            selected_data = season_data[(season_data['time'] >= onset_date) & (season_data['time'] <= peak_date)]
            
            # Append to result DataFrame
            onset_to_peak = pd.concat([onset_to_peak, selected_data])

6. Calculate Standardized SWEI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate the Standardized Snow Water Equivalent Index (SWEI):

.. code-block:: python

    # Perturb zeros
    onset_to_peak['pertub_SWE'] = sswei.perturb_zeros(onset_to_peak['daily_SWE_change'])
    
    # Define integration function
    def integrate_season(group):
        group = group.sort_values(by='time')
        days_since_start = (group['time'] - group['time'].min()).dt.days
        total_swe_integration = trapz(group['SWE'], days_since_start)
        
        return pd.Series({
            'coordinate_id': group['coordinate_id'].iloc[0],
            'season_year': group['season_year'].iloc[0],
            'total_SWE_integration': total_swe_integration
        })
    
    # Integrate SWE over each season
    SWE_integration = onset_to_peak.groupby(['coordinate_id', 'season_year']).apply(integrate_season).reset_index(drop=True)
    
    # Calculate Gringorten plotting positions
    SWE_integration['gringorten_probability'] = SWE_integration.groupby('coordinate_id')['total_SWE_integration'].transform(sswei.gringorten_probabilities)
    
    # Compute SSWEI
    SWE_integration['SSWEI'] = SWE_integration.groupby('coordinate_id')['gringorten_probability'].transform(sswei.compute_swei)
    
    # Calculate average SSWEI per year
    average_SSWEI_per_year = SWE_integration.groupby('season_year')['SSWEI'].mean().reset_index()
    
    # Classify drought based on SSWEI
    average_SSWEI_per_year['Classification'] = average_SSWEI_per_year['SSWEI'].apply(sswei.classify_drought)

7. Visualize SWEI Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a time series plot of the SWEI values:

.. code-block:: python

    # Extract necessary columns
    plot_data = average_SSWEI_per_year[['season_year', 'SSWEI', 'Classification']]
    
    # Create plot
    plt.figure(figsize=(15, 6))
    plt.plot(plot_data['season_year'], plot_data['SSWEI'], marker='o', label='SSWEI', color='black')
    
    # Add thresholds for drought classifications
    plt.axhline(-2.0, color='r', linestyle='--', label='Exceptional Drought Threshold')
    plt.axhline(-1.5, color='orange', linestyle='--', label='Extreme Drought Threshold')
    plt.axhline(-1.0, color='yellow', linestyle='--', label='Severe Drought Threshold')
    plt.axhline(-0.5, color='gray', linestyle='--', label='Near Normal Threshold')
    
    # Customize the plot
    plt.title('SWEI Trends by Season Year')
    plt.xlabel('Season Year')
    plt.ylabel('Standardized SWEI')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.savefig('SWEI_trends.png')
    plt.show()

8. Apply Heldmyer Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply the Heldmyer et al. classification method using K-means clustering:

.. code-block:: python

    # Calculate cumulative precipitation
    onset_to_peak['cumulative_P'] = onset_to_peak.groupby(['season_year', 'coordinate_id'])['P'].cumsum()
    
    # Calculate statistics for each coordinate_id and season
    onset_to_peak_mean_all = pd.DataFrame()
    
    for coordinate_id in onset_to_peak['coordinate_id'].unique():
        coord_data = onset_to_peak[onset_to_peak['coordinate_id'] == coordinate_id]
        
        # Calculate statistics
        onset_to_peak_mean = coord_data.groupby('season_year').agg(
            mean_SWE=('SWE', 'mean'),
            mean_cumulative_P=('cumulative_P', 'mean'),
            max_SWE=('SWE', 'max'),
            max_cumulative_P=('cumulative_P', 'max')
        ).reset_index()
        
        # Calculate anomalies and ratios
        onset_to_peak_mean['cumulative_P_anomaly'] = onset_to_peak_mean['mean_cumulative_P'] - onset_to_peak_mean['mean_cumulative_P'].mean()
        onset_to_peak_mean['mean_SWE/cumulative_P'] = onset_to_peak_mean['mean_SWE'] / onset_to_peak_mean['max_cumulative_P']
        onset_to_peak_mean['coordinate_id'] = coordinate_id
        
        # Append to result
        onset_to_peak_mean_all = pd.concat([onset_to_peak_mean_all, onset_to_peak_mean])
    
    # Calculate mean over all coordinate_ids
    mean_over_coordinate_id = onset_to_peak_mean_all.groupby('season_year').mean().reset_index()
    
    # Identify snow drought years (max_SWE < mean max_SWE)
    snow_drought_years = mean_over_coordinate_id[mean_over_coordinate_id['max_SWE'] < mean_over_coordinate_id['max_SWE'].mean()]
    normal_years = mean_over_coordinate_id[mean_over_coordinate_id['max_SWE'] > mean_over_coordinate_id['max_SWE'].mean()]

9. Apply K-means Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply K-means clustering to classify snow drought types:

.. code-block:: python

    from sklearn.cluster import KMeans
    
    # Standardize features
    snow_drought_years['cumulative_P_anomaly_z'] = (snow_drought_years['cumulative_P_anomaly'] - snow_drought_years['cumulative_P_anomaly'].mean()) / snow_drought_years['cumulative_P_anomaly'].std()
    snow_drought_years['mean_SWE/cumulative_P_z'] = (snow_drought_years['mean_SWE/cumulative_P'] - snow_drought_years['mean_SWE/cumulative_P'].mean()) / snow_drought_years['mean_SWE/cumulative_P'].std()
    
    # Apply K-means clustering
    cluster_feature = snow_drought_years[['mean_SWE/cumulative_P_z', 'cumulative_P_anomaly_z']]
    kmean = KMeans(n_clusters=3, random_state=0)
    snow_drought_years['cluster'] = kmean.fit_predict(cluster_feature)
    
    # Assign names to clusters
    cluster_labels = {
        0: 'Warm',
        1: 'Warm & Dry',
        2: 'Dry'
    }
    
    snow_drought_years['cluster_name'] = snow_drought_years['cluster'].map(cluster_labels)

10. Visualize Drought Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a scatter plot to visualize the different drought types:

.. code-block:: python

    plt.figure(figsize=(10, 6))
    
    # Define colors for each cluster
    cluster_colors = {'Warm': 'red', 'Dry': 'blue', 'Warm & Dry': 'purple'}
    
    # Create scatter plot
    plt.scatter(snow_drought_years['cumulative_P_anomaly'], snow_drought_years['mean_SWE/cumulative_P'], 
                c=snow_drought_years['cluster_name'].map(cluster_colors))
    plt.scatter(normal_years['cumulative_P_anomaly'], normal_years['mean_SWE/cumulative_P'], 
                c='grey', label='Normal')
    
    # Add labels and annotations
    plt.xlabel('Cumulative Precipitation Anomaly (mm)')
    plt.ylabel('SWE/P Ratio')
    plt.axvline(0, color='black', linewidth=1.5, linestyle='-')
    
    # Add year labels to each point
    for i, row in snow_drought_years.iterrows():
        plt.annotate(row['season_year'], (row['cumulative_P_anomaly'], row['mean_SWE/cumulative_P']), 
                     fontsize=8, color='black', alpha=0.7)
    
    # Create legend
    for name, color in cluster_colors.items():
        plt.bar(0, 0, color=color, label=name)
    
    plt.legend(title='Cluster')
    plt.title('K-means Clustering of Snow Drought (K=3)')
    plt.grid(True)
    plt.savefig('Kmeans_Clustering.png')
    plt.show()

Complete Example
---------------

A complete example notebook is available in the package repository:

``notebooks/workflows/3_Full_workflow_comparison.ipynb``

This notebook demonstrates the full workflow with example data.

Comparison of Classification Methods
-----------------------------------

The workflow allows you to compare two different snow drought classification approaches:

1. **Standardized SWEI Classification**:
   - Based on statistical distribution of SWE values
   - Provides a continuous index (SWEI) that can be categorized into drought severity classes
   - Focuses on the statistical rarity of SWE conditions

2. **Heldmyer et al. Classification**:
   - Uses K-means clustering to identify different types of snow drought
   - Distinguishes between warm, dry, and warm & dry snow droughts
   - Considers both SWE and precipitation anomalies

Benefits of this comparison include:

- Understanding different drought mechanisms (precipitation deficit vs. temperature effects)
- Identifying which classification method is most appropriate for your region
- Gaining insights into the temporal patterns of different drought types
- Developing more targeted drought monitoring and management strategies

Next Steps
---------

After completing this workflow, you can:

- Analyze the frequency of different snow drought types in your region
- Investigate the relationship between drought types and climate variables
- Develop region-specific drought monitoring approaches
- Create custom visualizations for drought communication
