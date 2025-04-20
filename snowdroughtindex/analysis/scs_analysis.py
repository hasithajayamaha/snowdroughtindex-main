"""
SCS analysis module for the Snow Drought Index package.

This module contains functions for analyzing snow drought conditions
using the Snow Cover Seasonality (SCS) approach.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.cm as cm


def calculate_daily_mean_swe(swe_dataset):
    """
    Calculate daily mean SWE for a basin from an xarray dataset.
    
    Parameters
    ----------
    swe_dataset : xarray.Dataset
        Dataset containing SWE data with time dimension
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with Date and mean_SWE columns
    """
    # Convert to dataframe
    swe_df = swe_dataset.to_dataframe()
    
    # Calculate daily mean SWE
    daily_mean_swe = swe_df.groupby('time')['SWE'].mean().reset_index()
    
    # Rename columns for clarity
    daily_mean_swe.columns = ['Date', 'mean_SWE']
    
    return daily_mean_swe


def filter_points_within_shapefile(coordinates_df, shapefile_path, station_name=None):
    """
    Filter data points that fall within a shapefile boundary.
    
    Parameters
    ----------
    coordinates_df : pandas.DataFrame
        DataFrame containing coordinates (longitude, latitude) and subid
    shapefile_path : str
        Path to the shapefile
    station_name : str, optional
        Name of the station to filter in the shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing points within the shapefile
    """
    # Create geometry from coordinates
    geometry = [Point(lon, lat) for lon, lat in zip(coordinates_df['longitude'], coordinates_df['latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(coordinates_df, geometry=geometry, crs="EPSG:4326").reset_index()
    
    # Read shapefile
    shapefile = gpd.read_file(shapefile_path)
    
    # Filter shapefile by station name if provided
    if station_name:
        shapefile = shapefile[shapefile["Station_Na"] == station_name]
    
    # Select points within the shapefile
    points_within = gpd.sjoin(gdf, shapefile, how='inner', op='within')
    
    # Clean up the result
    if 'index_right' in points_within.columns:
        points_within = points_within.drop(columns=['index_right'])
    
    return points_within


def calculate_basin_mean_precipitation(precip_data, station_ids):
    """
    Calculate mean precipitation across selected stations.
    
    Parameters
    ----------
    precip_data : pandas.DataFrame
        DataFrame containing precipitation data with Date column and station columns
    station_ids : list
        List of station IDs to include in the mean calculation
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with Date and mean_precip columns
    """
    # Filter precipitation data to include only selected stations
    filtered_precip = precip_data[['Date'] + [col for col in precip_data.columns if col in station_ids]]
    
    # Calculate mean across all station columns
    filtered_precip['mean_precip'] = filtered_precip.iloc[:, 1:].mean(axis=1) * 10  # Convert to mm
    
    # Extract only Date and mean_precip columns
    mean_ppt = filtered_precip[['Date', 'mean_precip']]
    mean_ppt['Date'] = pd.to_datetime(mean_ppt['Date'])
    
    return mean_ppt


def merge_swe_precip_data(swe_data, precip_data):
    """
    Merge SWE and precipitation data on common dates.
    
    Parameters
    ----------
    swe_data : pandas.DataFrame
        DataFrame with Date and mean_SWE columns
    precip_data : pandas.DataFrame
        DataFrame with Date and mean_precip columns
        
    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with Date, mean_SWE, and mean_precip columns
    """
    # Merge on Date
    merged_data = pd.merge(swe_data, precip_data[['Date', 'mean_precip']], on='Date', how='inner')
    
    # Extract year and month-day for filtering
    merged_data['Year'] = merged_data['Date'].dt.year
    merged_data['Month-Day'] = merged_data['Date'].dt.strftime('%m-%d')
    
    return merged_data


def filter_snow_season(data, start_month=11, start_day=1, end_month=5, end_day=1):
    """
    Filter data for the snow season (November to May by default).
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with Date, Year, and Month-Day columns
    start_month : int, optional
        Start month of the snow season
    start_day : int, optional
        Start day of the snow season
    end_month : int, optional
        End month of the snow season
    end_day : int, optional
        End day of the snow season
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame for the snow season with season_year column added
    """
    # Define the period of interest
    season_filter = (data['Month-Day'] >= f'{start_month:02d}-{start_day:02d}') | (data['Month-Day'] <= f'{end_month:02d}-{end_day:02d}')
    
    # Filter the data
    filtered_data = data[season_filter].copy()
    
    # Adjust the year to group by winter season
    filtered_data['season_year'] = filtered_data['Year']
    filtered_data.loc[filtered_data['Month-Day'] <= f'{end_month:02d}-{end_day:02d}', 'season_year'] -= 1
    
    return filtered_data


def calculate_seasonal_means(data):
    """
    Calculate mean SWE and precipitation for each season.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with season_year, mean_SWE, and mean_precip columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with season_year and seasonal means for SWE and precipitation
    """
    # Calculate seasonal means
    seasonal_means = data.groupby('season_year')[['mean_SWE', 'mean_precip']].mean().reset_index()
    
    return seasonal_means


def filter_complete_seasons(data, swe_threshold=15, start_month=11, start_day=1, end_month=5, end_day=1):
    """
    Filter for complete snow seasons based on SWE threshold and date range.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with Date, mean_SWE, and Year columns
    swe_threshold : float, optional
        SWE threshold (mm) to define the start of the snow season
    start_month : int, optional
        Start month of the snow season
    start_day : int, optional
        Start day of the snow season
    end_month : int, optional
        End month of the snow season
    end_day : int, optional
        End day of the snow season
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing only complete snow seasons
    """
    # Add season_year column
    data['season_year'] = data['Date'].apply(lambda x: x.year if x.month >= start_month else x.year - 1)
    
    # Find the first date with threshold SWE each year
    season_starts = data[data['mean_SWE'] >= swe_threshold].groupby('season_year')['Date'].min()
    
    # Filter seasons
    filtered_seasons = []
    
    for year, start_date in season_starts.items():
        if start_date.month < start_month:
            continue  # Skip incomplete seasons at the beginning
            
        end_date = pd.Timestamp(year + 1, end_month, end_day)
        season_data = data[(data['Date'] >= start_date) & (data['Date'] < end_date)]
        
        # Check if season has data from start_date to end_date
        if not season_data.empty and season_data['Date'].max() >= end_date - pd.Timedelta(days=1):
            filtered_seasons.append(season_data)
    
    # Combine all complete seasons
    if filtered_seasons:
        complete_season_data = pd.concat(filtered_seasons, ignore_index=True)
        
        # Adjust the year to group by winter season
        complete_season_data['season_year'] = complete_season_data['Year']
        complete_season_data.loc[complete_season_data['Month-Day'] <= f'{end_month:02d}-{end_day:02d}', 'season_year'] -= 1
        
        return complete_season_data
    else:
        return pd.DataFrame()


def calculate_swe_p_ratio(data):
    """
    Calculate SWE/P ratio and cumulative precipitation.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with season_year, mean_SWE, and mean_precip columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added cumulative_P and SWE_P_ratio columns
    """
    # Calculate cumulative precipitation for each season
    data['cumulative_P'] = data.groupby('season_year')['mean_precip'].cumsum()
    
    # Calculate SWE/P ratio
    data['SWE_P_ratio'] = data['mean_SWE'] / data['cumulative_P']
    
    return data


def calculate_seasonal_metrics(data):
    """
    Calculate seasonal metrics including max SWE, mean SWE/P ratio, and mean cumulative precipitation.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with season_year, mean_SWE, SWE_P_ratio, and cumulative_P columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with seasonal metrics and anomalies
    """
    # Calculate metrics for each season
    seasonal_metrics = data.groupby('season_year').agg(
        SWEmax=('mean_SWE', 'max'),
        SWE_P_ratio=('SWE_P_ratio', 'mean'),
        cumulative_P=('cumulative_P', 'mean')
    ).reset_index()
    
    # Calculate precipitation anomaly
    seasonal_metrics['cum_P_anom'] = seasonal_metrics['cumulative_P'] - seasonal_metrics['cumulative_P'].mean()
    
    return seasonal_metrics


def standardize_metrics(data, ratio_max=1.0):
    """
    Standardize metrics for clustering.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with SWE_P_ratio and cum_P_anom columns
    ratio_max : float, optional
        Maximum value for SWE_P_ratio to filter outliers
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized metrics
    """
    # Filter data if needed
    if ratio_max < 1.0:
        filtered_data = data[data['SWE_P_ratio'] <= ratio_max].copy()
    else:
        filtered_data = data.copy()
    
    # Standardize metrics
    filtered_data['cum_P_anom_z'] = (filtered_data['cum_P_anom'] - filtered_data['cum_P_anom'].mean()) / filtered_data['cum_P_anom'].std()
    filtered_data['SWE_P_ratio_z'] = (filtered_data['SWE_P_ratio'] - filtered_data['SWE_P_ratio'].mean()) / filtered_data['SWE_P_ratio'].std()
    
    return filtered_data


def classify_snow_drought(data, n_clusters=3, random_state=0):
    """
    Classify snow drought types using K-means clustering.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with standardized metrics (SWE_P_ratio_z and cum_P_anom_z)
    n_clusters : int, optional
        Number of clusters for K-means
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with cluster assignments and names
    """
    # Extract features for clustering
    cluster_features = data[['SWE_P_ratio_z', 'cum_P_anom_z']]
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    data['cluster'] = kmeans.fit_predict(cluster_features)
    
    # Analyze cluster centers to assign meaningful names
    centers = kmeans.cluster_centers_
    
    # Default cluster labels
    cluster_labels = {i: f'Cluster {i}' for i in range(n_clusters)}
    
    # Assign names based on cluster characteristics
    for i, center in enumerate(centers):
        swe_p_ratio_z, cum_p_anom_z = center
        
        if cum_p_anom_z < -0.5 and swe_p_ratio_z < -0.5:
            cluster_labels[i] = 'Warm & Dry'
        elif cum_p_anom_z < -0.5 and swe_p_ratio_z >= -0.5:
            cluster_labels[i] = 'Dry'
        elif cum_p_anom_z >= -0.5 and swe_p_ratio_z < -0.5:
            cluster_labels[i] = 'Warm'
        else:
            cluster_labels[i] = 'Normal'
    
    # Map cluster numbers to names
    data['cluster_name'] = data['cluster'].map(cluster_labels)
    
    return data, kmeans.cluster_centers_, cluster_labels


def plot_seasonal_swe_precip(seasonal_means):
    """
    Plot seasonal mean SWE vs precipitation.
    
    Parameters
    ----------
    seasonal_means : pandas.DataFrame
        DataFrame with season_year, mean_SWE, and mean_precip columns
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with colors based on years
    scatter = ax.scatter(
        seasonal_means['mean_precip'],
        seasonal_means['mean_SWE'],
        c=seasonal_means['season_year'],
        cmap='tab20_r'
    )
    
    # Add trend line
    z = np.polyfit(seasonal_means['mean_precip'], seasonal_means['mean_SWE'], 1)
    p = np.poly1d(z)
    ax.plot(seasonal_means['mean_precip'], p(seasonal_means['mean_precip']), "r--")
    
    # Add labels and legend
    ax.set_xlabel('Precipitation (Seasonal Mean)')
    ax.set_ylabel('SWE (Seasonal Mean)')
    ax.set_title('Seasonal Mean SWE vs Precipitation')
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(num=16), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig


def plot_snow_drought_classification(data, cluster_colors=None):
    """
    Plot snow drought classification based on SWE/P ratio and precipitation anomaly.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with cum_P_anom, SWE_P_ratio, and cluster_name columns
    cluster_colors : dict, optional
        Dictionary mapping cluster names to colors
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define default colors if not provided
    if cluster_colors is None:
        cluster_colors = {
            'Warm': 'red',
            'Dry': 'blue',
            'Warm & Dry': 'purple',
            'Normal': 'grey'
        }
    
    # Create scatter plot
    for cluster_name, color in cluster_colors.items():
        cluster_data = data[data['cluster_name'] == cluster_name]
        ax.scatter(
            cluster_data['cum_P_anom'],
            cluster_data['SWE_P_ratio'],
            c=color,
            label=cluster_name
        )
    
    # Add labels and legend
    ax.set_xlabel('Cumulative Precipitation Anomaly (mm)')
    ax.set_ylabel('SWE/P Ratio')
    ax.set_ylim(bottom=0)
    ax.set_title('K-means Clustering of Seasonal Data')
    ax.legend(title='Cluster')
    
    plt.tight_layout()
    
    return fig


def plot_drought_time_series(data, metric, cluster_colors=None, year_range=None):
    """
    Plot time series of drought metrics with cluster coloring.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with season_year, cluster_name, and the specified metric
    metric : str
        Column name of the metric to plot
    cluster_colors : dict, optional
        Dictionary mapping cluster names to colors
    year_range : tuple, optional
        (min_year, max_year) to limit the x-axis range
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define default colors if not provided
    if cluster_colors is None:
        cluster_colors = {
            'Warm': 'red',
            'Dry': 'blue',
            'Warm & Dry': 'purple',
            'Normal': 'grey'
        }
    
    # Ensure season_year is integer
    data['season_year'] = data['season_year'].astype(int)
    
    # Create bar plot
    bars = ax.bar(
        data['season_year'],
        data[metric],
        color=data['cluster_name'].map(cluster_colors)
    )
    
    # Add labels
    ax.set_xlabel('Year')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Year')
    
    # Add horizontal line at zero
    ax.axhline(color='black')
    
    # Set x-axis range if specified
    if year_range:
        ax.set_xlim(year_range)
    
    # Add legend
    for name, color in cluster_colors.items():
        ax.bar(0, 0, color=color, label=name)
    ax.legend(title='Cluster')
    
    plt.tight_layout()
    
    return fig


def run_scs_analysis(swe_data, precip_data, station_ids, swe_threshold=15, n_clusters=3):
    """
    Run the complete SCS analysis workflow.
    
    Parameters
    ----------
    swe_data : pandas.DataFrame
        DataFrame with Date and mean_SWE columns
    precip_data : pandas.DataFrame
        DataFrame with Date and precipitation columns for stations
    station_ids : list
        List of station IDs to include in the analysis
    swe_threshold : float, optional
        SWE threshold (mm) to define the start of the snow season
    n_clusters : int, optional
        Number of clusters for K-means
        
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Calculate basin mean precipitation
    mean_precip = calculate_basin_mean_precipitation(precip_data, station_ids)
    
    # Merge SWE and precipitation data
    merged_data = merge_swe_precip_data(swe_data, mean_precip)
    
    # Filter for complete snow seasons
    complete_seasons = filter_complete_seasons(merged_data, swe_threshold=swe_threshold)
    
    # Calculate SWE/P ratio and cumulative precipitation
    complete_seasons = calculate_swe_p_ratio(complete_seasons)
    
    # Calculate seasonal metrics
    seasonal_metrics = calculate_seasonal_metrics(complete_seasons)
    
    # Standardize metrics for clustering
    standardized_metrics = standardize_metrics(seasonal_metrics, ratio_max=1.0)
    
    # Classify snow drought types
    classified_data, cluster_centers, cluster_labels = classify_snow_drought(
        standardized_metrics, n_clusters=n_clusters
    )
    
    # Calculate ratio anomaly
    classified_data['ratio_anomaly'] = classified_data['SWE_P_ratio'] - classified_data['SWE_P_ratio'].mean()
    
    # Calculate peak SWE anomaly
    classified_data['peak_SWE_anomaly'] = classified_data['SWEmax'] - classified_data['SWEmax'].mean()
    
    # Return results
    return {
        'complete_seasons': complete_seasons,
        'seasonal_metrics': seasonal_metrics,
        'classified_data': classified_data,
        'cluster_centers': cluster_centers,
        'cluster_labels': cluster_labels
    }
