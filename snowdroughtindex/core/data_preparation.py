"""
Data preparation module for the Snow Drought Index package.

This module contains functions for loading, cleaning, and preprocessing data,
as well as station extraction and filtering.
"""

from typing import Union, List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon

def load_swe_data(file_path: str) -> xr.Dataset:
    """
    Load SWE data from a NetCDF file.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file containing SWE data.
        
    Returns
    -------
    xarray.Dataset
        Dataset containing SWE data.
    """
    return xr.open_dataset(file_path)

def load_precip_data(file_path: str) -> xr.Dataset:
    """
    Load precipitation data from a NetCDF file.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file containing precipitation data.
        
    Returns
    -------
    xarray.Dataset
        Dataset containing precipitation data.
    """
    return xr.open_dataset(file_path)

def load_basin_data(file_path: str) -> gpd.GeoDataFrame:
    """
    Load basin shapefile data.
    
    Parameters
    ----------
    file_path : str
        Path to the shapefile containing basin data.
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing basin data.
    """
    return gpd.read_file(file_path)

def preprocess_swe(swe_data: xr.Dataset) -> pd.DataFrame:
    """
    Preprocess SWE data by converting to a DataFrame and adding necessary metadata.
    
    Parameters
    ----------
    swe_data : xarray.Dataset
        Dataset containing SWE data.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing preprocessed SWE data.
    """
    # Convert to DataFrame
    swe_df = swe_data.to_dataframe()
    
    # Reset index to make coordinates accessible as columns
    if isinstance(swe_df.index, pd.MultiIndex):
        swe_df = swe_df.reset_index()
    
    return swe_df

def preprocess_precip(precip_data: xr.Dataset) -> pd.DataFrame:
    """
    Preprocess precipitation data by converting to a DataFrame and adding necessary metadata.
    
    Parameters
    ----------
    precip_data : xarray.Dataset
        Dataset containing precipitation data.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing preprocessed precipitation data.
    """
    # Convert to DataFrame
    precip_df = precip_data.to_dataframe()
    
    # Reset index to make coordinates accessible as columns
    if isinstance(precip_df.index, pd.MultiIndex):
        precip_df = precip_df.reset_index()
    
    # Convert time to datetime if it's not already
    if 'time' in precip_df.columns:
        precip_df['time'] = pd.to_datetime(precip_df['time'])
    
    return precip_df

def extract_stations_in_basin(
    stations: gpd.GeoDataFrame,
    basin_shapefile: gpd.GeoDataFrame,
    basin_id: str,
    buffer_km: float = 0
) -> Tuple[gpd.GeoDataFrame, Union[gpd.GeoSeries, int]]:
    """
    Extract stations within a specified basin (with or without a buffer).
    
    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        GeoDataFrame containing station data with geometry.
    basin_shapefile : geopandas.GeoDataFrame
        GeoDataFrame containing basin shapefile data.
    basin_id : str
        ID of the basin to extract stations from.
    buffer_km : float, optional
        Buffer distance in kilometers around the basin, by default 0.
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing stations within the basin.
    shapely.geometry.Polygon or int
        Buffer geometry if buffer_km > 0, otherwise 0.
    """
    # Extract stations within basin only (i.e., no buffer)
    if buffer_km == 0:
        basin_buffer = 0
        mask = stations.within(basin_shapefile.loc[basin_shapefile['Station_ID'] == basin_id].iloc[0].loc["geometry"])

    # Extract stations within specified buffer of basin
    elif buffer_km > 0:
        # Convert basin & stations geometry to a different CRS to be able to add a buffer in meters
        basin_crs_conversion = basin_shapefile.loc[basin_shapefile['Station_ID'] == basin_id].to_crs(epsg=3763)
        stations_crs_conversion = stations.to_crs(epsg=3763)

        # Add a buffer in meters around the basin
        buffer_m = buffer_km * 1000
        basin_buffer = basin_crs_conversion.buffer(buffer_m)
        mask = stations_crs_conversion.within(basin_buffer.iloc[0])

        # Convert the buffer back to the original CRS for plotting
        basin_buffer = basin_buffer.to_crs(epsg=4326)

    stations_in_basin = stations.loc[mask].assign(basin=basin_id)

    return stations_in_basin, basin_buffer

def filter_stations(
    data: Union[pd.DataFrame, xr.Dataset],
    stations_list: List[str]
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Filter data to include only specified stations.
    
    Parameters
    ----------
    data : pandas.DataFrame or xarray.Dataset
        Data to filter.
    stations_list : list
        List of station IDs to include.
        
    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Filtered data containing only the specified stations.
    """
    if isinstance(data, pd.DataFrame):
        if 'station_id' in data.columns:
            return data[data['station_id'].isin(stations_list)]
        else:
            raise ValueError("DataFrame does not contain 'station_id' column")
    elif isinstance(data, xr.Dataset):
        return data.sel(station_id=stations_list)
    else:
        raise TypeError("Data must be a pandas DataFrame or xarray Dataset")

def assess_data_availability(
    data: xr.Dataset,
    time_dim: str = 'time',
    station_dim: str = 'station_id'
) -> xr.DataArray:
    """
    Assess the availability of data for each station over time.
    
    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing data to assess.
    time_dim : str, optional
        Name of the time dimension, by default 'time'.
    station_dim : str, optional
        Name of the station dimension, by default 'station_id'.
        
    Returns
    -------
    xarray.DataArray
        DataArray containing the percentage of available data for each station over time.
    """
    # Count non-NaN values along the time dimension
    counts = data.count(dim=time_dim)
    
    # Calculate the total number of time points
    total_times = len(data[time_dim])
    
    # Calculate the percentage of available data
    availability = (counts / total_times) * 100
    
    return availability

def convert_to_geodataframe(
    df: pd.DataFrame,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    crs: str = 'epsg:4326'
) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with longitude and latitude columns to a GeoDataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing longitude and latitude columns.
    lon_col : str, optional
        Name of the longitude column, by default 'lon'.
    lat_col : str, optional
        Name of the latitude column, by default 'lat'.
    crs : str, optional
        Coordinate reference system, by default 'epsg:4326'.
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with geometry created from longitude and latitude.
    """
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf

def spatial_join(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    how: str = 'inner',
    op: str = 'intersects'
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between points and polygons.
    
    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries.
    polygons_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries.
    how : str, optional
        Type of join, by default 'inner'.
    op : str, optional
        Spatial operation to use, by default 'intersects'.
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the result of the spatial join.
    """
    return gpd.sjoin(points_gdf, polygons_gdf, how=how, op=op)

def update_coordinates(
    data: Union[xr.Dataset, pd.DataFrame],
    coordinates_df: pd.DataFrame,
    id_col: str = 'station_id',
    lat_col: str = 'New_Lat',
    lon_col: str = 'New_Lon'
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Update station coordinates in a dataset.
    
    Parameters
    ----------
    data : xarray.Dataset or pandas.DataFrame
        Dataset containing station data.
    coordinates_df : pandas.DataFrame
        DataFrame containing updated coordinates.
    id_col : str, optional
        Name of the station ID column in coordinates_df, by default 'station_id'.
    lat_col : str, optional
        Name of the latitude column in coordinates_df, by default 'New_Lat'.
    lon_col : str, optional
        Name of the longitude column in coordinates_df, by default 'New_Lon'.
        
    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        Dataset with updated coordinates.
    """
    if isinstance(data, pd.DataFrame):
        # Create a mapping for quick lookup from the coordinates DataFrame
        coord_map = coordinates_df.set_index(id_col)[[lat_col, lon_col]].to_dict('index')
        
        # Update coordinates in the DataFrame
        for idx, row in data.iterrows():
            if row['station_id'] in coord_map:
                data.at[idx, 'lat'] = coord_map[row['station_id']][lat_col]
                data.at[idx, 'lon'] = coord_map[row['station_id']][lon_col]
                
        return data
    
    elif isinstance(data, xr.Dataset):
        # Create a mapping for quick lookup from the coordinates DataFrame
        coord_map = coordinates_df.set_index(id_col)[[lat_col, lon_col]].to_dict('index')
        
        # Get station IDs from the dataset
        station_ids = data['station_id'].values
        
        # Update lat/lon in the dataset
        updated_lat = data['lat'].values.copy()
        updated_lon = data['lon'].values.copy()
        
        for i, sid in enumerate(station_ids):
            if sid in coord_map:
                if len(data['lat'].dims) == 1:  # 1D array
                    updated_lat[i] = coord_map[sid][lat_col]
                    updated_lon[i] = coord_map[sid][lon_col]
                else:  # 2D array
                    updated_lat[i, :] = coord_map[sid][lat_col]
                    updated_lon[i, :] = coord_map[sid][lon_col]
        
        # Assign the updated coordinates back to the dataset
        data['lat'] = (data['lat'].dims, updated_lat)
        data['lon'] = (data['lon'].dims, updated_lon)
        
        return data
    
    else:
        raise TypeError("Data must be a pandas DataFrame or xarray Dataset")

def filter_data_within_shape(
    dataset: xr.Dataset,
    shape: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Filter data to include only points within a specified shape.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing data to filter.
    shape : geopandas.GeoDataFrame
        GeoDataFrame containing the shape to filter by.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing filtered data.
    """
    # Convert the dataset to a DataFrame
    df = dataset.to_dataframe().reset_index()
    
    # Ensure lon and lat are float
    df['lon'] = df['lon'].astype(float)
    df['lat'] = df['lat'].astype(float)
    
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='epsg:4326')
    
    # Perform spatial join to filter points within the shape
    filtered_gdf = gpd.sjoin(gdf, shape, how='inner', op='intersects')
    
    # Drop the geometry column to convert back to a regular DataFrame
    filtered_df = filtered_gdf.drop(columns='geometry')
    
    return filtered_df

def convert_hourly_to_daily(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = 'time'
) -> pd.DataFrame:
    """
    Convert hourly data to daily data by summing values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing hourly data.
    value_col : str
        Name of the column containing values to sum.
    time_col : str, optional
        Name of the time column, by default 'time'.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing daily data.
    """
    # Ensure time column is datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Create a date column
    df['date'] = df[time_col].dt.date
    
    # Group by date and sum values
    daily_df = df.groupby(['coordinate_id', 'lon', 'lat', 'date'])[value_col].sum().reset_index()
    
    # Rename columns for clarity
    daily_df.rename(columns={'date': time_col, value_col: f'daily_{value_col}'}, inplace=True)
    
    return daily_df
