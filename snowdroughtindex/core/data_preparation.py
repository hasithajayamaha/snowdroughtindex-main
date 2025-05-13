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
from shapely.geometry import Point, Polygon, MultiPolygon
from snowdroughtindex.utils.progress import ProgressTracker, track_progress, monitor_memory

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

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=4,  # Number of major steps
        desc="Preprocessing SWE data",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Convert to DataFrame
        progress_tracker.update(1, "Converting to DataFrame")
        swe_df = swe_data.to_dataframe()
        
        # Step 2: Reset index to make coordinates accessible as columns
        progress_tracker.update(1, "Resetting index")
        if isinstance(swe_df.index, pd.MultiIndex):
            swe_df = swe_df.reset_index()
        
        # Step 3: Add metadata
        progress_tracker.update(1, "Adding metadata")
        swe_df['data_source'] = 'SWE'
        swe_df['units'] = 'mm'
        
        # Step 4: Clean up and return
        progress_tracker.update(1, "Finalizing preprocessing")
        return swe_df
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during SWE preprocessing: {str(e)}")

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=4,  # Number of major steps
        desc="Preprocessing precipitation data",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Convert to DataFrame
        progress_tracker.update(1, "Converting to DataFrame")
        precip_df = precip_data.to_dataframe()
        
        # Step 2: Reset index to make coordinates accessible as columns
        progress_tracker.update(1, "Resetting index")
        if isinstance(precip_df.index, pd.MultiIndex):
            precip_df = precip_df.reset_index()
        
        # Step 3: Convert time to datetime if it's not already
        progress_tracker.update(1, "Converting time format")
        if 'time' in precip_df.columns:
            precip_df['time'] = pd.to_datetime(precip_df['time'])
        
        # Step 4: Add metadata
        progress_tracker.update(1, "Adding metadata")
        precip_df['data_source'] = 'Precipitation'
        precip_df['units'] = 'mm'
        
        return precip_df
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during precipitation preprocessing: {str(e)}")

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=3,  # Number of major steps
        desc="Extracting stations in basin",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Extract basin geometry
        progress_tracker.update(1, "Extracting basin geometry")
        basin_geom = basin_shapefile.loc[basin_shapefile['Station_ID'] == basin_id].iloc[0].loc["geometry"]
        
        # Step 2: Handle buffer if specified
        progress_tracker.update(1, "Processing buffer")
        if buffer_km > 0:
            # Convert basin & stations geometry to a different CRS to be able to add a buffer in meters
            basin_crs_conversion = basin_shapefile.loc[basin_shapefile['Station_ID'] == basin_id].to_crs(epsg=3763)
            stations_crs_conversion = stations.to_crs(epsg=3763)
            
            # Add a buffer in meters around the basin
            buffer_m = buffer_km * 1000
            basin_buffer = basin_crs_conversion.buffer(buffer_m)
            mask = stations_crs_conversion.within(basin_buffer.iloc[0])
            
            # Convert the buffer back to the original CRS for plotting
            basin_buffer = basin_buffer.to_crs(epsg=4326)
        else:
            basin_buffer = 0
            mask = stations.within(basin_geom)
        
        # Step 3: Extract stations and return results
        progress_tracker.update(1, "Extracting stations")
        stations_in_basin = stations.loc[mask].assign(basin=basin_id)
        
        return stations_in_basin, basin_buffer
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during station extraction: {str(e)}")

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=2,  # Number of major steps
        desc="Filtering stations",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Validate input data
        progress_tracker.update(1, "Validating input data")
        if isinstance(data, pd.DataFrame):
            if 'station_id' not in data.columns:
                raise ValueError("DataFrame does not contain 'station_id' column")
            filtered_data = data[data['station_id'].isin(stations_list)]
        elif isinstance(data, xr.Dataset):
            filtered_data = data.sel(station_id=stations_list)
        else:
            raise TypeError("Data must be a pandas DataFrame or xarray Dataset")
        
        # Step 2: Return filtered data
        progress_tracker.update(1, "Returning filtered data")
        return filtered_data
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during station filtering: {str(e)}")

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=3,  # Number of major steps
        desc="Assessing data availability",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Count non-NaN values along the time dimension
        progress_tracker.update(1, "Counting non-NaN values")
        counts = data.count(dim=time_dim)
        
        # Step 2: Calculate the total number of time points
        progress_tracker.update(1, "Calculating total time points")
        total_times = len(data[time_dim])
        
        # Step 3: Calculate the percentage of available data
        progress_tracker.update(1, "Calculating availability percentages")
        availability = (counts / total_times) * 100
        
        return availability
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during data availability assessment: {str(e)}")

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=3,  # Number of major steps
        desc="Converting to GeoDataFrame",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Validate input columns
        progress_tracker.update(1, "Validating input columns")
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{lon_col}' and '{lat_col}' columns")
        
        # Step 2: Create Point geometries
        progress_tracker.update(1, "Creating Point geometries")
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        
        # Step 3: Create and return GeoDataFrame
        progress_tracker.update(1, "Creating GeoDataFrame")
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        
        return gdf
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during GeoDataFrame conversion: {str(e)}")

@monitor_memory
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
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=4,  # Number of major steps
        desc="Performing spatial join",
        unit="steps",
        memory_monitoring=True
    )
    
    try:
        # Step 1: Validate input geometries
        progress_tracker.update(1, "Validating input geometries")
        if not all(isinstance(geom, Point) for geom in points_gdf.geometry):
            raise ValueError("points_gdf must contain only Point geometries")
        if not all(isinstance(geom, (Polygon, MultiPolygon)) for geom in polygons_gdf.geometry):
            raise ValueError("polygons_gdf must contain only Polygon or MultiPolygon geometries")
        
        # Step 2: Ensure CRS match
        progress_tracker.update(1, "Checking coordinate systems")
        if points_gdf.crs != polygons_gdf.crs:
            raise ValueError("Coordinate systems of points and polygons must match")
        
        # Step 3: Perform spatial join
        progress_tracker.update(1, "Executing spatial join")
        result = gpd.sjoin(points_gdf, polygons_gdf, how=how, op=op)
        
        # Step 4: Clean up and return
        progress_tracker.update(1, "Finalizing results")
        return result
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during spatial join: {str(e)}")

@monitor_memory
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
        
        # Get unique station IDs
        station_ids = data['station_id'].unique()
        n_stations = len(station_ids)
        
        # Initialize progress tracking
        progress_tracker = ProgressTracker(
            total=n_stations,
            desc="Updating coordinates",
            unit="stations",
            memory_monitoring=True
        )
        
        try:
            # Update coordinates in the DataFrame
            for i, station_id in enumerate(station_ids):
                if station_id in coord_map:
                    mask = data['station_id'] == station_id
                    data.loc[mask, 'lat'] = coord_map[station_id][lat_col]
                    data.loc[mask, 'lon'] = coord_map[station_id][lon_col]
                
                # Update progress
                progress_tracker.update(1, f"Updated coordinates for station {i+1}/{n_stations}")
            
            return data
            
        except Exception as e:
            progress_tracker.close()
            raise RuntimeError(f"Error during coordinate update: {str(e)}")
    
    elif isinstance(data, xr.Dataset):
        # Create a mapping for quick lookup from the coordinates DataFrame
        coord_map = coordinates_df.set_index(id_col)[[lat_col, lon_col]].to_dict('index')
        
        # Get station IDs from the dataset
        station_ids = data['station_id'].values
        n_stations = len(station_ids)
        
        # Initialize progress tracking
        progress_tracker = ProgressTracker(
            total=n_stations,
            desc="Updating coordinates",
            unit="stations",
            memory_monitoring=True
        )
        
        try:
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
                
                # Update progress
                progress_tracker.update(1, f"Updated coordinates for station {i+1}/{n_stations}")
            
            # Assign the updated coordinates back to the dataset
            data['lat'] = (data['lat'].dims, updated_lat)
            data['lon'] = (data['lon'].dims, updated_lon)
            
            return data
            
        except Exception as e:
            progress_tracker.close()
            raise RuntimeError(f"Error during coordinate update: {str(e)}")
    
    else:
        raise TypeError("Data must be a pandas DataFrame or xarray Dataset")

@monitor_memory
def filter_data_within_shape(
    dataset: xr.Dataset,
    shape: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Filter data to include only points within a specified shape.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the data to filter.
    shape : geopandas.GeoDataFrame
        GeoDataFrame containing the shape to filter within.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the data points within the specified shape.
    """
    # Convert dataset to DataFrame
    df = dataset.to_dataframe()
    
    # Get unique station IDs
    station_ids = df['station_id'].unique()
    n_stations = len(station_ids)
    
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=n_stations,
        desc="Filtering stations within shape",
        unit="stations",
        memory_monitoring=True
    )
    
    try:
        # Initialize list to store filtered data
        filtered_data = []
        
        # Process each station
        for i, station_id in enumerate(station_ids):
            # Get station data
            station_data = df[df['station_id'] == station_id]
            
            # Create Point geometry for station
            point = Point(station_data['lon'].iloc[0], station_data['lat'].iloc[0])
            
            # Check if point is within shape
            if point.within(shape.geometry.iloc[0]):
                filtered_data.append(station_data)
            
            # Update progress
            progress_tracker.update(1, f"Processed station {i+1}/{n_stations}")
        
        # Combine filtered data
        if filtered_data:
            result = pd.concat(filtered_data, ignore_index=True)
        else:
            result = pd.DataFrame()
        
        return result
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during data filtering: {str(e)}")

@monitor_memory
def convert_hourly_to_daily(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = 'time'
) -> pd.DataFrame:
    """
    Convert hourly data to daily data by taking the mean of each day.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing hourly data.
    value_col : str
        Name of the column containing the values to convert.
    time_col : str, optional
        Name of the time column, by default 'time'.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing daily data.
    """
    # Ensure time column is datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Get unique station IDs
    station_ids = df['station_id'].unique()
    n_stations = len(station_ids)
    
    # Initialize progress tracking
    progress_tracker = ProgressTracker(
        total=n_stations,
        desc="Converting hourly to daily data",
        unit="stations",
        memory_monitoring=True
    )
    
    try:
        # Initialize list to store daily data
        daily_data = []
        
        # Process each station
        for i, station_id in enumerate(station_ids):
            # Get station data
            station_data = df[df['station_id'] == station_id]
            
            # Resample to daily
            daily_station = station_data.set_index(time_col).resample('D')[value_col].mean().reset_index()
            daily_station['station_id'] = station_id
            
            daily_data.append(daily_station)
            
            # Update progress
            progress_tracker.update(1, f"Processed station {i+1}/{n_stations}")
        
        # Combine daily data
        result = pd.concat(daily_data, ignore_index=True)
        
        return result
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during hourly to daily conversion: {str(e)}")
