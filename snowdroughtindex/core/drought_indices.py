"""
Module for calculating various drought indices based on SWE/P ratio.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_swe_p_ratio(swe: Union[xr.DataArray, np.ndarray],
                         precipitation: Union[xr.DataArray, np.ndarray],
                         time_period: str = 'monthly') -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the SWE/P ratio for drought assessment.
    
    Parameters
    ----------
    swe : xarray.DataArray or numpy.ndarray
        Snow Water Equivalent data
    precipitation : xarray.DataArray or numpy.ndarray
        Precipitation data
    time_period : str, optional
        Time period for ratio calculation ('monthly', 'seasonal', 'annual')
        Default: 'monthly'
        
    Returns
    -------
    xarray.DataArray or numpy.ndarray
        SWE/P ratio values
    """
    # Ensure inputs are xarray DataArrays with named dimensions
    if isinstance(swe, np.ndarray):
        swe = xr.DataArray(swe, dims=['time', 'lat', 'lon'])
    if isinstance(precipitation, np.ndarray):
        precipitation = xr.DataArray(precipitation, dims=['time', 'lat', 'lon'])
    
    # If dimensions are not named, rename them
    if not all(dim in ['time', 'lat', 'lon'] for dim in swe.dims):
        swe = swe.rename({swe.dims[0]: 'time', swe.dims[1]: 'lat', swe.dims[2]: 'lon'})
    if not all(dim in ['time', 'lat', 'lon'] for dim in precipitation.dims):
        precipitation = precipitation.rename({precipitation.dims[0]: 'time', precipitation.dims[1]: 'lat', precipitation.dims[2]: 'lon'})
    
    # Handle zero precipitation values
    precipitation = precipitation.where(precipitation > 0, 1e-10)
    
    # Calculate ratio
    ratio = swe / precipitation
    
    # Apply time period aggregation if needed
    if time_period == 'monthly':
        # Keep the time dimension but aggregate by month
        ratio = ratio.groupby('time').mean()
    elif time_period == 'seasonal':
        # Keep the time dimension but aggregate by season
        ratio = ratio.groupby('time.season').mean('time')
        # Convert season coordinate to time
        ratio = ratio.rename({'season': 'time'})
    elif time_period == 'annual':
        # Keep the time dimension but aggregate by year
        ratio = ratio.groupby('time.year').mean('time')
        # Convert year coordinate to time
        ratio = ratio.rename({'year': 'time'})
    else:
        raise ValueError(f"Invalid time_period: {time_period}. Must be one of: 'monthly', 'seasonal', 'annual'")
    
    return ratio

def calculate_swe_p_drought_index(swe: Union[xr.DataArray, np.ndarray],
                               precipitation: Union[xr.DataArray, np.ndarray],
                               time_period: str = 'monthly',
                               climatology_start_year: int = None,
                               climatology_end_year: int = None) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the SWE/P drought index.
    
    Parameters
    ----------
    swe : xarray.DataArray or numpy.ndarray
        Snow Water Equivalent data
    precipitation : xarray.DataArray or numpy.ndarray
        Precipitation data
    time_period : str, optional
        Time period for ratio calculation ('monthly', 'seasonal', 'annual')
        Default: 'monthly'
    climatology_start_year : int, optional
        Start year for climatology calculation
    climatology_end_year : int, optional
        End year for climatology calculation
        
    Returns
    -------
    xarray.DataArray or numpy.ndarray
        SWE/P drought index values
    """
    # Handle numpy arrays separately
    if isinstance(swe, np.ndarray) and isinstance(precipitation, np.ndarray):
        # Handle zero precipitation values
        precipitation = np.where(precipitation > 0, precipitation, 1e-10)
        
        # Calculate ratio
        ratio = swe / precipitation
        
        # Calculate standardization
        mean = np.nanmean(ratio, axis=0, keepdims=True)
        std = np.nanstd(ratio, axis=0, keepdims=True)
        
        # Handle zero standard deviation
        std = np.where(std == 0, 1e-10, std)
        
        # Calculate standardized anomaly
        anomaly = (ratio - mean) / std
        
        # Fill nan values with 0
        anomaly = np.nan_to_num(anomaly)
        
        return anomaly
    
    # Calculate SWE/P ratio
    ratio = calculate_swe_p_ratio(swe, precipitation, time_period)
    
    # Calculate anomaly
    if isinstance(ratio, xr.DataArray):
        if climatology_start_year is not None and climatology_end_year is not None:
            # Validate climatology years
            if climatology_start_year > climatology_end_year:
                raise ValueError("climatology_start_year must be less than or equal to climatology_end_year")
            
            # Select climatology period
            if 'time' in ratio.dims:
                climatology = ratio.sel(
                    time=slice(str(climatology_start_year), str(climatology_end_year))
                )
                if len(climatology.time) == 0:
                    raise ValueError(f"No data found in climatology period {climatology_start_year}-{climatology_end_year}")
            else:
                climatology = ratio
        else:
            climatology = ratio
            
        # Calculate mean and standard deviation
        if 'time' in climatology.dims:
            # Group by appropriate time period
            if time_period == 'monthly':
                # Add month coordinate to both climatology and full ratio
                climatology = climatology.assign_coords(month=('time', climatology.time.dt.month.values))
                ratio = ratio.assign_coords(month=('time', ratio.time.dt.month.values))
                
                # Calculate monthly statistics from climatology period
                mean = climatology.groupby('month').mean()
                std = climatology.groupby('month').std()
                
                # Map statistics back to each time point in the full dataset
                mean = mean.sel(month=ratio['month'])
                std = std.sel(month=ratio['month'])
                
                # Drop the month coordinate as it's no longer needed
                ratio = ratio.drop('month')
                mean = mean.drop('month')
                std = std.drop('month')
            elif time_period == 'seasonal':
                # Convert months to seasons
                month_to_season = {12: 'DJF', 1: 'DJF', 2: 'DJF',
                                 3: 'MAM', 4: 'MAM', 5: 'MAM',
                                 6: 'JJA', 7: 'JJA', 8: 'JJA',
                                 9: 'SON', 10: 'SON', 11: 'SON'}
                
                climatology = climatology.assign_coords(
                    season=('time', [month_to_season[m.item()] for m in climatology.time.dt.month])
                )
                ratio = ratio.assign_coords(
                    season=('time', [month_to_season[m.item()] for m in ratio.time.dt.month])
                )
                
                mean = climatology.groupby('season').mean()
                std = climatology.groupby('season').std()
                
                mean = mean.sel(season=ratio['season'])
                std = std.sel(season=ratio['season'])
                
                ratio = ratio.drop('season')
                mean = mean.drop('season')
                std = std.drop('season')
            elif time_period == 'annual':
                # Add year coordinate
                climatology = climatology.assign_coords(year=('time', climatology.time.dt.year.values))
                ratio = ratio.assign_coords(year=('time', ratio.time.dt.year.values))
                
                mean = climatology.groupby('year').mean()
                std = climatology.groupby('year').std()
                
                mean = mean.sel(year=ratio['year'])
                std = std.sel(year=ratio['year'])
                
                ratio = ratio.drop('year')
                mean = mean.drop('year')
                std = std.drop('year')
            else:
                raise ValueError(f"Invalid time_period: {time_period}")
            
            # Handle zero standard deviation
            std = xr.where(std == 0, 1e-10, std)
            
            # Calculate standardized anomaly
            anomaly = (ratio - mean) / std
            
            # Fill nan values with 0
            anomaly = anomaly.fillna(0)
            
            return anomaly
        else:
            mean = climatology.mean()
            std = climatology.std()
            
            # Handle zero standard deviation
            std = xr.where(std == 0, 1e-10, std)
            
            # Calculate standardized anomaly
            anomaly = (ratio - mean) / std
            
            # Fill nan values with 0
            anomaly = anomaly.fillna(0)
            
            return anomaly
    else:
        raise ValueError("Input must be either numpy arrays or xarray DataArrays")

def calculate_swe_p_anomaly(ratio: Union[xr.DataArray, np.ndarray],
                          climatology_period: Optional[tuple] = None) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate anomaly of SWE/P ratio from climatology.
    
    Parameters
    ----------
    ratio : xarray.DataArray or numpy.ndarray
        SWE/P ratio values
    climatology_period : tuple, optional
        Start and end years for climatology calculation
        Default: None (uses entire period)
        
    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Anomaly values (standard deviations from mean)
    """
    # Ensure input is xarray DataArray
    if isinstance(ratio, np.ndarray):
        ratio = xr.DataArray(ratio)
    
    # Calculate climatology
    if climatology_period and 'time' in ratio.dims:
        climatology = ratio.sel(time=slice(*climatology_period))
    else:
        climatology = ratio
    
    # Calculate mean and standard deviation
    if 'time' in climatology.dims:
        mean = climatology.mean('time')
        std = climatology.std('time')
    else:
        # For monthly data
        mean = climatology
        std = climatology.std('month')
    
    # Calculate anomaly
    anomaly = (ratio - mean) / std
    
    return anomaly

def classify_drought_severity(anomaly: Union[xr.DataArray, np.ndarray]) -> Union[xr.DataArray, np.ndarray]:
    """
    Classify drought severity based on anomaly values.
    
    Parameters
    ----------
    anomaly : xarray.DataArray or numpy.ndarray
        Anomaly values (standard deviations from mean)
        
    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Drought severity classification:
        -2: Extreme drought
        -1: Severe drought
        0: Moderate drought
        1: Mild drought
        2: Normal
        3: Mild wet
        4: Moderate wet
        5: Severe wet
        6: Extreme wet
    """
    # Ensure input is xarray DataArray
    if isinstance(anomaly, np.ndarray):
        anomaly = xr.DataArray(anomaly)
    
    # Initialize severity array
    severity = xr.zeros_like(anomaly)
    
    # Classify severity
    severity = xr.where(anomaly <= -2.0, -2, severity)  # Extreme drought
    severity = xr.where((anomaly > -2.0) & (anomaly <= -1.5), -1, severity)  # Severe drought
    severity = xr.where((anomaly > -1.5) & (anomaly <= -1.0), 0, severity)  # Moderate drought
    severity = xr.where((anomaly > -1.0) & (anomaly <= -0.5), 1, severity)  # Mild drought
    severity = xr.where((anomaly > -0.5) & (anomaly <= 0.5), 2, severity)  # Normal
    severity = xr.where((anomaly > 0.5) & (anomaly <= 1.0), 3, severity)  # Mild wet
    severity = xr.where((anomaly > 1.0) & (anomaly <= 1.5), 4, severity)  # Moderate wet
    severity = xr.where((anomaly > 1.5) & (anomaly <= 2.0), 5, severity)  # Severe wet
    severity = xr.where(anomaly > 2.0, 6, severity)  # Extreme wet
    
    return severity 