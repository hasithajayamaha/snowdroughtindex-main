"""
SSWEI (Standardized Snow Water Equivalent Index) module for the Snow Drought Index package.

This module contains functions for calculating the SSWEI, including SWE integration,
probability transformation, and drought classification.

The SSWEI methodology is based on Huning & AghaKouchak's approach.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import integrate
from scipy.stats import norm
import psutil
import gc
from snowdroughtindex.utils.progress import ProgressTracker, track_progress, monitor_memory

def perturb_zeros(swe_column: pd.Series) -> pd.Series:
    """
    Perturb zero values with small positive values to avoid issues with log transformations.
    
    Parameters
    ----------
    swe_column : pandas.Series
        Series containing SWE values.
        
    Returns
    -------
    pandas.Series
        Series with zero values replaced by small perturbations.
    """
    swe_array = swe_column.to_numpy()  # Convert to NumPy array for efficient manipulation
    
    # If there are no positive values, use a very small default
    if (swe_array > 0).sum() == 0:
        nonzero_min = 0.01
    else:
        nonzero_min = swe_array[swe_array > 0].min()  # Find the smallest nonzero value
    
    # Generate perturbations for zero values
    perturbation = np.random.uniform(0, nonzero_min / 2, size=swe_column[swe_column == 0].shape)
    
    # Replace zeros with perturbation
    swe_column_perturbed = swe_column.copy()
    swe_column_perturbed[swe_column_perturbed == 0] = perturbation
    
    return swe_column_perturbed

def prepare_season_data(
    daily_mean: pd.DataFrame,
    start_month: int = 11,
    start_day: int = 1,
    end_month: int = 4,
    end_day: int = 30,
    min_swe: float = 15
) -> pd.DataFrame:
    """
    Prepare seasonal data by filtering for complete snow seasons.
    
    Parameters
    ----------
    daily_mean : pandas.DataFrame
        DataFrame containing daily mean SWE values with 'date' and 'mean_SWE' columns.
    start_month : int, optional
        Starting month of the snow season, by default 11 (November).
    start_day : int, optional
        Starting day of the snow season, by default 1.
    end_month : int, optional
        Ending month of the snow season, by default 4 (April).
    end_day : int, optional
        Ending day of the snow season, by default 30.
    min_swe : float, optional
        Minimum SWE value to consider as the start of the snow season, by default 15.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing filtered seasonal data with additional columns for season_year and Month-Day.
    """
    # Add season_year column (the year that the season starts in)
    daily_mean['season_year'] = daily_mean['date'].apply(lambda x: x.year if x.month >= start_month else x.year - 1)
    
    # Find the first date with min_swe SWE each year to set as season start
    season_starts = daily_mean[daily_mean['mean_SWE'] >= min_swe].groupby('season_year')['date'].min()
    
    # Filter seasons based on season start and ensure they run through to the end date of the next year
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
    if not filtered_seasons:
        return pd.DataFrame()  # Return empty DataFrame if no complete seasons
    
    season_data = pd.concat(filtered_seasons, ignore_index=True)
    
    # Extract the year and month-day for filtering
    season_data['Year'] = season_data['date'].dt.year
    season_data['Month-Day'] = season_data['date'].dt.strftime('%m-%d')
    season_data['Year_Month'] = season_data['date'].dt.strftime('%Y-%m')
    
    # Perturb zero values
    season_data['mean_SWE'] = perturb_zeros(season_data['mean_SWE'])
    
    return season_data

def integrate_season(group: pd.DataFrame) -> pd.Series:
    """
    Integrate SWE values over a time period.
    
    Parameters
    ----------
    group : pandas.DataFrame
        DataFrame containing SWE values for a specific time period.
        
    Returns
    -------
    pandas.Series
        Series containing the integrated SWE value.
    """
    # Ensure dates are sorted
    group = group.sort_values(by='date')
    
    # Convert dates to numerical days since start of the period
    days_since_start = (group['date'] - group['date'].min()).dt.days
    
    # Integrate SWE over the period
    total_swe_integration = integrate.trapezoid(group['mean_SWE'], days_since_start)
    
    return pd.Series({'total_SWE_integration': total_swe_integration})

def calculate_seasonal_integration(
    season_data: pd.DataFrame,
    start_month: int = 11
) -> pd.DataFrame:
    """
    Calculate seasonal integration of SWE values.
    
    Parameters
    ----------
    season_data : pandas.DataFrame
        DataFrame containing seasonal SWE data.
    start_month : int, optional
        Starting month of the snow season, by default 11 (November).
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing integrated SWE values for each season.
    """
    # Group by month and compute integration
    integrated_data_monthly = season_data.groupby('Year_Month').apply(integrate_season).reset_index()
    
    # Ensure season_year corresponds to each month
    integrated_data_monthly['season_year'] = integrated_data_monthly['Year_Month'].apply(
        lambda x: int(x.split('-')[0]) if int(x.split('-')[1]) >= start_month else int(x.split('-')[0]) - 1
    )
    
    # Group by season_year and compute integration
    integrated_data_season = integrated_data_monthly.groupby('season_year').sum().reset_index()
    
    return integrated_data_season

def gringorten_probabilities(values: np.ndarray) -> np.ndarray:
    """
    Compute Gringorten plotting position probabilities.
    
    Parameters
    ----------
    values : array-like
        Array of values to compute probabilities for.
        
    Returns
    -------
    numpy.ndarray
        Array of Gringorten probabilities.
    """
    sorted_values = np.sort(values)
    ranks = np.argsort(np.argsort(values)) + 1  # Rank from smallest to largest
    n = len(values)
    probabilities = (ranks - 0.44) / (n + 0.12)
    
    return probabilities

def compute_swei(probabilities: np.ndarray) -> np.ndarray:
    """
    Transform probabilities to SWEI using the inverse normal distribution.
    
    Parameters
    ----------
    probabilities : array-like
        Array of probabilities.
        
    Returns
    -------
    numpy.ndarray
        Array of SWEI values.
    """
    return norm.ppf(probabilities)

def classify_drought(swei: float) -> str:
    """
    Classify drought conditions based on SWEI values.
    
    Parameters
    ----------
    swei : float
        SWEI value.
        
    Returns
    -------
    str
        Drought classification.
    """
    if swei <= -2.0:
        return "Exceptional Drought"
    elif -2.0 < swei <= -1.5:
        return "Extreme Drought"
    elif -1.5 < swei <= -1.0:
        return "Severe Drought"
    elif -1.0 < swei <= -0.5:
        return "Moderate Drought"
    elif -0.5 < swei <= 0.5:
        return "Near Normal"
    elif 0.5 < swei <= 1.0:
        return "Abnormally Wet"
    elif 1.0 < swei <= 1.5:
        return "Moderately Wet"
    elif 1.5 < swei <= 2.0:
        return "Very Wet"
    else:
        return "Extremely Wet"

@monitor_memory
def calculate_sswei(
    daily_mean: pd.DataFrame,
    start_month: int = 11,
    start_day: int = 1,
    end_month: int = 4,
    end_day: int = 30,
    min_swe: float = 15,
    chunk_size: int = 1000
) -> pd.DataFrame:
    """
    Calculate the Standardized Snow Water Equivalent Index (SSWEI) for each station.
    Memory-optimized version using chunking and numpy arrays.
    
    Parameters
    ----------
    daily_mean : pandas.DataFrame
        DataFrame containing daily mean SWE values for each station.
    start_month : int, optional
        Starting month of the snow season. Default is 11 (November).
    start_day : int, optional
        Starting day of the snow season. Default is 1.
    end_month : int, optional
        Ending month of the snow season. Default is 4 (April).
    end_day : int, optional
        Ending day of the snow season. Default is 30.
    min_swe : float, optional
        Minimum SWE value to consider for integration. Default is 15.
    chunk_size : int, optional
        Number of stations to process in each chunk. Default is 1000.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing SSWEI values for each station.
    """
    # Convert DataFrame to numpy array for memory efficiency
    swe_data = daily_mean.to_numpy(dtype=np.float32)
    station_names = daily_mean.columns
    dates = daily_mean.index
    
    # Initialize output array
    n_stations = len(station_names)
    sswei_results = np.zeros((n_stations,), dtype=np.float32)
    
    # Initialize progress tracking
    total_chunks = (n_stations + chunk_size - 1) // chunk_size
    progress_tracker = ProgressTracker(
        total=total_chunks,
        desc="Processing stations",
        unit="chunks",
        memory_monitoring=True
    )
    
    try:
        # Process stations in chunks
        for chunk_start in range(0, n_stations, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_stations)
            chunk_stations = station_names[chunk_start:chunk_end]
            chunk_data = swe_data[:, chunk_start:chunk_end]
            
            # Prepare season data for chunk
            season_data = prepare_season_data(
                pd.DataFrame(chunk_data, index=dates, columns=chunk_stations),
                start_month=start_month,
                start_day=start_day,
                end_month=end_month,
                end_day=end_day,
                min_swe=min_swe
            )
            
            # Calculate seasonal integration for chunk
            seasonal_integration = calculate_seasonal_integration(
                season_data,
                start_month=start_month
            )
            
            # Calculate probabilities and SSWEI for chunk
            for i, station in enumerate(chunk_stations):
                if station in seasonal_integration.columns:
                    values = seasonal_integration[station].dropna().values
                    if len(values) > 0:
                        probs = gringorten_probabilities(values)
                        swei = compute_swei(probs)
                        sswei_results[chunk_start + i] = swei[-1] if len(swei) > 0 else np.nan
            
            # Free memory
            del season_data, seasonal_integration
            gc.collect()
            
            # Update progress
            progress_tracker.update(1, f"Processed chunk {chunk_start//chunk_size + 1}")
        
        # Create final DataFrame
        sswei_df = pd.DataFrame(
            {'SSWEI': sswei_results},
            index=station_names,
            dtype=np.float32
        )
        
        return sswei_df
        
    except Exception as e:
        progress_tracker.close()
        raise RuntimeError(f"Error during SSWEI calculation: {str(e)}")

def plot_sswei(
    sswei_data: pd.DataFrame,
    ax: Optional[Axes] = None
) -> Figure:
    """
    Plot SSWEI values with drought classification thresholds.
    
    Parameters
    ----------
    sswei_data : pandas.DataFrame
        DataFrame containing SSWEI values with 'season_year' and 'SWEI' columns.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    import matplotlib.pyplot as plt
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Extract necessary columns and sort by season_year
    plot_data = sswei_data[['season_year', 'SWEI']].sort_values(by='season_year')
    
    # Plot SSWEI values
    ax.plot(plot_data['season_year'], plot_data['SWEI'], marker='o', label='SWEI', color='black')
    
    # Add thresholds for drought classifications
    ax.axhline(-2.0, color='r', linestyle='--', label='Exceptional Drought Threshold')
    ax.axhline(-1.5, color='orange', linestyle='--', label='Extreme Drought Threshold')
    ax.axhline(-1.0, color='yellow', linestyle='--', label='Severe Drought Threshold')
    ax.axhline(-0.5, color='gray', linestyle='--', label='Moderate Drought Threshold')
    ax.axhline(0.5, color='lightblue', linestyle='--', label='Abnormally Wet Threshold')
    ax.axhline(1.0, color='blue', linestyle='--', label='Moderately Wet Threshold')
    ax.axhline(1.5, color='darkblue', linestyle='--', label='Very Wet Threshold')
    ax.axhline(2.0, color='purple', linestyle='--', label='Extremely Wet Threshold')
    
    # Customize the plot
    ax.set_title('SSWEI Trends by Season Year')
    ax.set_xlabel('Season Year')
    ax.set_ylabel('Standardized SSWEI')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add legend below the plot
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        borderaxespad=0.,
        frameon=True
    )
    
    plt.tight_layout()
    
    return fig
