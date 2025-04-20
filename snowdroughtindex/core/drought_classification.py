"""
Drought Classification module for the Snow Drought Index package.

This module contains functions for classifying drought conditions based on SSWEI values,
analyzing drought characteristics, and performing temporal analysis of drought events.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Default drought classification thresholds
DEFAULT_THRESHOLDS = {
    "exceptional": -2.0,
    "extreme": -1.5,
    "severe": -1.0,
    "moderate": -0.5,
    "normal_lower": -0.5,
    "normal_upper": 0.5,
    "abnormally_wet": 1.0,
    "moderately_wet": 1.5,
    "very_wet": 2.0
}

def classify_drought(swei: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Classify drought conditions based on SSWEI values with configurable thresholds.
    
    Parameters
    ----------
    swei : float
        SSWEI value.
    thresholds : dict, optional
        Dictionary of threshold values for different drought classifications.
        If None, default thresholds are used.
        
    Returns
    -------
    str
        Drought classification.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    if swei <= thresholds.get("exceptional", -2.0):
        return "Exceptional Drought"
    elif swei <= thresholds.get("extreme", -1.5):
        return "Extreme Drought"
    elif swei <= thresholds.get("severe", -1.0):
        return "Severe Drought"
    elif swei <= thresholds.get("moderate", -0.5):
        return "Moderate Drought"
    elif swei <= thresholds.get("normal_upper", 0.5):
        return "Near Normal"
    elif swei <= thresholds.get("abnormally_wet", 1.0):
        return "Abnormally Wet"
    elif swei <= thresholds.get("moderately_wet", 1.5):
        return "Moderately Wet"
    elif swei <= thresholds.get("very_wet", 2.0):
        return "Very Wet"
    else:
        return "Extremely Wet"

def classify_drought_series(swei_series: pd.Series, thresholds: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Apply drought classification to a series of SSWEI values.
    
    Parameters
    ----------
    swei_series : pandas.Series
        Series of SSWEI values.
    thresholds : dict, optional
        Dictionary of threshold values for different drought classifications.
        If None, default thresholds are used.
        
    Returns
    -------
    pandas.Series
        Series of drought classifications.
    """
    return swei_series.apply(lambda x: classify_drought(x, thresholds))

def get_drought_severity(swei: float) -> float:
    """
    Calculate drought severity based on SSWEI value.
    Severity is defined as the absolute value of SSWEI for values below -0.5,
    and 0 for values above -0.5.
    
    Parameters
    ----------
    swei : float
        SSWEI value.
        
    Returns
    -------
    float
        Drought severity value.
    """
    return abs(min(swei, -0.5)) if swei < -0.5 else 0.0

def calculate_drought_characteristics(
    sswei_data: pd.DataFrame, 
    year_column: str = 'season_year',
    swei_column: str = 'SWEI',
    threshold: float = -0.5
) -> pd.DataFrame:
    """
    Calculate drought characteristics including frequency, duration, and severity.
    
    Parameters
    ----------
    sswei_data : pandas.DataFrame
        DataFrame containing SSWEI values with year and SWEI columns.
    year_column : str, optional
        Name of the column containing year information, by default 'season_year'.
    swei_column : str, optional
        Name of the column containing SWEI values, by default 'SWEI'.
    threshold : float, optional
        Threshold for defining drought conditions, by default -0.5.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing drought characteristics.
    """
    # Ensure data is sorted by year
    sswei_data = sswei_data.sort_values(by=year_column)
    
    # Identify drought years
    drought_mask = sswei_data[swei_column] < threshold
    sswei_data['is_drought'] = drought_mask
    
    # Calculate drought severity
    sswei_data['drought_severity'] = sswei_data[swei_column].apply(get_drought_severity)
    
    # Identify drought events (consecutive drought years)
    sswei_data['drought_event'] = (sswei_data['is_drought'] != sswei_data['is_drought'].shift()).cumsum()
    drought_events = sswei_data[sswei_data['is_drought']].groupby('drought_event')
    
    # Calculate drought characteristics
    drought_characteristics = []
    
    for event_id, event_data in drought_events:
        start_year = event_data[year_column].min()
        end_year = event_data[year_column].max()
        duration = len(event_data)
        severity = event_data['drought_severity'].sum()
        intensity = severity / duration
        min_swei = event_data[swei_column].min()
        
        drought_characteristics.append({
            'event_id': event_id,
            'start_year': start_year,
            'end_year': end_year,
            'duration': duration,
            'severity': severity,
            'intensity': intensity,
            'min_swei': min_swei,
            'classification': classify_drought(min_swei)
        })
    
    if not drought_characteristics:
        return pd.DataFrame()
    
    return pd.DataFrame(drought_characteristics)

def analyze_drought_trends(
    sswei_data: pd.DataFrame, 
    year_column: str = 'season_year',
    swei_column: str = 'SWEI',
    window_size: int = 10
) -> pd.DataFrame:
    """
    Analyze drought trends over time using a moving window approach.
    
    Parameters
    ----------
    sswei_data : pandas.DataFrame
        DataFrame containing SSWEI values with year and SWEI columns.
    year_column : str, optional
        Name of the column containing year information, by default 'season_year'.
    swei_column : str, optional
        Name of the column containing SWEI values, by default 'SWEI'.
    window_size : int, optional
        Size of the moving window in years, by default 10.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing drought trend analysis.
    """
    # Ensure data is sorted by year
    sswei_data = sswei_data.sort_values(by=year_column)
    
    # Calculate drought frequency and severity in moving windows
    trend_data = []
    
    for i in range(len(sswei_data) - window_size + 1):
        window = sswei_data.iloc[i:i+window_size]
        start_year = window[year_column].min()
        end_year = window[year_column].max()
        
        drought_years = window[window[swei_column] < -0.5]
        drought_frequency = len(drought_years) / window_size
        
        if not drought_years.empty:
            mean_severity = drought_years[swei_column].apply(get_drought_severity).mean()
            max_severity = drought_years[swei_column].apply(get_drought_severity).max()
        else:
            mean_severity = 0
            max_severity = 0
        
        trend_data.append({
            'start_year': start_year,
            'end_year': end_year,
            'mid_year': (start_year + end_year) / 2,
            'drought_frequency': drought_frequency,
            'mean_severity': mean_severity,
            'max_severity': max_severity
        })
    
    return pd.DataFrame(trend_data)

def plot_drought_classification(
    sswei_data: pd.DataFrame, 
    year_column: str = 'season_year',
    swei_column: str = 'SWEI',
    classification_column: str = 'Drought_Classification',
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_thresholds: bool = True
) -> Figure:
    """
    Plot SSWEI values with drought classification color coding.
    
    Parameters
    ----------
    sswei_data : pandas.DataFrame
        DataFrame containing SSWEI values with year, SWEI, and classification columns.
    year_column : str, optional
        Name of the column containing year information, by default 'season_year'.
    swei_column : str, optional
        Name of the column containing SWEI values, by default 'SWEI'.
    classification_column : str, optional
        Name of the column containing drought classifications, by default 'Drought_Classification'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    figsize : tuple, optional
        Figure size, by default (12, 6).
    show_thresholds : bool, optional
        Whether to show threshold lines, by default True.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Ensure data is sorted by year
    plot_data = sswei_data.sort_values(by=year_column)
    
    # Define colors for different drought classifications
    colors = {
        'Exceptional Drought': 'darkred',
        'Extreme Drought': 'red',
        'Severe Drought': 'orangered',
        'Moderate Drought': 'orange',
        'Near Normal': 'gray',
        'Abnormally Wet': 'lightblue',
        'Moderately Wet': 'deepskyblue',
        'Very Wet': 'blue',
        'Extremely Wet': 'darkblue'
    }
    
    # Plot SSWEI values with color coding
    for classification, color in colors.items():
        mask = plot_data[classification_column] == classification
        if mask.any():
            ax.scatter(
                plot_data.loc[mask, year_column],
                plot_data.loc[mask, swei_column],
                color=color,
                label=classification,
                s=50,
                alpha=0.8
            )
    
    # Connect points with a line
    ax.plot(plot_data[year_column], plot_data[swei_column], color='black', alpha=0.5, linestyle='-', linewidth=1)
    
    # Add threshold lines if requested
    if show_thresholds:
        ax.axhline(-2.0, color='darkred', linestyle='--', alpha=0.5, label='Exceptional Drought Threshold')
        ax.axhline(-1.5, color='red', linestyle='--', alpha=0.5, label='Extreme Drought Threshold')
        ax.axhline(-1.0, color='orangered', linestyle='--', alpha=0.5, label='Severe Drought Threshold')
        ax.axhline(-0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Drought Threshold')
        ax.axhline(0.5, color='lightblue', linestyle='--', alpha=0.5, label='Abnormally Wet Threshold')
        ax.axhline(1.0, color='deepskyblue', linestyle='--', alpha=0.5, label='Moderately Wet Threshold')
        ax.axhline(1.5, color='blue', linestyle='--', alpha=0.5, label='Very Wet Threshold')
        ax.axhline(2.0, color='darkblue', linestyle='--', alpha=0.5, label='Extremely Wet Threshold')
    
    # Customize the plot
    ax.set_title('SSWEI Drought Classification by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Standardized SSWEI')
    ax.grid(True, alpha=0.3)
    
    # Add legend with two columns
    handles, labels = ax.get_legend_handles_labels()
    classification_handles = handles[:len(colors)]
    classification_labels = labels[:len(colors)]
    
    # Create a separate legend for classifications
    classification_legend = ax.legend(
        classification_handles, 
        classification_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        title='Drought Classifications'
    )
    
    # Add the classification legend to the plot
    ax.add_artist(classification_legend)
    
    # If showing thresholds, add a separate legend for threshold lines
    if show_thresholds:
        threshold_handles = handles[len(colors):]
        threshold_labels = labels[len(colors):]
        
        # Create a separate legend for thresholds
        ax.legend(
            threshold_handles, 
            threshold_labels,
            loc='upper right',
            title='Thresholds'
        )
    
    plt.tight_layout()
    
    return fig

def plot_drought_characteristics(
    drought_characteristics: pd.DataFrame,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[Figure]:
    """
    Plot drought characteristics including duration, severity, and intensity.
    
    Parameters
    ----------
    drought_characteristics : pandas.DataFrame
        DataFrame containing drought characteristics from calculate_drought_characteristics.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    figsize : tuple, optional
        Figure size, by default (12, 8).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if drought_characteristics.empty:
        print("No drought events to plot.")
        return None
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Sort by start year
    plot_data = drought_characteristics.sort_values(by='start_year')
    
    # Define colors based on classification
    classification_colors = {
        'Exceptional Drought': 'darkred',
        'Extreme Drought': 'red',
        'Severe Drought': 'orangered',
        'Moderate Drought': 'orange'
    }
    
    # Create bar colors based on classification
    bar_colors = [classification_colors.get(c, 'gray') for c in plot_data['classification']]
    
    # Create x-axis labels with start and end years
    x_labels = [f"{start}-{end}" for start, end in zip(plot_data['start_year'], plot_data['end_year'])]
    
    # Plot drought duration as bar height
    bars = ax.bar(
        range(len(plot_data)),
        plot_data['duration'],
        color=bar_colors,
        alpha=0.7
    )
    
    # Add severity as text on top of bars
    for i, (bar, severity) in enumerate(zip(bars, plot_data['severity'])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"Sev: {severity:.1f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Customize the plot
    ax.set_title('Drought Event Characteristics')
    ax.set_xlabel('Drought Event Period')
    ax.set_ylabel('Duration (years)')
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add a legend for classification colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=classification)
        for classification, color in classification_colors.items()
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        title='Drought Classification'
    )
    
    plt.tight_layout()
    
    return fig

def plot_drought_trends(
    trend_data: pd.DataFrame,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot drought trends over time.
    
    Parameters
    ----------
    trend_data : pandas.DataFrame
        DataFrame containing drought trend analysis from analyze_drought_trends.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    figsize : tuple, optional
        Figure size, by default (12, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = ax1.twinx()
    
    # Plot drought frequency
    line1 = ax1.plot(
        trend_data['mid_year'],
        trend_data['drought_frequency'],
        color='blue',
        marker='o',
        linestyle='-',
        label='Drought Frequency'
    )
    
    # Plot mean severity on secondary y-axis
    line2 = ax2.plot(
        trend_data['mid_year'],
        trend_data['mean_severity'],
        color='red',
        marker='s',
        linestyle='-',
        label='Mean Severity'
    )
    
    # Customize the plot
    ax1.set_title('Drought Trends Over Time')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Drought Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Mean Drought Severity', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add a legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    return fig
