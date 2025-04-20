"""
Visualization module for the Snow Drought Index package.

This module contains functions for visualizing SWE data, drought classifications,
and other analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

def plot_swe_timeseries(swe_data, stations=None, figsize=(12, 6)):
    """
    Plot SWE time series for selected stations.
    
    Parameters
    ----------
    swe_data : pandas.DataFrame
        DataFrame containing SWE data with time as index.
    stations : list, optional
        List of station IDs to plot. If None, all stations are plotted.
    figsize : tuple, optional
        Figure size, by default (12, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if stations is None:
        stations = swe_data.columns
    
    for station in stations:
        if station in swe_data.columns:
            ax.plot(swe_data.index, swe_data[station], label=station)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('SWE (mm)')
    ax.set_title('SWE Time Series')
    ax.grid(True, alpha=0.3)
    
    if len(stations) <= 10:  # Only show legend if there aren't too many stations
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_data_availability(swe_stations, original_swe_data, gapfilled_swe_data=None, figsize=(14, 8)):
    """
    Plot the percentage of SWE stations available on the first day of each month of each year.
    
    Parameters
    ----------
    swe_stations : pandas.DataFrame or geopandas.GeoDataFrame
        DataFrame containing SWE station information.
    original_swe_data : xarray.DataArray or pandas.DataFrame
        Original SWE observations.
    gapfilled_swe_data : xarray.DataArray or pandas.DataFrame, optional
        Gap-filled SWE observations. If provided, a comparison plot will be made.
    figsize : tuple, optional
        Figure size, by default (14, 8).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Initialize plot
    fig, axs = plt.subplots(6, 2, sharex=True, sharey=True, figsize=figsize)
    elem = -1
    column = 0
    
    # Convert to xarray if pandas DataFrame
    if isinstance(original_swe_data, pd.DataFrame):
        original_swe_data = original_swe_data.to_xarray()
    
    if gapfilled_swe_data is not None and isinstance(gapfilled_swe_data, pd.DataFrame):
        gapfilled_swe_data = gapfilled_swe_data.to_xarray()
    
    # Loop over months
    for m in range(1, 12+1):
        # Controls for plotting on right subplot (i.e., month)
        elem += 1
        if elem == 6:
            column += 1
            elem = 0
        
        # For SWE data with gap filling
        if gapfilled_swe_data is not None:
            # Extract data on the first of the month m
            data_month_gapfilled = gapfilled_swe_data.sel(
                station_id=swe_stations.station_id.values, 
                time=((gapfilled_swe_data['time.month'] == m) & (gapfilled_swe_data['time.day'] == 1))
            )
            
            # Count the % of stations with data on those dates
            data_month_gapfilled_count = data_month_gapfilled.count(dim='station_id') / len(swe_stations) * 100
            
            # Get the data values - handle both DataArray and Dataset
            if hasattr(data_month_gapfilled_count, 'data'):
                # It's a DataArray
                gapfilled_count_values = data_month_gapfilled_count.data
            else:
                # It's a Dataset, get the first data variable
                var_name = list(data_month_gapfilled_count.data_vars)[0]
                gapfilled_count_values = data_month_gapfilled_count[var_name].values
            
            # Plot bar chart of available data
            axs[elem, column].bar(
                data_month_gapfilled_count['time.year'], 
                gapfilled_count_values, 
                color='r', 
                alpha=.5
            )
        
        # Same process as above but for original SWE data
        data_month = original_swe_data.sel(
            station_id=swe_stations.station_id.values, 
            time=((original_swe_data['time.month'] == m) & (original_swe_data['time.day'] == 1))
        )
        data_month_count = data_month.count(dim='station_id') / len(swe_stations) * 100
        
        # Get the data values - handle both DataArray and Dataset
        if hasattr(data_month_count, 'data'):
            # It's a DataArray
            count_values = data_month_count.data
        else:
            # It's a Dataset, get the first data variable
            var_name = list(data_month_count.data_vars)[0]
            count_values = data_month_count[var_name].values
            
        axs[elem, column].bar(data_month_count['time.year'], count_values, color='b')
        
        # Add plot labels
        if elem == 5 and column == 0:
            axs[elem, column].set_ylabel('% of SWE stations \n with data in basin')
        
        month_name = pd.Timestamp(year=2000, month=m, day=1).strftime("%b")
        axs[elem, column].set_title(f'1st {month_name}', fontweight='bold')
    
    if gapfilled_swe_data is not None:
        bluepatch = mpatches.Patch(color='b', label='original data')
        redpatch = mpatches.Patch(color='r', alpha=.5, label='after gap filling')
        plt.legend(handles=[bluepatch, redpatch])
    
    plt.tight_layout()
    return fig

def plot_gap_filling_evaluation(evaluation_scores, figsize=(9, 5)):
    """
    Plot evaluation results for the artificial gap filling.
    
    Parameters
    ----------
    evaluation_scores : dict
        Dictionary containing the artificial gap filling evaluation results for several metrics.
    figsize : tuple, optional
        Figure size, by default (9, 5).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Initialize figure
    ncols = 3
    fig, axs = plt.subplots(2, ncols, sharex=True, sharey=False, figsize=figsize)
    elem = -1
    row = 0
    
    # Define metrics used & their optimal values
    metrics = list(evaluation_scores.keys())
    metrics_optimal_values = {
        'RMSE': 0, 
        "KGE''": 1, 
        "KGE''_corr": 1, 
        "KGE''_bias": 0, 
        "KGE''_var": 1
    }
    units = {
        'RMSE': 'mm', 
        "KGE''": '-', 
        "KGE''_corr": '-', 
        "KGE''_bias": '-', 
        "KGE''_var": '-'
    }
    
    # Loop over metrics
    for m in metrics:
        # Controls for plotting on right subplot
        elem += 1
        if elem == ncols:
            row += 1
            elem = 0
        
        # Loop over iterations
        for i in range(evaluation_scores[m].shape[2]):
            # Plot boxplot for each month
            for mo in range(1, 12+1):
                nonan = evaluation_scores[m][mo-1, :, i][~np.isnan(evaluation_scores[m][mo-1, :, i])]
                bp = axs[row, elem].boxplot(nonan, positions=[mo], patch_artist=True, showfliers=False, widths=.7)
                plt.setp(bp['boxes'], color='b', alpha=.5)
                plt.setp(bp['whiskers'], color='b')
                plt.setp(bp['medians'], color='k')
        
        # Add elements to the plot
        axs[row, elem].plot(
            np.arange(0, 13+1), 
            [metrics_optimal_values[m]]*14, 
            color='grey', 
            ls='--', 
            label='best values'
        )
        axs[row, elem].set_xlim([0, 13])
        axs[row, elem].set_xticks(np.arange(1, 12+1))
        axs[row, elem].set_ylabel(f"{m} [{units[m]}]", fontweight='bold')
        axs[row, elem].tick_params(axis='y', labelsize=8)
        
        if row == 1:
            axs[row, elem].set_xticklabels(np.arange(1, 12+1), fontsize=8)
    
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].set_xlabel('months (1st of)', fontweight='bold')
    
    # Remove unused subplot
    if len(metrics) < 6:
        for i in range(len(metrics), 6):
            row = i // ncols
            elem = i % ncols
            fig.delaxes(axs[row][elem])
    
    plt.tight_layout()
    return fig

def plot_sswei_timeseries(sswei_data, year_column='season_year', swei_column='SWEI', 
                         classification_column='Drought_Classification', figsize=(12, 6)):
    """
    Plot SSWEI time series with drought classification color coding.
    
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
    figsize : tuple, optional
        Figure size, by default (12, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    # Add threshold lines
    ax.axhline(-2.0, color='darkred', linestyle='--', alpha=0.5, label='Exceptional Drought Threshold')
    ax.axhline(-1.5, color='red', linestyle='--', alpha=0.5, label='Extreme Drought Threshold')
    ax.axhline(-1.0, color='orangered', linestyle='--', alpha=0.5, label='Severe Drought Threshold')
    ax.axhline(-0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Drought Threshold')
    ax.axhline(0.5, color='lightblue', linestyle='--', alpha=0.5, label='Abnormally Wet Threshold')
    ax.axhline(1.0, color='deepskyblue', linestyle='--', alpha=0.5, label='Moderately Wet Threshold')
    ax.axhline(1.5, color='blue', linestyle='--', alpha=0.5, label='Very Wet Threshold')
    ax.axhline(2.0, color='darkblue', linestyle='--', alpha=0.5, label='Extremely Wet Threshold')
    
    # Customize the plot
    ax.set_title('SSWEI Trends by Season Year')
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
    
    # Add a separate legend for threshold lines
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

def plot_drought_characteristics(drought_characteristics, figsize=(12, 8)):
    """
    Plot drought characteristics including duration, severity, and intensity.
    
    Parameters
    ----------
    drought_characteristics : pandas.DataFrame
        DataFrame containing drought characteristics.
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
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

def plot_drought_trends(trend_data, figsize=(12, 6)):
    """
    Plot drought trends over time.
    
    Parameters
    ----------
    trend_data : pandas.DataFrame
        DataFrame containing drought trend analysis.
    figsize : tuple, optional
        Figure size, by default (12, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
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

def plot_drought_classification_heatmap(sswei_data, figsize=(12, 8)):
    """
    Create a heatmap of drought classifications by decade.
    
    Parameters
    ----------
    sswei_data : pandas.DataFrame
        DataFrame containing SSWEI data with season_year and Drought_Classification columns.
    figsize : tuple, optional
        Figure size, by default (12, 8).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Add decade column
    sswei_data['decade'] = (sswei_data['season_year'] // 10) * 10
    
    # Count classifications by decade
    decade_counts = pd.crosstab(sswei_data['decade'], sswei_data['Drought_Classification'])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(decade_counts, cmap='YlOrRd', annot=True, fmt='d', cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Drought Classifications by Decade')
    ax.set_ylabel('Decade')
    ax.set_xlabel('Drought Classification')
    
    plt.tight_layout()
    return fig

def plot_drought_severity_distribution(sswei_data, figsize=(10, 6)):
    """
    Plot the distribution of drought severity values.
    
    Parameters
    ----------
    sswei_data : pandas.DataFrame
        DataFrame containing SSWEI data with SWEI column.
    figsize : tuple, optional
        Figure size, by default (10, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Calculate drought severity
    from snowdroughtindex.core.drought_classification import get_drought_severity
    sswei_data['drought_severity'] = sswei_data['SWEI'].apply(get_drought_severity)
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(sswei_data[sswei_data['drought_severity'] > 0]['drought_severity'], bins=10, kde=True, ax=ax)
    ax.set_title('Distribution of Drought Severity')
    ax.set_xlabel('Drought Severity')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_seasonal_swe(daily_mean, figsize=(10, 6)):
    """
    Plot daily mean SWE values by year to show seasonal patterns.
    
    Parameters
    ----------
    daily_mean : pandas.DataFrame
        DataFrame containing daily mean SWE values with date and mean_SWE columns.
    figsize : tuple, optional
        Figure size, by default (10, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Extract year and day of year
    daily_mean['Year'] = daily_mean['date'].dt.year
    daily_mean['DayOfYear'] = daily_mean['date'].dt.dayofyear
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group data by year and plot each year separately
    for year, data in daily_mean.groupby('Year'):
        ax.plot(data['DayOfYear'], data['mean_SWE'], label=str(year))
    
    # Add labels and title
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('SWE (Snow Water Equivalent)')
    ax.set_title('Daily SWE by Year')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(correlations, figsize=(10, 8)):
    """
    Plot a correlation matrix between predictors and predictands.
    
    Parameters
    ----------
    correlations : pandas.DataFrame
        DataFrame of correlations between predictors and predictands.
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot matrix of correlations
    cmap = sns.cm.rocket_r
    sns.heatmap(correlations, annot=True, cmap=cmap, cbar_kws={'label': 'R²'}, vmin=0, vmax=1, ax=ax)
    ax.set(xlabel="Flow accumulation periods", ylabel="SWE dates (1st)")
    
    plt.tight_layout()
    return fig
