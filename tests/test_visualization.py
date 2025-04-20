"""
Unit tests for the visualization module.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

from snowdroughtindex.utils import visualization

# Additional fixtures for visualization tests
@pytest.fixture
def sample_sswei_data():
    """
    Generate a sample SSWEI DataFrame.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample SSWEI data.
    """
    # Create sample data
    np.random.seed(42)  # For reproducibility
    
    # Create years
    years = list(range(2000, 2020))
    
    # Create SWEI values with some randomness
    swei_values = np.random.normal(0, 1, len(years))
    
    # Create classifications based on SWEI values
    classifications = []
    for val in swei_values:
        if val <= -2.0:
            classifications.append('Exceptional Drought')
        elif val <= -1.5:
            classifications.append('Extreme Drought')
        elif val <= -1.0:
            classifications.append('Severe Drought')
        elif val <= -0.5:
            classifications.append('Moderate Drought')
        elif val < 0.5:
            classifications.append('Near Normal')
        elif val < 1.0:
            classifications.append('Abnormally Wet')
        elif val < 1.5:
            classifications.append('Moderately Wet')
        elif val < 2.0:
            classifications.append('Very Wet')
        else:
            classifications.append('Extremely Wet')
    
    # Create DataFrame
    df = pd.DataFrame({
        'season_year': years,
        'SWEI': swei_values,
        'Drought_Classification': classifications
    })
    
    return df

@pytest.fixture
def sample_drought_characteristics():
    """
    Generate a sample drought characteristics DataFrame.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample drought characteristics.
    """
    # Create sample data
    data = {
        'start_year': [2002, 2007, 2012, 2017],
        'end_year': [2004, 2008, 2014, 2018],
        'duration': [3, 2, 3, 2],
        'severity': [2.5, 1.8, 3.2, 1.5],
        'intensity': [0.83, 0.9, 1.07, 0.75],
        'classification': ['Severe Drought', 'Moderate Drought', 'Extreme Drought', 'Moderate Drought']
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_trend_data():
    """
    Generate a sample drought trend DataFrame.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample drought trend data.
    """
    # Create sample data
    data = {
        'mid_year': [1985, 1990, 1995, 2000, 2005, 2010, 2015],
        'drought_frequency': [0.2, 0.3, 0.25, 0.4, 0.35, 0.45, 0.5],
        'mean_severity': [1.2, 1.5, 1.3, 1.8, 1.6, 2.0, 2.2]
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_daily_mean(sample_dates):
    """
    Generate a sample daily mean SWE DataFrame.
    
    Parameters
    ----------
    sample_dates : list
        List of datetime objects.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample daily mean SWE data.
    """
    # Create sample data
    np.random.seed(42)  # For reproducibility
    
    # Create a DataFrame with random SWE values
    data = {
        'date': sample_dates,
        'mean_SWE': np.random.rand(len(sample_dates)) * 100
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_correlation_matrix():
    """
    Generate a sample correlation matrix.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing sample correlation matrix.
    """
    # Create sample data
    np.random.seed(42)  # For reproducibility
    
    # Create row and column labels
    swe_dates = ['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1']
    flow_periods = ['Apr-Jun', 'May-Jul', 'Jun-Aug', 'Jul-Sep']
    
    # Create correlation matrix
    correlations = np.random.rand(len(swe_dates), len(flow_periods))
    
    # Create DataFrame
    df = pd.DataFrame(correlations, index=swe_dates, columns=flow_periods)
    
    return df

@pytest.fixture
def sample_evaluation_scores():
    """
    Generate sample evaluation scores for gap filling.
    
    Returns
    -------
    dict
        Dictionary containing sample evaluation scores.
    """
    # Create sample data
    np.random.seed(42)  # For reproducibility
    
    # Create metrics
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    
    # Create evaluation scores
    scores = {}
    for metric in metrics:
        # Create 3D array: (months, stations, iterations)
        scores[metric] = np.random.rand(12, 5, 3)
        
        # Adjust values based on metric
        if metric == 'RMSE':
            scores[metric] *= 10  # RMSE typically higher
        elif metric == "KGE''":
            scores[metric] = 0.5 + scores[metric] * 0.5  # KGE between 0.5 and 1
    
    return scores


class TestVisualization:
    """
    Test class for the visualization module.
    """
    
    def test_plot_swe_timeseries(self, sample_swe_dataframe):
        """
        Test the plot_swe_timeseries function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Plot SWE time series
        fig = visualization.plot_swe_timeseries(sample_swe_dataframe)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Date'
        assert ax.get_ylabel() == 'SWE (mm)'
        assert ax.get_title() == 'SWE Time Series'
        
        # Check that all stations are plotted
        assert len(ax.lines) == len(sample_swe_dataframe.columns)
        
        # Test with specific stations
        stations = [f'station_{i}' for i in range(1, 3)]  # First two stations
        fig = visualization.plot_swe_timeseries(sample_swe_dataframe, stations=stations)
        
        # Check that only the specified stations are plotted
        assert len(fig.axes[0].lines) == len(stations)
        
        # Test with custom figsize
        figsize = (8, 4)
        fig = visualization.plot_swe_timeseries(sample_swe_dataframe, figsize=figsize)
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_data_availability(self, sample_stations, sample_swe_dataset):
        """
        Test the plot_data_availability function.
        
        Parameters
        ----------
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        sample_swe_dataset : xarray.Dataset
            Sample SWE dataset.
        """
        # Plot data availability
        fig = visualization.plot_data_availability(sample_stations, sample_swe_dataset)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected number of subplots
        assert len(fig.axes) == 12  # 12 months
        
        # Test with gapfilled data
        fig = visualization.plot_data_availability(
            sample_stations, sample_swe_dataset, gapfilled_swe_data=sample_swe_dataset
        )
        
        # Check that the figure has the expected number of subplots
        assert len(fig.axes) == 12  # 12 months
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_data_availability(
            sample_stations, sample_swe_dataset, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_gap_filling_evaluation(self, sample_evaluation_scores):
        """
        Test the plot_gap_filling_evaluation function.
        
        Parameters
        ----------
        sample_evaluation_scores : dict
            Sample evaluation scores.
        """
        # Plot gap filling evaluation
        fig = visualization.plot_gap_filling_evaluation(sample_evaluation_scores)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected number of subplots
        assert len(fig.axes) == len(sample_evaluation_scores)
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_gap_filling_evaluation(
            sample_evaluation_scores, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_sswei_timeseries(self, sample_sswei_data):
        """
        Test the plot_sswei_timeseries function.
        
        Parameters
        ----------
        sample_sswei_data : pandas.DataFrame
            Sample SSWEI DataFrame.
        """
        # Plot SSWEI time series
        fig = visualization.plot_sswei_timeseries(sample_sswei_data)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Year'
        assert ax.get_ylabel() == 'Standardized SSWEI'
        assert ax.get_title() == 'SSWEI Trends by Season Year'
        
        # Test with custom column names
        fig = visualization.plot_sswei_timeseries(
            sample_sswei_data, 
            year_column='season_year', 
            swei_column='SWEI', 
            classification_column='Drought_Classification'
        )
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_sswei_timeseries(
            sample_sswei_data, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_drought_characteristics(self, sample_drought_characteristics):
        """
        Test the plot_drought_characteristics function.
        
        Parameters
        ----------
        sample_drought_characteristics : pandas.DataFrame
            Sample drought characteristics DataFrame.
        """
        # Plot drought characteristics
        fig = visualization.plot_drought_characteristics(sample_drought_characteristics)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Drought Event Period'
        assert ax.get_ylabel() == 'Duration (years)'
        assert ax.get_title() == 'Drought Event Characteristics'
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = visualization.plot_drought_characteristics(empty_df)
        
        # Check that the result is None for empty DataFrame
        assert result is None
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_drought_characteristics(
            sample_drought_characteristics, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_drought_trends(self, sample_trend_data):
        """
        Test the plot_drought_trends function.
        
        Parameters
        ----------
        sample_trend_data : pandas.DataFrame
            Sample drought trend DataFrame.
        """
        # Plot drought trends
        fig = visualization.plot_drought_trends(sample_trend_data)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax1 = fig.axes[0]
        assert ax1.get_xlabel() == 'Year'
        assert ax1.get_ylabel() == 'Drought Frequency'
        assert ax1.get_title() == 'Drought Trends Over Time'
        
        # Check that there are two y-axes
        assert len(fig.axes) == 2
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_drought_trends(
            sample_trend_data, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_drought_classification_heatmap(self, sample_sswei_data):
        """
        Test the plot_drought_classification_heatmap function.
        
        Parameters
        ----------
        sample_sswei_data : pandas.DataFrame
            Sample SSWEI DataFrame.
        """
        # Plot drought classification heatmap
        fig = visualization.plot_drought_classification_heatmap(sample_sswei_data)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Drought Classification'
        assert ax.get_ylabel() == 'Decade'
        assert ax.get_title() == 'Drought Classifications by Decade'
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_drought_classification_heatmap(
            sample_sswei_data, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_drought_severity_distribution(self, sample_sswei_data):
        """
        Test the plot_drought_severity_distribution function.
        
        Parameters
        ----------
        sample_sswei_data : pandas.DataFrame
            Sample SSWEI DataFrame.
        """
        # Plot drought severity distribution
        fig = visualization.plot_drought_severity_distribution(sample_sswei_data)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Drought Severity'
        assert ax.get_ylabel() == 'Frequency'
        assert ax.get_title() == 'Distribution of Drought Severity'
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_drought_severity_distribution(
            sample_sswei_data, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_seasonal_swe(self, sample_daily_mean):
        """
        Test the plot_seasonal_swe function.
        
        Parameters
        ----------
        sample_daily_mean : pandas.DataFrame
            Sample daily mean SWE DataFrame.
        """
        # Plot seasonal SWE
        fig = visualization.plot_seasonal_swe(sample_daily_mean)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Day of Year'
        assert ax.get_ylabel() == 'SWE (Snow Water Equivalent)'
        assert ax.get_title() == 'Daily SWE by Year'
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_seasonal_swe(
            sample_daily_mean, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
    
    def test_plot_correlation_matrix(self, sample_correlation_matrix):
        """
        Test the plot_correlation_matrix function.
        
        Parameters
        ----------
        sample_correlation_matrix : pandas.DataFrame
            Sample correlation matrix.
        """
        # Plot correlation matrix
        fig = visualization.plot_correlation_matrix(sample_correlation_matrix)
        
        # Check that the result is a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Flow accumulation periods'
        assert ax.get_ylabel() == 'SWE dates (1st)'
        
        # Test with custom figsize
        figsize = (10, 6)
        fig = visualization.plot_correlation_matrix(
            sample_correlation_matrix, figsize=figsize
        )
        
        # Check that the figure has the expected size
        assert fig.get_size_inches().tolist() == list(figsize)
        
        plt.close('all')
