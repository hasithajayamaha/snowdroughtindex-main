"""
Unit tests for the data preparation module.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from snowdroughtindex.core import data_preparation

class TestDataPreparation:
    """
    Test class for the data preparation module.
    """
    
    def test_preprocess_swe(self, sample_swe_dataset):
        """
        Test the preprocess_swe function.
        
        Parameters
        ----------
        sample_swe_dataset : xarray.Dataset
            Sample SWE dataset.
        """
        # Preprocess the dataset
        df = data_preparation.preprocess_swe(sample_swe_dataset)
        
        # Check that the result is a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = [f'station_{i}' for i in range(1, 6)]
        assert all(col in df.columns for col in expected_columns)
        
        # Check that the DataFrame has the expected index
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check that the DataFrame has the expected number of rows
        assert len(df) == len(sample_swe_dataset.time)
    
    def test_extract_stations_in_basin(self, sample_stations, sample_basins):
        """
        Test the extract_stations_in_basin function.
        
        Parameters
        ----------
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        sample_basins : pandas.DataFrame
            Sample basins DataFrame.
        """
        # Extract stations in the first basin
        basin_id = sample_basins['basin_id'].iloc[0]
        
        # Add a basin_id column to the stations DataFrame for testing
        sample_stations['basin_id'] = np.random.choice(
            sample_basins['basin_id'].values, 
            size=len(sample_stations)
        )
        
        # Ensure at least one station is in the target basin
        sample_stations.loc[0, 'basin_id'] = basin_id
        
        # Extract stations in the basin
        stations_in_basin, basin_info = data_preparation.extract_stations_in_basin(
            sample_stations, sample_basins, basin_id, buffer_km=0
        )
        
        # Check that the result is a DataFrame
        assert isinstance(stations_in_basin, pd.DataFrame)
        
        # Check that the basin info is a Series
        assert isinstance(basin_info, pd.Series)
        
        # Check that the basin info has the expected basin_id
        assert basin_info['basin_id'] == basin_id
        
        # Check that all stations in the result have the correct basin_id
        assert all(station['basin_id'] == basin_id for _, station in stations_in_basin.iterrows())
    
    def test_extract_monthly_data(self, sample_swe_dataframe):
        """
        Test the extract_monthly_data function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Extract data for January
        month = 1
        monthly_data = data_preparation.extract_monthly_data(sample_swe_dataframe, month, plot=False)
        
        # Check that the result is a DataFrame
        assert isinstance(monthly_data, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = [f'station_{i}' for i in range(1, 6)]
        assert all(col in monthly_data.columns for col in expected_columns)
        
        # Check that all dates in the result are in the specified month
        assert all(date.month == month for date in monthly_data.index)
        
        # Check that there is one entry per year
        years = [date.year for date in monthly_data.index]
        assert len(years) == len(set(years))
    
    def test_calculate_data_availability(self, sample_swe_dataframe, sample_stations):
        """
        Test the calculate_data_availability function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        """
        # Calculate data availability
        availability = data_preparation.calculate_data_availability(
            sample_stations, sample_swe_dataframe
        )
        
        # Check that the result is a DataFrame
        assert isinstance(availability, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['year', 'month', 'percentage']
        assert all(col in availability.columns for col in expected_columns)
        
        # Check that the percentage values are between 0 and 100
        assert all(0 <= p <= 100 for p in availability['percentage'])
        
        # Check that there is one entry per year-month combination
        year_months = [(row['year'], row['month']) for _, row in availability.iterrows()]
        assert len(year_months) == len(set(year_months))
    
    def test_add_artificial_gaps(self, sample_swe_dataframe):
        """
        Test the add_artificial_gaps function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Add artificial gaps
        gap_percentage = 20
        df_with_gaps = data_preparation.add_artificial_gaps(
            sample_swe_dataframe, gap_percentage
        )
        
        # Check that the result is a DataFrame
        assert isinstance(df_with_gaps, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = [f'station_{i}' for i in range(1, 6)]
        assert all(col in df_with_gaps.columns for col in expected_columns)
        
        # Check that the DataFrame has the expected number of rows
        assert len(df_with_gaps) == len(sample_swe_dataframe)
        
        # Check that the DataFrame has the expected number of NaN values
        total_cells = df_with_gaps.size
        nan_cells = df_with_gaps.isna().sum().sum()
        actual_gap_percentage = (nan_cells / total_cells) * 100
        
        # Allow for some randomness in the gap percentage
        assert abs(actual_gap_percentage - gap_percentage) < 5
