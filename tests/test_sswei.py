"""
Unit tests for the SSWEI module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from snowdroughtindex.core import sswei

class TestSSWEI:
    """
    Test class for the SSWEI module.
    """
    
    def test_prepare_seasonal_data(self, sample_swe_dataframe):
        """
        Test the prepare_seasonal_data function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Prepare seasonal data for December to March
        start_month = 12
        end_month = 3
        min_years = 2
        
        seasonal_data = sswei.prepare_seasonal_data(
            sample_swe_dataframe, start_month, end_month, min_years
        )
        
        # Check that the result is a dictionary
        assert isinstance(seasonal_data, dict)
        
        # Check that the dictionary contains the expected keys
        expected_keys = ['data', 'years', 'stations']
        assert all(key in seasonal_data for key in expected_keys)
        
        # Check that the data is a DataFrame
        assert isinstance(seasonal_data['data'], pd.DataFrame)
        
        # Check that the years is a list
        assert isinstance(seasonal_data['years'], list)
        
        # Check that the stations is a list
        assert isinstance(seasonal_data['stations'], list)
        
        # Check that the data contains only dates in the specified months
        months = [date.month for date in seasonal_data['data'].index]
        assert all(month in [12, 1, 2, 3] for month in months)
        
        # Check that the stations match the columns in the data
        assert seasonal_data['stations'] == list(seasonal_data['data'].columns)
    
    def test_integrate_season(self, sample_swe_dataframe):
        """
        Test the integrate_season function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Prepare seasonal data
        start_month = 12
        end_month = 3
        min_years = 2
        
        seasonal_data = sswei.prepare_seasonal_data(
            sample_swe_dataframe, start_month, end_month, min_years
        )
        
        # Integrate the seasonal data
        integrated_swe = sswei.integrate_season(seasonal_data)
        
        # Check that the result is a DataFrame
        assert isinstance(integrated_swe, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['season_year', 'integrated_swe']
        assert all(col in integrated_swe.columns for col in expected_columns)
        
        # Check that the DataFrame has one row per year
        assert len(integrated_swe) == len(seasonal_data['years'])
        
        # Check that the integrated_swe values are non-negative
        assert all(integrated_swe['integrated_swe'] >= 0)
    
    def test_calculate_sswei(self, sample_swe_dataframe):
        """
        Test the calculate_sswei function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Prepare seasonal data
        start_month = 12
        end_month = 3
        min_years = 2
        
        seasonal_data = sswei.prepare_seasonal_data(
            sample_swe_dataframe, start_month, end_month, min_years
        )
        
        # Integrate the seasonal data
        integrated_swe = sswei.integrate_season(seasonal_data)
        
        # Calculate SSWEI
        sswei_data = sswei.calculate_sswei(integrated_swe, distribution='gamma')
        
        # Check that the result is a DataFrame
        assert isinstance(sswei_data, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI']
        assert all(col in sswei_data.columns for col in expected_columns)
        
        # Check that the DataFrame has one row per year
        assert len(sswei_data) == len(integrated_swe)
        
        # Check that the SWEI values are finite
        assert all(np.isfinite(sswei_data['SWEI']))
    
    def test_calculate_sswei_with_reference_period(self, sample_swe_dataframe):
        """
        Test the calculate_sswei function with a reference period.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Prepare seasonal data
        start_month = 12
        end_month = 3
        min_years = 2
        
        seasonal_data = sswei.prepare_seasonal_data(
            sample_swe_dataframe, start_month, end_month, min_years
        )
        
        # Integrate the seasonal data
        integrated_swe = sswei.integrate_season(seasonal_data)
        
        # Define a reference period
        reference_period = (2010, 2011)  # First two years
        
        # Calculate SSWEI with reference period
        sswei_data = sswei.calculate_sswei(
            integrated_swe, distribution='gamma', reference_period=reference_period
        )
        
        # Check that the result is a DataFrame
        assert isinstance(sswei_data, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI']
        assert all(col in sswei_data.columns for col in expected_columns)
        
        # Check that the DataFrame has one row per year
        assert len(sswei_data) == len(integrated_swe)
        
        # Check that the SWEI values are finite
        assert all(np.isfinite(sswei_data['SWEI']))
    
    def test_calculate_sswei_with_normal_distribution(self, sample_swe_dataframe):
        """
        Test the calculate_sswei function with a normal distribution.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Prepare seasonal data
        start_month = 12
        end_month = 3
        min_years = 2
        
        seasonal_data = sswei.prepare_seasonal_data(
            sample_swe_dataframe, start_month, end_month, min_years
        )
        
        # Integrate the seasonal data
        integrated_swe = sswei.integrate_season(seasonal_data)
        
        # Calculate SSWEI with normal distribution
        sswei_data = sswei.calculate_sswei(integrated_swe, distribution='normal')
        
        # Check that the result is a DataFrame
        assert isinstance(sswei_data, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI']
        assert all(col in sswei_data.columns for col in expected_columns)
        
        # Check that the DataFrame has one row per year
        assert len(sswei_data) == len(integrated_swe)
        
        # Check that the SWEI values are finite
        assert all(np.isfinite(sswei_data['SWEI']))
    
    def test_fit_gamma_distribution(self):
        """
        Test the fit_gamma_distribution function.
        """
        # Create sample data
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # Fit gamma distribution
        shape, scale = sswei.fit_gamma_distribution(data)
        
        # Check that the shape and scale are positive
        assert shape > 0
        assert scale > 0
        
        # Check that the shape and scale are reasonable
        assert 0.1 < shape < 100
        assert 0.1 < scale < 1000
    
    def test_gamma_cdf(self):
        """
        Test the gamma_cdf function.
        """
        # Create sample data
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # Fit gamma distribution
        shape, scale = sswei.fit_gamma_distribution(data)
        
        # Calculate gamma CDF
        cdf = sswei.gamma_cdf(data, shape, scale)
        
        # Check that the result is a numpy array
        assert isinstance(cdf, np.ndarray)
        
        # Check that the CDF values are between 0 and 1
        assert all(0 <= p <= 1 for p in cdf)
        
        # Check that the CDF is monotonically increasing
        assert all(cdf[i] <= cdf[i+1] for i in range(len(cdf)-1))
    
    def test_normal_cdf(self):
        """
        Test the normal_cdf function.
        """
        # Create sample data
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # Calculate normal CDF
        cdf = sswei.normal_cdf(data)
        
        # Check that the result is a numpy array
        assert isinstance(cdf, np.ndarray)
        
        # Check that the CDF values are between 0 and 1
        assert all(0 <= p <= 1 for p in cdf)
        
        # Check that the CDF is monotonically increasing
        assert all(cdf[i] <= cdf[i+1] for i in range(len(cdf)-1))
