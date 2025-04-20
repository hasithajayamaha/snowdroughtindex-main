"""
Unit tests for the gap filling module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from snowdroughtindex.core import gap_filling

class TestGapFilling:
    """
    Test class for the gap filling module.
    """
    
    def test_find_donor_stations(self, sample_swe_dataframe):
        """
        Test the find_donor_stations function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Add some NaN values to the sample data
        df_with_gaps = sample_swe_dataframe.copy()
        df_with_gaps.iloc[10:20, 0] = np.nan  # Add gaps to the first station
        
        # Find donor stations for the first station
        target_station = df_with_gaps.columns[0]
        donor_stations = gap_filling.find_donor_stations(
            df_with_gaps, target_station, min_obs_corr=5, min_corr=0.5
        )
        
        # Check that the result is a list
        assert isinstance(donor_stations, list)
        
        # Check that the target station is not in the donor stations
        assert target_station not in donor_stations
        
        # Check that all donor stations are in the original DataFrame
        assert all(station in df_with_gaps.columns for station in donor_stations)
    
    def test_calculate_correlation(self, sample_swe_dataframe):
        """
        Test the calculate_correlation function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Calculate correlation between the first two stations
        station1 = sample_swe_dataframe.columns[0]
        station2 = sample_swe_dataframe.columns[1]
        
        corr = gap_filling.calculate_correlation(
            sample_swe_dataframe[station1], sample_swe_dataframe[station2]
        )
        
        # Check that the result is a float
        assert isinstance(corr, float)
        
        # Check that the correlation is between -1 and 1
        assert -1 <= corr <= 1
    
    def test_calculate_cdf(self, sample_swe_dataframe):
        """
        Test the calculate_cdf function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Calculate CDF for the first station
        station = sample_swe_dataframe.columns[0]
        values = sample_swe_dataframe[station].values
        
        cdf = gap_filling.calculate_cdf(values)
        
        # Check that the result is a numpy array
        assert isinstance(cdf, np.ndarray)
        
        # Check that the CDF values are between 0 and 1
        assert all(0 <= p <= 1 for p in cdf)
        
        # Check that the CDF has the same length as the input values
        assert len(cdf) == len(values)
    
    def test_qm_gap_filling(self, sample_swe_dataframe):
        """
        Test the qm_gap_filling function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Add some NaN values to the sample data
        df_with_gaps = sample_swe_dataframe.copy()
        df_with_gaps.iloc[10:20, 0] = np.nan  # Add gaps to the first station
        df_with_gaps.iloc[30:40, 1] = np.nan  # Add gaps to the second station
        
        # Perform gap filling
        filled_data, data_type_flags, donor_stations = gap_filling.qm_gap_filling(
            df_with_gaps, window_days=15, min_obs_corr=5, min_obs_cdf=5, min_corr=0.5
        )
        
        # Check that the result is a DataFrame
        assert isinstance(filled_data, pd.DataFrame)
        
        # Check that the data_type_flags is a DataFrame
        assert isinstance(data_type_flags, pd.DataFrame)
        
        # Check that the donor_stations is a dictionary
        assert isinstance(donor_stations, dict)
        
        # Check that the filled data has the same shape as the input data
        assert filled_data.shape == df_with_gaps.shape
        
        # Check that the data_type_flags has the same shape as the input data
        assert data_type_flags.shape == df_with_gaps.shape
        
        # Check that the number of NaN values in the filled data is less than in the input data
        assert filled_data.isna().sum().sum() <= df_with_gaps.isna().sum().sum()
    
    def test_artificial_gap_filling(self, sample_swe_dataframe):
        """
        Test the artificial_gap_filling function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Perform artificial gap filling
        evaluation = gap_filling.artificial_gap_filling(
            sample_swe_dataframe, iterations=2, artificial_gap_perc=10,
            window_days=15, min_obs_corr=5, min_obs_cdf=5, min_corr=0.5,
            min_obs_KGE=5, flag=0
        )
        
        # Check that the result is a dictionary
        assert isinstance(evaluation, dict)
        
        # Check that the evaluation contains the expected keys
        expected_keys = ['KGE', 'RMSE', 'MAE', 'BIAS', 'R']
        assert all(key in evaluation for key in expected_keys)
        
        # Check that the evaluation metrics are numpy arrays
        assert all(isinstance(evaluation[key], np.ndarray) for key in expected_keys)
        
        # Check that the evaluation metrics have the expected shape (iterations, stations)
        assert all(evaluation[key].shape == (2, len(sample_swe_dataframe.columns)) for key in expected_keys)
    
    def test_calculate_KGE(self):
        """
        Test the calculate_KGE function.
        """
        # Create sample observed and simulated data
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 2.2, 2.8, 3.9, 5.1])
        
        # Calculate KGE
        kge = gap_filling.calculate_KGE(observed, simulated)
        
        # Check that the result is a float
        assert isinstance(kge, float)
        
        # Check that the KGE is between -infinity and 1
        assert kge <= 1
        
        # Check that the KGE is close to 1 for similar data
        assert kge > 0.9
        
        # Check that the KGE is lower for dissimilar data
        dissimilar = np.array([5, 4, 3, 2, 1])
        kge_dissimilar = gap_filling.calculate_KGE(observed, dissimilar)
        assert kge_dissimilar < kge
    
    def test_calculate_RMSE(self):
        """
        Test the calculate_RMSE function.
        """
        # Create sample observed and simulated data
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 2.2, 2.8, 3.9, 5.1])
        
        # Calculate RMSE
        rmse = gap_filling.calculate_RMSE(observed, simulated)
        
        # Check that the result is a float
        assert isinstance(rmse, float)
        
        # Check that the RMSE is non-negative
        assert rmse >= 0
        
        # Check that the RMSE is close to 0 for similar data
        assert rmse < 0.5
        
        # Check that the RMSE is higher for dissimilar data
        dissimilar = np.array([5, 4, 3, 2, 1])
        rmse_dissimilar = gap_filling.calculate_RMSE(observed, dissimilar)
        assert rmse_dissimilar > rmse
    
    def test_calculate_MAE(self):
        """
        Test the calculate_MAE function.
        """
        # Create sample observed and simulated data
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 2.2, 2.8, 3.9, 5.1])
        
        # Calculate MAE
        mae = gap_filling.calculate_MAE(observed, simulated)
        
        # Check that the result is a float
        assert isinstance(mae, float)
        
        # Check that the MAE is non-negative
        assert mae >= 0
        
        # Check that the MAE is close to 0 for similar data
        assert mae < 0.5
        
        # Check that the MAE is higher for dissimilar data
        dissimilar = np.array([5, 4, 3, 2, 1])
        mae_dissimilar = gap_filling.calculate_MAE(observed, dissimilar)
        assert mae_dissimilar > mae
    
    def test_calculate_BIAS(self):
        """
        Test the calculate_BIAS function.
        """
        # Create sample observed and simulated data
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 2.2, 2.8, 3.9, 5.1])
        
        # Calculate BIAS
        bias = gap_filling.calculate_BIAS(observed, simulated)
        
        # Check that the result is a float
        assert isinstance(bias, float)
        
        # Check that the BIAS is close to 0 for similar data
        assert abs(bias) < 0.1
        
        # Check that the BIAS is positive when simulated > observed
        simulated_high = observed * 1.1
        bias_high = gap_filling.calculate_BIAS(observed, simulated_high)
        assert bias_high > 0
        
        # Check that the BIAS is negative when simulated < observed
        simulated_low = observed * 0.9
        bias_low = gap_filling.calculate_BIAS(observed, simulated_low)
        assert bias_low < 0
