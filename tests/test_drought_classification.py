"""
Unit tests for the drought classification module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from snowdroughtindex.core import drought_classification

class TestDroughtClassification:
    """
    Test class for the drought classification module.
    """
    
    def test_classify_drought(self, sample_swe_dataframe):
        """
        Test the classify_drought function.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create a sample SSWEI DataFrame
        np.random.seed(42)
        years = range(2010, 2020)
        sswei_values = np.random.normal(0, 1, len(years))
        
        sswei_df = pd.DataFrame({
            'season_year': years,
            'integrated_swe': np.random.rand(len(years)) * 100,
            'SWEI': sswei_values
        })
        
        # Classify drought
        drought_df = drought_classification.classify_drought(sswei_df)
        
        # Check that the result is a DataFrame
        assert isinstance(drought_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI', 'drought_class', 'drought_category']
        assert all(col in drought_df.columns for col in expected_columns)
        
        # Check that the DataFrame has one row per year
        assert len(drought_df) == len(sswei_df)
        
        # Check that the drought_class values are integers
        assert all(isinstance(val, (int, np.integer)) for val in drought_df['drought_class'])
        
        # Check that the drought_category values are strings
        assert all(isinstance(val, str) for val in drought_df['drought_category'])
    
    def test_classify_drought_with_custom_thresholds(self, sample_swe_dataframe):
        """
        Test the classify_drought function with custom thresholds.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create a sample SSWEI DataFrame
        np.random.seed(42)
        years = range(2010, 2020)
        sswei_values = np.random.normal(0, 1, len(years))
        
        sswei_df = pd.DataFrame({
            'season_year': years,
            'integrated_swe': np.random.rand(len(years)) * 100,
            'SWEI': sswei_values
        })
        
        # Define custom thresholds
        thresholds = {
            'exceptional': -2.5,
            'extreme': -2.0,
            'severe': -1.5,
            'moderate': -1.0
        }
        
        # Classify drought with custom thresholds
        drought_df = drought_classification.classify_drought(sswei_df, thresholds=thresholds)
        
        # Check that the result is a DataFrame
        assert isinstance(drought_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI', 'drought_class', 'drought_category']
        assert all(col in drought_df.columns for col in expected_columns)
        
        # Check that the DataFrame has one row per year
        assert len(drought_df) == len(sswei_df)
        
        # Check that the drought_class values are integers
        assert all(isinstance(val, (int, np.integer)) for val in drought_df['drought_class'])
        
        # Check that the drought_category values are strings
        assert all(isinstance(val, str) for val in drought_df['drought_category'])
        
        # Check that the classification is consistent with the custom thresholds
        for _, row in drought_df.iterrows():
            sswei = row['SWEI']
            drought_class = row['drought_class']
            drought_category = row['drought_category']
            
            if sswei <= thresholds['exceptional']:
                assert drought_class == 4
                assert drought_category == 'Exceptional'
            elif sswei <= thresholds['extreme']:
                assert drought_class == 3
                assert drought_category == 'Extreme'
            elif sswei <= thresholds['severe']:
                assert drought_class == 2
                assert drought_category == 'Severe'
            elif sswei <= thresholds['moderate']:
                assert drought_class == 1
                assert drought_category == 'Moderate'
            else:
                assert drought_class == 0
                assert drought_category == 'Normal'
    
    def test_get_drought_duration(self):
        """
        Test the get_drought_duration function.
        """
        # Create a sample drought classification DataFrame
        years = range(2010, 2020)
        drought_classes = [0, 0, 1, 2, 2, 1, 0, 0, 3, 2]  # Some drought periods
        
        drought_df = pd.DataFrame({
            'season_year': years,
            'drought_class': drought_classes
        })
        
        # Get drought duration
        duration_df = drought_classification.get_drought_duration(drought_df)
        
        # Check that the result is a DataFrame
        assert isinstance(duration_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['start_year', 'end_year', 'duration', 'max_class', 'max_category']
        assert all(col in duration_df.columns for col in expected_columns)
        
        # Check that the duration values are integers
        assert all(isinstance(val, (int, np.integer)) for val in duration_df['duration'])
        
        # Check that the max_class values are integers
        assert all(isinstance(val, (int, np.integer)) for val in duration_df['max_class'])
        
        # Check that the max_category values are strings
        assert all(isinstance(val, str) for val in duration_df['max_category'])
        
        # Check that the number of drought periods is correct
        # We have two drought periods: 2012-2015 and 2018-2019
        assert len(duration_df) == 2
        
        # Check that the duration of the first drought period is correct
        assert duration_df.iloc[0]['duration'] == 4
        
        # Check that the duration of the second drought period is correct
        assert duration_df.iloc[1]['duration'] == 2
    
    def test_get_drought_severity(self):
        """
        Test the get_drought_severity function.
        """
        # Create a sample drought classification DataFrame
        years = range(2010, 2020)
        drought_classes = [0, 0, 1, 2, 2, 1, 0, 0, 3, 2]  # Some drought periods
        sswei_values = [0.5, 0.2, -0.6, -1.2, -1.3, -0.7, 0.1, 0.3, -1.8, -1.4]
        
        drought_df = pd.DataFrame({
            'season_year': years,
            'drought_class': drought_classes,
            'SWEI': sswei_values
        })
        
        # Get drought severity
        severity_df = drought_classification.get_drought_severity(drought_df)
        
        # Check that the result is a DataFrame
        assert isinstance(severity_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['start_year', 'end_year', 'duration', 'severity', 'intensity']
        assert all(col in severity_df.columns for col in expected_columns)
        
        # Check that the duration values are integers
        assert all(isinstance(val, (int, np.integer)) for val in severity_df['duration'])
        
        # Check that the severity values are floats
        assert all(isinstance(val, (float, np.floating)) for val in severity_df['severity'])
        
        # Check that the intensity values are floats
        assert all(isinstance(val, (float, np.floating)) for val in severity_df['intensity'])
        
        # Check that the number of drought periods is correct
        # We have two drought periods: 2012-2015 and 2018-2019
        assert len(severity_df) == 2
        
        # Check that the severity of the first drought period is correct
        # Severity is the sum of absolute SSWEI values during the drought period
        first_period_severity = abs(-0.6) + abs(-1.2) + abs(-1.3) + abs(-0.7)
        assert abs(severity_df.iloc[0]['severity'] - first_period_severity) < 1e-6
        
        # Check that the intensity of the first drought period is correct
        # Intensity is the severity divided by the duration
        first_period_intensity = first_period_severity / 4
        assert abs(severity_df.iloc[0]['intensity'] - first_period_intensity) < 1e-6
    
    def test_get_drought_frequency(self):
        """
        Test the get_drought_frequency function.
        """
        # Create a sample drought classification DataFrame
        years = range(2010, 2020)
        drought_classes = [0, 0, 1, 2, 2, 1, 0, 0, 3, 2]  # Some drought periods
        
        drought_df = pd.DataFrame({
            'season_year': years,
            'drought_class': drought_classes
        })
        
        # Get drought frequency
        frequency = drought_classification.get_drought_frequency(drought_df)
        
        # Check that the result is a float
        assert isinstance(frequency, float)
        
        # Check that the frequency is between 0 and 1
        assert 0 <= frequency <= 1
        
        # Check that the frequency is correct
        # We have 6 years with drought out of 10 years total
        expected_frequency = 6 / 10
        assert abs(frequency - expected_frequency) < 1e-6
    
    def test_get_drought_class_distribution(self):
        """
        Test the get_drought_class_distribution function.
        """
        # Create a sample drought classification DataFrame
        years = range(2010, 2020)
        drought_classes = [0, 0, 1, 2, 2, 1, 0, 0, 3, 2]  # Some drought periods
        
        drought_df = pd.DataFrame({
            'season_year': years,
            'drought_class': drought_classes
        })
        
        # Get drought class distribution
        distribution = drought_classification.get_drought_class_distribution(drought_df)
        
        # Check that the result is a dictionary
        assert isinstance(distribution, dict)
        
        # Check that the dictionary contains the expected keys
        expected_keys = [0, 1, 2, 3, 4]  # All possible drought classes
        assert all(key in distribution for key in expected_keys)
        
        # Check that the distribution values are between 0 and 1
        assert all(0 <= val <= 1 for val in distribution.values())
        
        # Check that the distribution sums to 1
        assert abs(sum(distribution.values()) - 1) < 1e-6
        
        # Check that the distribution is correct
        expected_distribution = {
            0: 4 / 10,  # 4 years with class 0
            1: 2 / 10,  # 2 years with class 1
            2: 3 / 10,  # 3 years with class 2
            3: 1 / 10,  # 1 year with class 3
            4: 0 / 10   # 0 years with class 4
        }
        for key, val in expected_distribution.items():
            assert abs(distribution[key] - val) < 1e-6
