"""
Integration tests for the Snow Drought Index package.

This module contains integration tests that verify the interaction between
different components of the package.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.sswei_class import SSWEI
from snowdroughtindex.core.drought_analysis import DroughtAnalysis
from snowdroughtindex.core.configuration import Configuration


class TestDatasetToSSWEI:
    """
    Test the integration between SWEDataset and SSWEI classes.
    """
    
    def test_dataset_to_sswei_workflow(self, sample_swe_dataframe):
        """
        Test the workflow from SWEDataset to SSWEI calculation.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create a SWEDataset object
        dataset = SWEDataset(sample_swe_dataframe)
        
        # Create an SSWEI object using the dataset
        sswei_obj = SSWEI(dataset)
        
        # Calculate SSWEI
        sswei_obj.calculate_sswei(
            start_month=12,
            end_month=3,
            min_years=2,  # Using a small value for testing
            distribution='gamma'
        )
        
        # Check that SSWEI data was calculated
        assert sswei_obj.sswei_data is not None
        
        # Check that the SSWEI data has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI', 'Drought_Classification']
        assert all(col in sswei_obj.sswei_data.columns for col in expected_columns)
        
        # Check that the SSWEI values are finite
        assert all(np.isfinite(sswei_obj.sswei_data['SWEI']))
        
        # Check that drought classifications were assigned
        assert all(isinstance(cls, str) for cls in sswei_obj.sswei_data['Drought_Classification'])
    
    def test_gap_filling_to_sswei_workflow(self, sample_swe_dataframe):
        """
        Test the workflow from gap filling to SSWEI calculation.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create a SWEDataset object
        dataset = SWEDataset(sample_swe_dataframe)
        
        # Introduce some artificial gaps
        df_with_gaps = sample_swe_dataframe.copy()
        np.random.seed(42)
        mask = np.random.random(df_with_gaps.shape) < 0.1  # 10% of data as gaps
        df_with_gaps = df_with_gaps.mask(mask)
        
        # Create a SWEDataset with gaps
        dataset_with_gaps = SWEDataset(df_with_gaps)
        
        # Perform gap filling
        dataset_with_gaps.gap_fill(
            window_days=15,
            min_obs_corr=5,  # Using a small value for testing
            min_obs_cdf=3,   # Using a small value for testing
            min_corr=0.5     # Using a small value for testing
        )
        
        # Create an SSWEI object using the gap-filled dataset
        sswei_obj = SSWEI(dataset_with_gaps)
        
        # Calculate SSWEI
        sswei_obj.calculate_sswei(
            start_month=12,
            end_month=3,
            min_years=2,  # Using a small value for testing
            distribution='gamma'
        )
        
        # Check that SSWEI data was calculated
        assert sswei_obj.sswei_data is not None
        
        # Check that the SSWEI data has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI', 'Drought_Classification']
        assert all(col in sswei_obj.sswei_data.columns for col in expected_columns)
        
        # Check that the SSWEI values are finite
        assert all(np.isfinite(sswei_obj.sswei_data['SWEI']))


class TestSSWEIToDroughtAnalysis:
    """
    Test the integration between SSWEI and DroughtAnalysis classes.
    """
    
    def test_sswei_to_drought_analysis_workflow(self, sample_swe_dataframe):
        """
        Test the workflow from SSWEI to DroughtAnalysis.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create datasets for different elevation bands
        np.random.seed(42)
        
        # Create three datasets with different elevation characteristics
        low_elev = sample_swe_dataframe.copy()
        mid_elev = sample_swe_dataframe.copy() * 1.2  # More SWE
        high_elev = sample_swe_dataframe.copy() * 1.5  # Even more SWE
        
        # Create SWEDataset objects
        low_dataset = SWEDataset(low_elev)
        mid_dataset = SWEDataset(mid_elev)
        high_dataset = SWEDataset(high_elev)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets for different elevation bands
        drought_analysis.add_dataset("1000-1500m", low_dataset)
        drought_analysis.add_dataset("1500-2000m", mid_dataset)
        drought_analysis.add_dataset("2000-2500m", high_dataset)
        
        # Calculate SSWEI for all datasets
        drought_analysis.calculate_sswei(
            start_month=12,
            end_month=3,
            min_years=2,  # Using a small value for testing
            distribution='gamma'
        )
        
        # Check that SSWEI objects were created
        assert len(drought_analysis.sswei_objects) == 3
        assert all(isinstance(obj, SSWEI) for obj in drought_analysis.sswei_objects.values())
        
        # Compare elevation bands
        comparison_df = drought_analysis.compare_elevation_bands()
        
        # Check that the comparison DataFrame has the expected columns
        expected_columns = ['Elevation_Band', 'Total_Years', 'Drought_Count', 
                           'Severe_Drought_Count', 'Drought_Frequency', 'Mean_Severity']
        assert all(col in comparison_df.columns for col in expected_columns)
        
        # Check that all elevation bands are included
        assert set(comparison_df['Elevation_Band']) == {"1000-1500m", "1500-2000m", "2000-2500m"}
        
        # Analyze drought synchronicity
        sync_data = drought_analysis.analyze_drought_synchronicity()
        
        # Check that the synchronicity DataFrame has the expected columns
        assert 'Datasets_in_Drought' in sync_data.columns
        assert 'Percent_in_Drought' in sync_data.columns
        
        # Analyze elevation sensitivity
        sensitivity_df = drought_analysis.analyze_elevation_sensitivity()
        
        # Check that the sensitivity DataFrame has the expected columns
        expected_columns = ['Elevation_Band', 'Mid_Elevation', 'Drought_Frequency', 
                           'Mean_Severity', 'Severe_Drought_Count']
        assert all(col in sensitivity_df.columns for col in expected_columns)


class TestConfigurationIntegration:
    """
    Test the integration of the Configuration class with other components.
    """
    
    def test_configuration_with_sswei_workflow(self, sample_swe_dataframe):
        """
        Test using Configuration with SSWEI calculation.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create a Configuration object with custom settings
        config = Configuration()
        config.set('sswei', 'start_month', 12)
        config.set('sswei', 'end_month', 3)
        config.set('sswei', 'min_years', 2)  # Using a small value for testing
        config.set('sswei', 'distribution', 'gamma')
        
        # Set custom drought classification thresholds
        config.set('drought_classification', 'exceptional', -2.0)
        config.set('drought_classification', 'extreme', -1.5)
        config.set('drought_classification', 'severe', -1.0)
        config.set('drought_classification', 'moderate', -0.5)
        
        # Create a SWEDataset object
        dataset = SWEDataset(sample_swe_dataframe)
        
        # Create an SSWEI object
        sswei_obj = SSWEI(dataset)
        
        # Set custom thresholds from configuration
        sswei_obj.set_thresholds(config.get_drought_classification_thresholds())
        
        # Calculate SSWEI using parameters from configuration
        sswei_params = config.get_sswei_params()
        sswei_obj.calculate_sswei(
            start_month=sswei_params['start_month'],
            end_month=sswei_params['end_month'],
            min_years=sswei_params['min_years'],
            distribution=sswei_params['distribution']
        )
        
        # Check that SSWEI data was calculated
        assert sswei_obj.sswei_data is not None
        
        # Check that the SSWEI data has the expected columns
        expected_columns = ['season_year', 'integrated_swe', 'SWEI', 'Drought_Classification']
        assert all(col in sswei_obj.sswei_data.columns for col in expected_columns)
    
    def test_configuration_with_gap_filling_workflow(self, sample_swe_dataframe):
        """
        Test using Configuration with gap filling.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create a Configuration object with custom settings
        config = Configuration()
        config.set('gap_filling', 'window_days', 15)
        config.set('gap_filling', 'min_obs_corr', 5)  # Using a small value for testing
        config.set('gap_filling', 'min_obs_cdf', 3)   # Using a small value for testing
        config.set('gap_filling', 'min_corr', 0.5)    # Using a small value for testing
        
        # Introduce some artificial gaps
        df_with_gaps = sample_swe_dataframe.copy()
        np.random.seed(42)
        mask = np.random.random(df_with_gaps.shape) < 0.1  # 10% of data as gaps
        df_with_gaps = df_with_gaps.mask(mask)
        
        # Create a SWEDataset with gaps
        dataset_with_gaps = SWEDataset(df_with_gaps)
        
        # Perform gap filling using parameters from configuration
        gap_filling_params = config.get_gap_filling_params()
        dataset_with_gaps.gap_fill(
            window_days=gap_filling_params['window_days'],
            min_obs_corr=gap_filling_params['min_obs_corr'],
            min_obs_cdf=gap_filling_params['min_obs_cdf'],
            min_corr=gap_filling_params['min_corr']
        )
        
        # Check that gap filling was performed
        assert dataset_with_gaps.data is not None
        assert dataset_with_gaps.data_type_flags is not None
        assert dataset_with_gaps.donor_stations is not None


class TestEndToEndWorkflow:
    """
    Test the end-to-end workflow from data loading to drought analysis.
    """
    
    def test_end_to_end_workflow(self, sample_swe_dataframe, sample_stations):
        """
        Test the complete workflow from data loading to drought analysis.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        """
        # Create a Configuration object
        config = Configuration()
        
        # Set gap filling parameters
        config.set('gap_filling', 'window_days', 15)
        config.set('gap_filling', 'min_obs_corr', 5)  # Using a small value for testing
        config.set('gap_filling', 'min_obs_cdf', 3)   # Using a small value for testing
        config.set('gap_filling', 'min_corr', 0.5)    # Using a small value for testing
        
        # Set SSWEI parameters
        config.set('sswei', 'start_month', 12)
        config.set('sswei', 'end_month', 3)
        config.set('sswei', 'min_years', 2)  # Using a small value for testing
        config.set('sswei', 'distribution', 'gamma')
        
        # Create datasets for different elevation bands
        np.random.seed(42)
        
        # Create three datasets with different elevation characteristics
        low_elev = sample_swe_dataframe.copy()
        mid_elev = sample_swe_dataframe.copy() * 1.2  # More SWE
        high_elev = sample_swe_dataframe.copy() * 1.5  # Even more SWE
        
        # Introduce some artificial gaps
        for df in [low_elev, mid_elev, high_elev]:
            mask = np.random.random(df.shape) < 0.1  # 10% of data as gaps
            df.mask(mask, inplace=True)
        
        # Create SWEDataset objects
        low_dataset = SWEDataset(low_elev, sample_stations.copy())
        mid_dataset = SWEDataset(mid_elev, sample_stations.copy())
        high_dataset = SWEDataset(high_elev, sample_stations.copy())
        
        # Perform gap filling using parameters from configuration
        gap_filling_params = config.get_gap_filling_params()
        for dataset in [low_dataset, mid_dataset, high_dataset]:
            dataset.gap_fill(
                window_days=gap_filling_params['window_days'],
                min_obs_corr=gap_filling_params['min_obs_corr'],
                min_obs_cdf=gap_filling_params['min_obs_cdf'],
                min_corr=gap_filling_params['min_corr']
            )
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets for different elevation bands
        drought_analysis.add_dataset("1000-1500m", low_dataset)
        drought_analysis.add_dataset("1500-2000m", mid_dataset)
        drought_analysis.add_dataset("2000-2500m", high_dataset)
        
        # Calculate SSWEI for all datasets using parameters from configuration
        sswei_params = config.get_sswei_params()
        drought_analysis.calculate_sswei(
            start_month=sswei_params['start_month'],
            end_month=sswei_params['end_month'],
            min_years=sswei_params['min_years'],
            distribution=sswei_params['distribution']
        )
        
        # Perform various analyses
        comparison_df = drought_analysis.compare_elevation_bands()
        sync_data = drought_analysis.analyze_drought_synchronicity()
        sensitivity_df = drought_analysis.analyze_elevation_sensitivity()
        temporal_trends = drought_analysis.analyze_temporal_trends(window_size=2)  # Small window for testing
        
        # Check that all analyses produced results
        assert comparison_df is not None and not comparison_df.empty
        assert sync_data is not None and not sync_data.empty
        assert sensitivity_df is not None and not sensitivity_df.empty
        assert temporal_trends is not None and len(temporal_trends) > 0
        
        # Check that plots can be generated
        fig1 = drought_analysis.plot_elevation_band_comparison()
        fig2, fig3 = drought_analysis.plot_drought_synchronicity()
        fig4 = drought_analysis.plot_elevation_sensitivity()
        fig5 = drought_analysis.plot_temporal_trends()
        
        # Check that all figures were created
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        assert isinstance(fig3, plt.Figure)
        assert isinstance(fig4, plt.Figure)
        assert isinstance(fig5, plt.Figure)
        
        # Close all figures to free memory
        plt.close('all')
