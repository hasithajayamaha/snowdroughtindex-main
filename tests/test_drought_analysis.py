"""
Unit tests for the DroughtAnalysis class.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from datetime import datetime

from snowdroughtindex.core.drought_analysis import DroughtAnalysis
from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.sswei_class import SSWEI

class TestDroughtAnalysis:
    """
    Test class for the DroughtAnalysis class.
    """
    
    def test_init(self):
        """
        Test the initialization of the DroughtAnalysis class.
        """
        # Initialize a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Check that the attributes are initialized correctly
        assert isinstance(drought_analysis.datasets, dict)
        assert isinstance(drought_analysis.sswei_objects, dict)
        assert isinstance(drought_analysis.elevation_bands, list)
        assert isinstance(drought_analysis.regions, list)
        assert isinstance(drought_analysis.analysis_results, dict)
        assert len(drought_analysis.datasets) == 0
        assert len(drought_analysis.sswei_objects) == 0
        assert len(drought_analysis.elevation_bands) == 0
        assert len(drought_analysis.regions) == 0
        assert len(drought_analysis.analysis_results) == 0
    
    def test_add_dataset(self, sample_swe_dataframe, sample_stations):
        """
        Test the add_dataset method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add a dataset as an elevation band
        result = drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        
        # Check that the result is the DroughtAnalysis object
        assert result is drought_analysis
        
        # Check that the dataset is added correctly
        assert "1500-2000m" in drought_analysis.datasets
        assert drought_analysis.datasets["1500-2000m"] is dataset
        assert "1500-2000m" in drought_analysis.elevation_bands
        assert "1500-2000m" not in drought_analysis.regions
        
        # Add a dataset as a region
        result = drought_analysis.add_dataset("Region1", dataset, is_elevation_band=False)
        
        # Check that the result is the DroughtAnalysis object
        assert result is drought_analysis
        
        # Check that the dataset is added correctly
        assert "Region1" in drought_analysis.datasets
        assert drought_analysis.datasets["Region1"] is dataset
        assert "Region1" in drought_analysis.regions
        assert "Region1" not in drought_analysis.elevation_bands
    
    def test_calculate_sswei(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the calculate_sswei method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add a dataset
        drought_analysis.add_dataset("1500-2000m", dataset)
        
        # Mock the SSWEI.calculate_sswei method
        def mock_calculate_sswei(self, start_month, end_month, min_years, distribution, reference_period):
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            self.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            return self
        
        # Apply the mock
        monkeypatch.setattr(SSWEI, "calculate_sswei", mock_calculate_sswei)
        
        # Calculate SSWEI
        result = drought_analysis.calculate_sswei(
            start_month=12,
            end_month=3,
            min_years=10,
            distribution='gamma'
        )
        
        # Check that the result is the DroughtAnalysis object
        assert result is drought_analysis
        
        # Check that the SSWEI objects are created correctly
        assert "1500-2000m" in drought_analysis.sswei_objects
        assert isinstance(drought_analysis.sswei_objects["1500-2000m"], SSWEI)
        assert drought_analysis.sswei_objects["1500-2000m"].sswei_data is not None
        
        # Test with no datasets
        drought_analysis = DroughtAnalysis()
        
        # Check that calculating SSWEI with no datasets raises an error
        with pytest.raises(ValueError):
            drought_analysis.calculate_sswei(
                start_month=12,
                end_month=3,
                min_years=10
            )
    
    def test_compare_elevation_bands(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the compare_elevation_bands method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets for different elevation bands
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        drought_analysis.add_dataset("2000-2500m", dataset, is_elevation_band=True)
        
        # Create mock SSWEI objects
        for band in drought_analysis.elevation_bands:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[band] = sswei_obj
        
        # Compare elevation bands
        comparison_df = drought_analysis.compare_elevation_bands()
        
        # Check that the result is a DataFrame
        assert isinstance(comparison_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['Elevation_Band', 'Total_Years', 'Drought_Count', 'Severe_Drought_Count', 'Drought_Frequency', 'Mean_Severity']
        assert all(col in comparison_df.columns for col in expected_columns)
        
        # Check that the DataFrame has the expected number of rows
        assert len(comparison_df) == len(drought_analysis.elevation_bands)
        
        # Check that the analysis results are updated
        assert 'elevation_band_comparison' in drought_analysis.analysis_results
        assert drought_analysis.analysis_results['elevation_band_comparison'] is comparison_df
        
        # Test with no elevation bands
        drought_analysis = DroughtAnalysis()
        
        # Check that comparing elevation bands with no elevation bands raises an error
        with pytest.raises(ValueError):
            drought_analysis.compare_elevation_bands()
        
        # Test with no SSWEI objects
        drought_analysis = DroughtAnalysis()
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        
        # Check that comparing elevation bands with no SSWEI objects raises an error
        with pytest.raises(ValueError):
            drought_analysis.compare_elevation_bands()
    
    def test_analyze_temporal_trends(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the analyze_temporal_trends method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock SSWEI objects
        for name in drought_analysis.datasets:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[name] = sswei_obj
        
        # Mock the SSWEI.analyze_drought_trends method
        def mock_analyze_drought_trends(self, window_size):
            # Create mock trend data
            window_centers = range(2015, 2020)
            
            trend_data = pd.DataFrame({
                'Start_Year': [center - window_size // 2 for center in window_centers],
                'End_Year': [center + window_size // 2 for center in window_centers],
                'Drought_Frequency': np.random.rand(len(window_centers)) * 100,
                'Mean_Severity': np.random.normal(0, 1, len(window_centers))
            })
            
            return trend_data
        
        # Apply the mock
        monkeypatch.setattr(SSWEI, "analyze_drought_trends", mock_analyze_drought_trends)
        
        # Analyze temporal trends
        trend_results = drought_analysis.analyze_temporal_trends(window_size=10)
        
        # Check that the result is a dictionary
        assert isinstance(trend_results, dict)
        
        # Check that the dictionary has the expected keys
        assert all(name in trend_results for name in drought_analysis.sswei_objects)
        
        # Check that the values are DataFrames
        assert all(isinstance(df, pd.DataFrame) for df in trend_results.values())
        
        # Check that the analysis results are updated
        assert 'temporal_trends' in drought_analysis.analysis_results
        assert drought_analysis.analysis_results['temporal_trends'] is trend_results
        
        # Test with no SSWEI objects
        drought_analysis = DroughtAnalysis()
        
        # Check that analyzing temporal trends with no SSWEI objects raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_temporal_trends()
    
    def test_analyze_drought_characteristics(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the analyze_drought_characteristics method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock SSWEI objects
        for name in drought_analysis.datasets:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[name] = sswei_obj
        
        # Mock the SSWEI.calculate_drought_characteristics method
        def mock_calculate_drought_characteristics(self):
            # Create mock drought characteristics data
            drought_chars = pd.DataFrame({
                'start_year': [2010, 2015],
                'end_year': [2012, 2017],
                'duration': [3, 3],
                'severity': [2.5, 3.0],
                'intensity': [0.83, 1.0]
            })
            
            return drought_chars
        
        # Apply the mock
        monkeypatch.setattr(SSWEI, "calculate_drought_characteristics", mock_calculate_drought_characteristics)
        
        # Analyze drought characteristics
        characteristics_results = drought_analysis.analyze_drought_characteristics()
        
        # Check that the result is a dictionary
        assert isinstance(characteristics_results, dict)
        
        # Check that the dictionary has the expected keys
        assert all(name in characteristics_results for name in drought_analysis.sswei_objects)
        
        # Check that the values are DataFrames
        assert all(isinstance(df, pd.DataFrame) for df in characteristics_results.values())
        
        # Check that the analysis results are updated
        assert 'drought_characteristics' in drought_analysis.analysis_results
        assert drought_analysis.analysis_results['drought_characteristics'] is characteristics_results
        
        # Test with no SSWEI objects
        drought_analysis = DroughtAnalysis()
        
        # Check that analyzing drought characteristics with no SSWEI objects raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_drought_characteristics()
    
    def test_analyze_drought_synchronicity(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the analyze_drought_synchronicity method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock SSWEI objects
        for name in drought_analysis.datasets:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[name] = sswei_obj
        
        # Analyze drought synchronicity
        sync_data = drought_analysis.analyze_drought_synchronicity()
        
        # Check that the result is a DataFrame
        assert isinstance(sync_data, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = list(drought_analysis.sswei_objects.keys()) + ['Datasets_in_Drought', 'Percent_in_Drought']
        assert all(col in sync_data.columns for col in expected_columns)
        
        # Check that the analysis results are updated
        assert 'drought_synchronicity' in drought_analysis.analysis_results
        assert isinstance(drought_analysis.analysis_results['drought_synchronicity'], dict)
        assert 'sync_data' in drought_analysis.analysis_results['drought_synchronicity']
        assert drought_analysis.analysis_results['drought_synchronicity']['sync_data'] is sync_data
        
        # Test with no SSWEI objects
        drought_analysis = DroughtAnalysis()
        
        # Check that analyzing drought synchronicity with no SSWEI objects raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_drought_synchronicity()
        
        # Test with only one SSWEI object
        drought_analysis = DroughtAnalysis()
        drought_analysis.add_dataset("1500-2000m", dataset)
        sswei_obj = SSWEI(dataset)
        sswei_obj.sswei_data = pd.DataFrame({
            'season_year': range(2010, 2020),
            'integrated_swe': np.random.rand(10) * 100,
            'SWEI': np.random.normal(0, 1, 10)
        })
        drought_analysis.sswei_objects["1500-2000m"] = sswei_obj
        
        # Check that analyzing drought synchronicity with only one SSWEI object raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_drought_synchronicity()
    
    def test_analyze_elevation_sensitivity(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the analyze_elevation_sensitivity method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets for different elevation bands
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        drought_analysis.add_dataset("2000-2500m", dataset, is_elevation_band=True)
        
        # Create mock SSWEI objects
        for band in drought_analysis.elevation_bands:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[band] = sswei_obj
        
        # Analyze elevation sensitivity
        elevation_df = drought_analysis.analyze_elevation_sensitivity()
        
        # Check that the result is a DataFrame
        assert isinstance(elevation_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['Elevation_Band', 'Mid_Elevation', 'Drought_Frequency', 'Mean_Severity', 'Severe_Drought_Count', 'Corr_Elevation_Frequency', 'Corr_Elevation_Severity']
        assert all(col in elevation_df.columns for col in expected_columns)
        
        # Check that the DataFrame has the expected number of rows
        assert len(elevation_df) == len(drought_analysis.elevation_bands)
        
        # Check that the analysis results are updated
        assert 'elevation_sensitivity' in drought_analysis.analysis_results
        assert drought_analysis.analysis_results['elevation_sensitivity'] is elevation_df
        
        # Test with no elevation bands
        drought_analysis = DroughtAnalysis()
        
        # Check that analyzing elevation sensitivity with no elevation bands raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_elevation_sensitivity()
        
        # Test with only one elevation band
        drought_analysis = DroughtAnalysis()
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        sswei_obj = SSWEI(dataset)
        sswei_obj.sswei_data = pd.DataFrame({
            'season_year': range(2010, 2020),
            'integrated_swe': np.random.rand(10) * 100,
            'SWEI': np.random.normal(0, 1, 10)
        })
        drought_analysis.sswei_objects["1500-2000m"] = sswei_obj
        
        # Check that analyzing elevation sensitivity with only one elevation band raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_elevation_sensitivity()
        
        # Test with no SSWEI objects
        drought_analysis = DroughtAnalysis()
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        drought_analysis.add_dataset("2000-2500m", dataset, is_elevation_band=True)
        
        # Check that analyzing elevation sensitivity with no SSWEI objects raises an error
        with pytest.raises(ValueError):
            drought_analysis.analyze_elevation_sensitivity()
    
    def test_plot_elevation_band_comparison(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the plot_elevation_band_comparison method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets for different elevation bands
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        drought_analysis.add_dataset("2000-2500m", dataset, is_elevation_band=True)
        
        # Create mock SSWEI objects
        for band in drought_analysis.elevation_bands:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[band] = sswei_obj
        
        # Create mock comparison data
        comparison_df = pd.DataFrame({
            'Elevation_Band': drought_analysis.elevation_bands,
            'Total_Years': [10, 10],
            'Drought_Count': [5, 6],
            'Severe_Drought_Count': [2, 3],
            'Drought_Frequency': [50.0, 60.0],
            'Mean_Severity': [-1.2, -1.5]
        })
        
        # Mock the compare_elevation_bands method
        def mock_compare_elevation_bands(self):
            return comparison_df
        
        # Apply the mock
        monkeypatch.setattr(DroughtAnalysis, "compare_elevation_bands", mock_compare_elevation_bands)
        
        # Set the analysis results
        drought_analysis.analysis_results['elevation_band_comparison'] = comparison_df
        
        # Plot elevation band comparison
        fig = drought_analysis.plot_elevation_band_comparison()
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
        
        # Test without pre-computed comparison data
        drought_analysis.analysis_results = {}
        
        # Mock the compare_elevation_bands method to be called when plotting
        monkeypatch.setattr(DroughtAnalysis, "compare_elevation_bands", mock_compare_elevation_bands)
        
        # Plot elevation band comparison
        fig = drought_analysis.plot_elevation_band_comparison()
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
    
    def test_plot_temporal_trends(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the plot_temporal_trends method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock trend data
        trend_data = {
            "1500-2000m": pd.DataFrame({
                'Start_Year': [2010, 2011, 2012, 2013, 2014],
                'End_Year': [2015, 2016, 2017, 2018, 2019],
                'Drought_Frequency': np.random.rand(5) * 100,
                'Mean_Severity': np.random.normal(0, 1, 5)
            }),
            "2000-2500m": pd.DataFrame({
                'Start_Year': [2010, 2011, 2012, 2013, 2014],
                'End_Year': [2015, 2016, 2017, 2018, 2019],
                'Drought_Frequency': np.random.rand(5) * 100,
                'Mean_Severity': np.random.normal(0, 1, 5)
            })
        }
        
        # Mock the analyze_temporal_trends method
        def mock_analyze_temporal_trends(self, window_size=10):
            return trend_data
        
        # Apply the mock
        monkeypatch.setattr(DroughtAnalysis, "analyze_temporal_trends", mock_analyze_temporal_trends)
        
        # Set the analysis results
        drought_analysis.analysis_results['temporal_trends'] = trend_data
        
        # Plot temporal trends
        fig = drought_analysis.plot_temporal_trends()
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
        
        # Test without pre-computed trend data
        drought_analysis.analysis_results = {}
        
        # Mock the analyze_temporal_trends method to be called when plotting
        monkeypatch.setattr(DroughtAnalysis, "analyze_temporal_trends", mock_analyze_temporal_trends)
        
        # Plot temporal trends
        fig = drought_analysis.plot_temporal_trends()
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
    
    def test_plot_drought_synchronicity(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the plot_drought_synchronicity method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock synchronicity data
        sync_data = pd.DataFrame({
            '1500-2000m': [True, False, True, False, True],
            '2000-2500m': [True, True, False, False, True],
            'Datasets_in_Drought': [2, 1, 1, 0, 2],
            'Percent_in_Drought': [100.0, 50.0, 50.0, 0.0, 100.0]
        }, index=[2010, 2011, 2012, 2013, 2014])
        
        agreement_matrix = pd.DataFrame({
            '1500-2000m': [1.0, 0.6],
            '2000-2500m': [0.6, 1.0]
        }, index=['1500-2000m', '2000-2500m'])
        
        sync_results = {
            'sync_data': sync_data,
            'all_drought_years': [2010, 2014],
            'no_drought_years': [2013],
            'agreement_matrix': agreement_matrix
        }
        
        # Mock the analyze_drought_synchronicity method
        def mock_analyze_drought_synchronicity(self):
            return sync_data
        
        # Apply the mock
        monkeypatch.setattr(DroughtAnalysis, "analyze_drought_synchronicity", mock_analyze_drought_synchronicity)
        
        # Set the analysis results
        drought_analysis.analysis_results['drought_synchronicity'] = sync_results
        
        # Plot drought synchronicity
        heatmap_fig, timeseries_fig = drought_analysis.plot_drought_synchronicity()
        
        # Check that the results are Figures
        assert isinstance(heatmap_fig, plt.Figure)
        assert isinstance(timeseries_fig, plt.Figure)
        
        # Test without pre-computed synchronicity data
        drought_analysis.analysis_results = {}
        
        # Mock the analyze_drought_synchronicity method to be called when plotting
        monkeypatch.setattr(DroughtAnalysis, "analyze_drought_synchronicity", mock_analyze_drought_synchronicity)
        
        # Create a new mock for the result structure
        def mock_analyze_drought_synchronicity_with_structure(self):
            self.analysis_results['drought_synchronicity'] = sync_results
            return sync_data
        
        # Apply the new mock
        monkeypatch.setattr(DroughtAnalysis, "analyze_drought_synchronicity", mock_analyze_drought_synchronicity_with_structure)
        
        # Plot drought synchronicity
        heatmap_fig, timeseries_fig = drought_analysis.plot_drought_synchronicity()
        
        # Check that the results are Figures
        assert isinstance(heatmap_fig, plt.Figure)
        assert isinstance(timeseries_fig, plt.Figure)
    
    def test_plot_elevation_sensitivity(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the plot_elevation_sensitivity method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets for different elevation bands
        drought_analysis.add_dataset("1500-2000m", dataset, is_elevation_band=True)
        drought_analysis.add_dataset("2000-2500m", dataset, is_elevation_band=True)
        
        # Create mock elevation sensitivity data
        elevation_df = pd.DataFrame({
            'Elevation_Band': drought_analysis.elevation_bands,
            'Mid_Elevation': [1750.0, 2250.0],
            'Drought_Frequency': [50.0, 60.0],
            'Mean_Severity': [-1.2, -1.5],
            'Severe_Drought_Count': [2, 3],
            'Corr_Elevation_Frequency': [0.8, 0.8],
            'Corr_Elevation_Severity': [-0.7, -0.7]
        })
        
        # Mock the analyze_elevation_sensitivity method
        def mock_analyze_elevation_sensitivity(self):
            return elevation_df
        
        # Apply the mock
        monkeypatch.setattr(DroughtAnalysis, "analyze_elevation_sensitivity", mock_analyze_elevation_sensitivity)
        
        # Set the analysis results
        drought_analysis.analysis_results['elevation_sensitivity'] = elevation_df
        
        # Plot elevation sensitivity
        fig = drought_analysis.plot_elevation_sensitivity()
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
        
        # Test without pre-computed elevation sensitivity data
        drought_analysis.analysis_results = {}
        
        # Mock the analyze_elevation_sensitivity method to be called when plotting
        monkeypatch.setattr(DroughtAnalysis, "analyze_elevation_sensitivity", mock_analyze_elevation_sensitivity)
        
        # Plot elevation sensitivity
        fig = drought_analysis.plot_elevation_sensitivity()
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
    
    def test_plot_sswei_comparison(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the plot_sswei_comparison method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock SSWEI objects
        for name in drought_analysis.datasets:
            sswei_obj = SSWEI(dataset)
            
            # Create mock SSWEI data
            years = range(2010, 2020)
            integrated_swe = np.random.rand(len(years)) * 100
            sswei_values = np.random.normal(0, 1, len(years))
            
            sswei_obj.sswei_data = pd.DataFrame({
                'season_year': years,
                'integrated_swe': integrated_swe,
                'SWEI': sswei_values,
                'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
            })
            
            drought_analysis.sswei_objects[name] = sswei_obj
        
        # Plot SSWEI comparison for a specific year
        year = 2015
        fig = drought_analysis.plot_sswei_comparison(year)
        
        # Check that the result is a Figure
        assert isinstance(fig, plt.Figure)
        
        # Test with a year that doesn't exist in the data
        with pytest.raises(ValueError):
            drought_analysis.plot_sswei_comparison(1900)
        
        # Test with no SSWEI objects
        drought_analysis = DroughtAnalysis()
        
        # Check that plotting SSWEI comparison with no SSWEI objects raises an error
        with pytest.raises(ValueError):
            drought_analysis.plot_sswei_comparison(2015)
    
    def test_export_results(self, sample_swe_dataframe, sample_stations, tmp_path, monkeypatch):
        """
        Test the export_results method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        tmp_path : pathlib.Path
            Temporary directory path.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object
        drought_analysis = DroughtAnalysis()
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        
        # Create mock analysis results
        # Elevation band comparison
        comparison_df = pd.DataFrame({
            'Elevation_Band': drought_analysis.elevation_bands,
            'Total_Years': [10, 10],
            'Drought_Count': [5, 6],
            'Severe_Drought_Count': [2, 3],
            'Drought_Frequency': [50.0, 60.0],
            'Mean_Severity': [-1.2, -1.5]
        })
        
        # Temporal trends
        trend_data = {
            "1500-2000m": pd.DataFrame({
                'Start_Year': [2010, 2011, 2012, 2013, 2014],
                'End_Year': [2015, 2016, 2017, 2018, 2019],
                'Drought_Frequency': np.random.rand(5) * 100,
                'Mean_Severity': np.random.normal(0, 1, 5)
            }),
            "2000-2500m": pd.DataFrame({
                'Start_Year': [2010, 2011, 2012, 2013, 2014],
                'End_Year': [2015, 2016, 2017, 2018, 2019],
                'Drought_Frequency': np.random.rand(5) * 100,
                'Mean_Severity': np.random.normal(0, 1, 5)
            })
        }
        
        # Drought characteristics
        drought_chars = {
            "1500-2000m": pd.DataFrame({
                'start_year': [2010, 2015],
                'end_year': [2012, 2017],
                'duration': [3, 3],
                'severity': [2.5, 3.0],
                'intensity': [0.83, 1.0]
            }),
            "2000-2500m": pd.DataFrame({
                'start_year': [2011, 2016],
                'end_year': [2013, 2018],
                'duration': [3, 3],
                'severity': [2.2, 2.8],
                'intensity': [0.73, 0.93]
            })
        }
        
        # Drought synchronicity
        sync_data = pd.DataFrame({
            '1500-2000m': [True, False, True, False, True],
            '2000-2500m': [True, True, False, False, True],
            'Datasets_in_Drought': [2, 1, 1, 0, 2],
            'Percent_in_Drought': [100.0, 50.0, 50.0, 0.0, 100.0]
        }, index=[2010, 2011, 2012, 2013, 2014])
        
        agreement_matrix = pd.DataFrame({
            '1500-2000m': [1.0, 0.6],
            '2000-2500m': [0.6, 1.0]
        }, index=['1500-2000m', '2000-2500m'])
        
        sync_results = {
            'sync_data': sync_data,
            'all_drought_years': [2010, 2014],
            'no_drought_years': [2013],
            'agreement_matrix': agreement_matrix
        }
        
        # Elevation sensitivity
        elevation_df = pd.DataFrame({
            'Elevation_Band': drought_analysis.elevation_bands,
            'Mid_Elevation': [1750.0, 2250.0],
            'Drought_Frequency': [50.0, 60.0],
            'Mean_Severity': [-1.2, -1.5],
            'Severe_Drought_Count': [2, 3],
            'Corr_Elevation_Frequency': [0.8, 0.8],
            'Corr_Elevation_Severity': [-0.7, -0.7]
        })
        
        # Set the analysis results
        drought_analysis.analysis_results = {
            'elevation_band_comparison': comparison_df,
            'temporal_trends': trend_data,
            'drought_characteristics': drought_chars,
            'drought_synchronicity': sync_results,
            'elevation_sensitivity': elevation_df
        }
        
        # Export results
        output_dir = tmp_path / "drought_analysis_results"
        drought_analysis.export_results(str(output_dir))
        
        # Check that the output directory exists
        assert output_dir.exists()
        
        # Check that the expected files exist
        assert (output_dir / "elevation_band_comparison.csv").exists()
        assert (output_dir / "temporal_trends_1500-2000m.csv").exists()
        assert (output_dir / "temporal_trends_2000-2500m.csv").exists()
        assert (output_dir / "drought_characteristics_1500-2000m.csv").exists()
        assert (output_dir / "drought_characteristics_2000-2500m.csv").exists()
        assert (output_dir / "drought_synchronicity.csv").exists()
        assert (output_dir / "drought_agreement_matrix.csv").exists()
        assert (output_dir / "elevation_sensitivity.csv").exists()
    
    def test_repr(self, sample_swe_dataframe, sample_stations):
        """
        Test the __repr__ method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Create a DroughtAnalysis object with no datasets
        drought_analysis = DroughtAnalysis()
        repr_str = repr(drought_analysis)
        
        # Check that the representation contains the expected information
        assert "DroughtAnalysis" in repr_str
        assert "datasets=[]" in repr_str
        
        # Add datasets
        drought_analysis.add_dataset("1500-2000m", dataset)
        drought_analysis.add_dataset("2000-2500m", dataset)
        repr_str = repr(drought_analysis)
        
        # Check that the representation contains the expected information
        assert "DroughtAnalysis" in repr_str
        assert "datasets=[1500-2000m, 2000-2500m]" in repr_str
