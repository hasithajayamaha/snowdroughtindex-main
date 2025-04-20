"""
Unit tests for the SSWEI class.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from datetime import datetime

from snowdroughtindex.core.sswei_class import SSWEI
from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core import drought_classification

class TestSSWEIClass:
    """
    Test class for the SSWEI class.
    """
    
    def test_init(self, sample_swe_dataframe, sample_stations):
        """
        Test the initialization of the SSWEI class.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        """
        # Create a SWEDataset
        dataset = SWEDataset(sample_swe_dataframe, sample_stations)
        
        # Initialize with SWEDataset
        sswei_obj = SSWEI(dataset)
        
        # Check that the data is set correctly
        assert sswei_obj.data is dataset.data
        assert sswei_obj.sswei_data is None
        assert sswei_obj.drought_classifications is None
        assert sswei_obj.thresholds == drought_classification.DEFAULT_THRESHOLDS
        
        # Initialize with DataFrame
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that the data is set correctly
        assert sswei_obj.data is sample_swe_dataframe
        assert sswei_obj.sswei_data is None
        assert sswei_obj.drought_classifications is None
        
        # Initialize with None
        sswei_obj = SSWEI()
        
        # Check that the data is None
        assert sswei_obj.data is None
        assert sswei_obj.sswei_data is None
        assert sswei_obj.drought_classifications is None
    
    def test_load_data(self, sample_swe_dataframe, sample_stations, monkeypatch):
        """
        Test the load_data method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        sample_stations : pandas.DataFrame
            Sample stations DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Mock the SWEDataset.load_from_file method
        def mock_load_from_file(self, file_path):
            self.data = sample_swe_dataframe
            self.stations = sample_stations
            return self
        
        # Apply the mock
        monkeypatch.setattr(SWEDataset, "load_from_file", mock_load_from_file)
        
        # Create an SSWEI object
        sswei_obj = SSWEI()
        
        # Load data
        sswei_obj.load_data("dummy_path.nc")
        
        # Check that the data is set correctly
        assert sswei_obj.data is sample_swe_dataframe
    
    def test_set_thresholds(self, sample_swe_dataframe):
        """
        Test the set_thresholds method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Define custom thresholds
        custom_thresholds = {
            "exceptional": -2.0,
            "extreme": -1.5,
            "severe": -1.0,
            "moderate": -0.5
        }
        
        # Set custom thresholds
        sswei_obj.set_thresholds(custom_thresholds)
        
        # Check that the thresholds are updated correctly
        for key, value in custom_thresholds.items():
            assert sswei_obj.thresholds[key] == value
        
        # Check that other thresholds are not affected
        for key in drought_classification.DEFAULT_THRESHOLDS:
            if key not in custom_thresholds:
                assert sswei_obj.thresholds[key] == drought_classification.DEFAULT_THRESHOLDS[key]
    
    def test_calculate_sswei(self, sample_swe_dataframe, monkeypatch):
        """
        Test the calculate_sswei method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the SSWEI calculation
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_seasonal_data = {
            'data': sample_swe_dataframe,
            'years': list(years),
            'stations': list(sample_swe_dataframe.columns)
        }
        
        mock_integrated_swe = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe
        })
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values
        })
        
        # Mock the sswei module functions
        def mock_prepare_seasonal_data(data, start_month, end_month, min_years):
            return mock_seasonal_data
        
        def mock_integrate_season(seasonal_data):
            return mock_integrated_swe
        
        def mock_calculate_sswei(integrated_swe, distribution, reference_period):
            return mock_sswei_data
        
        # Apply the mocks
        monkeypatch.setattr("snowdroughtindex.core.sswei.prepare_season_data", mock_prepare_seasonal_data)
        monkeypatch.setattr("snowdroughtindex.core.sswei.integrate_season", mock_integrate_season)
        monkeypatch.setattr("snowdroughtindex.core.sswei.calculate_sswei", mock_calculate_sswei)
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Calculate SSWEI
        result = sswei_obj.calculate_sswei(
            start_month=12,
            end_month=3,
            min_years=10,
            distribution='gamma'
        )
        
        # Check that the result is the SSWEI object
        assert result is sswei_obj
        
        # Check that the sswei_data is set correctly
        assert sswei_obj.sswei_data is mock_sswei_data
        
        # Test with missing data
        sswei_obj = SSWEI()
        
        # Check that calculating SSWEI with missing data raises an error
        with pytest.raises(ValueError):
            sswei_obj.calculate_sswei(
                start_month=12,
                end_month=3,
                min_years=10
            )
    
    def test_classify_drought(self, sample_swe_dataframe, monkeypatch):
        """
        Test the classify_drought method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the drought classification
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values
        })
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Classify drought
        result = sswei_obj.classify_drought()
        
        # Check that the result is the SSWEI object
        assert result is sswei_obj
        
        # Check that the Drought_Classification column is added to sswei_data
        assert 'Drought_Classification' in sswei_obj.sswei_data.columns
        
        # Test with custom thresholds
        custom_thresholds = {
            "exceptional": -2.0,
            "extreme": -1.5,
            "severe": -1.0,
            "moderate": -0.5
        }
        
        # Reset the sswei_data
        sswei_obj.sswei_data = mock_sswei_data.copy()
        
        # Classify drought with custom thresholds
        result = sswei_obj.classify_drought(custom_thresholds)
        
        # Check that the result is the SSWEI object
        assert result is sswei_obj
        
        # Check that the Drought_Classification column is added to sswei_data
        assert 'Drought_Classification' in sswei_obj.sswei_data.columns
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that classifying drought with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.classify_drought()
    
    def test_calculate_drought_characteristics(self, sample_swe_dataframe, monkeypatch):
        """
        Test the calculate_drought_characteristics method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the drought characteristics
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        mock_drought_chars = pd.DataFrame({
            'start_year': [2010, 2015],
            'end_year': [2012, 2017],
            'duration': [3, 3],
            'severity': [2.5, 3.0],
            'intensity': [0.83, 1.0]
        })
        
        # Mock the drought_classification.calculate_drought_characteristics function
        def mock_calculate_drought_characteristics(data, year_column, swei_column):
            return mock_drought_chars
        
        # Apply the mock
        monkeypatch.setattr(
            "snowdroughtindex.core.drought_classification.calculate_drought_characteristics",
            mock_calculate_drought_characteristics
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Calculate drought characteristics
        drought_chars = sswei_obj.calculate_drought_characteristics()
        
        # Check that the result is the mock drought characteristics
        assert drought_chars is mock_drought_chars
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that calculating drought characteristics with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.calculate_drought_characteristics()
    
    def test_analyze_drought_trends(self, sample_swe_dataframe, monkeypatch):
        """
        Test the analyze_drought_trends method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the drought trends
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        mock_trend_data = pd.DataFrame({
            'window_center': range(2015, 2020),
            'drought_frequency': np.random.rand(5),
            'mean_severity': np.random.rand(5),
            'mean_duration': np.random.rand(5) * 3
        })
        
        # Mock the drought_classification.analyze_drought_trends function
        def mock_analyze_drought_trends(data, year_column, swei_column, window_size):
            return mock_trend_data
        
        # Apply the mock
        monkeypatch.setattr(
            "snowdroughtindex.core.drought_classification.analyze_drought_trends",
            mock_analyze_drought_trends
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Analyze drought trends
        trend_data = sswei_obj.analyze_drought_trends(window_size=10)
        
        # Check that the result is the mock trend data
        assert trend_data is mock_trend_data
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that analyzing drought trends with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.analyze_drought_trends()
    
    def test_plot_sswei_timeseries(self, sample_swe_dataframe, monkeypatch):
        """
        Test the plot_sswei_timeseries method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the SSWEI time series
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        # Create a mock figure
        mock_fig = plt.figure()
        
        # Mock the visualization.plot_sswei_timeseries function
        def mock_plot_sswei_timeseries(data, year_column, swei_column, classification_column, figsize):
            return mock_fig
        
        # Apply the mock
        monkeypatch.setattr(
            "snowdroughtindex.utils.visualization.plot_sswei_timeseries",
            mock_plot_sswei_timeseries
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Plot SSWEI time series
        fig = sswei_obj.plot_sswei_timeseries()
        
        # Check that the result is the mock figure
        assert fig is mock_fig
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that plotting SSWEI time series with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.plot_sswei_timeseries()
    
    def test_plot_drought_characteristics(self, sample_swe_dataframe, monkeypatch):
        """
        Test the plot_drought_characteristics method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the drought characteristics
        mock_drought_chars = pd.DataFrame({
            'start_year': [2010, 2015],
            'end_year': [2012, 2017],
            'duration': [3, 3],
            'severity': [2.5, 3.0],
            'intensity': [0.83, 1.0]
        })
        
        # Create a mock figure
        mock_fig = plt.figure()
        
        # Mock the calculate_drought_characteristics method
        def mock_calculate_drought_characteristics(self):
            return mock_drought_chars
        
        # Mock the drought_classification.plot_drought_characteristics function
        def mock_plot_drought_characteristics(drought_chars, figsize):
            return mock_fig
        
        # Apply the mocks
        monkeypatch.setattr(SSWEI, "calculate_drought_characteristics", mock_calculate_drought_characteristics)
        monkeypatch.setattr(
            "snowdroughtindex.core.drought_classification.plot_drought_characteristics",
            mock_plot_drought_characteristics
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Plot drought characteristics
        fig = sswei_obj.plot_drought_characteristics()
        
        # Check that the result is the mock figure
        assert fig is mock_fig
    
    def test_plot_drought_trends(self, sample_swe_dataframe, monkeypatch):
        """
        Test the plot_drought_trends method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the drought trends
        mock_trend_data = pd.DataFrame({
            'window_center': range(2015, 2020),
            'drought_frequency': np.random.rand(5),
            'mean_severity': np.random.rand(5),
            'mean_duration': np.random.rand(5) * 3
        })
        
        # Create a mock figure
        mock_fig = plt.figure()
        
        # Mock the analyze_drought_trends method
        def mock_analyze_drought_trends(self, window_size):
            return mock_trend_data
        
        # Mock the drought_classification.plot_drought_trends function
        def mock_plot_drought_trends(trend_data, figsize):
            return mock_fig
        
        # Apply the mocks
        monkeypatch.setattr(SSWEI, "analyze_drought_trends", mock_analyze_drought_trends)
        monkeypatch.setattr(
            "snowdroughtindex.core.drought_classification.plot_drought_trends",
            mock_plot_drought_trends
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Plot drought trends
        fig = sswei_obj.plot_drought_trends()
        
        # Check that the result is the mock figure
        assert fig is mock_fig
    
    def test_plot_drought_classification_heatmap(self, sample_swe_dataframe, monkeypatch):
        """
        Test the plot_drought_classification_heatmap method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the SSWEI
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        # Create a mock figure
        mock_fig = plt.figure()
        
        # Mock the visualization.plot_drought_classification_heatmap function
        def mock_plot_drought_classification_heatmap(data, figsize):
            return mock_fig
        
        # Apply the mock
        monkeypatch.setattr(
            "snowdroughtindex.utils.visualization.plot_drought_classification_heatmap",
            mock_plot_drought_classification_heatmap
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Plot drought classification heatmap
        fig = sswei_obj.plot_drought_classification_heatmap()
        
        # Check that the result is the mock figure
        assert fig is mock_fig
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that plotting drought classification heatmap with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.plot_drought_classification_heatmap()
    
    def test_plot_drought_severity_distribution(self, sample_swe_dataframe, monkeypatch):
        """
        Test the plot_drought_severity_distribution method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the SSWEI
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        # Create a mock figure
        mock_fig = plt.figure()
        
        # Mock the visualization.plot_drought_severity_distribution function
        def mock_plot_drought_severity_distribution(data, figsize):
            return mock_fig
        
        # Apply the mock
        monkeypatch.setattr(
            "snowdroughtindex.utils.visualization.plot_drought_severity_distribution",
            mock_plot_drought_severity_distribution
        )
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Plot drought severity distribution
        fig = sswei_obj.plot_drought_severity_distribution()
        
        # Check that the result is the mock figure
        assert fig is mock_fig
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that plotting drought severity distribution with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.plot_drought_severity_distribution()
    
    def test_save_results(self, sample_swe_dataframe, tmp_path, monkeypatch):
        """
        Test the save_results method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        tmp_path : pathlib.Path
            Temporary directory path.
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Create mock data for the SSWEI
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        # Create an SSWEI object
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        
        # Save results
        file_path = tmp_path / "sswei_results.csv"
        sswei_obj.save_results(str(file_path))
        
        # Check that the file exists
        assert file_path.exists()
        
        # Test with missing sswei_data
        sswei_obj = SSWEI(sample_swe_dataframe)
        
        # Check that saving results with missing sswei_data raises an error
        with pytest.raises(ValueError):
            sswei_obj.save_results(str(file_path))
    
    def test_repr(self, sample_swe_dataframe):
        """
        Test the __repr__ method.
        
        Parameters
        ----------
        sample_swe_dataframe : pandas.DataFrame
            Sample SWE DataFrame.
        """
        # Create an SSWEI object with data but no SSWEI calculation
        sswei_obj = SSWEI(sample_swe_dataframe)
        repr_str = repr(sswei_obj)
        
        # Check that the representation contains the expected information
        assert "SSWEI" in repr_str
        assert "data=<loaded>" in repr_str
        assert "sswei=<not calculated>" in repr_str
        
        # Create mock data for the SSWEI
        years = range(2010, 2020)
        integrated_swe = np.random.rand(len(years)) * 100
        sswei_values = np.random.normal(0, 1, len(years))
        
        mock_sswei_data = pd.DataFrame({
            'season_year': years,
            'integrated_swe': integrated_swe,
            'SWEI': sswei_values,
            'Drought_Classification': np.random.choice(['Normal', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], len(years))
        })
        
        # Create an SSWEI object with data and SSWEI calculation
        sswei_obj = SSWEI(sample_swe_dataframe)
        sswei_obj.sswei_data = mock_sswei_data
        repr_str = repr(sswei_obj)
        
        # Check that the representation contains the expected information
        assert "SSWEI" in repr_str
        assert "data=<loaded>" in repr_str
        assert "sswei=<calculated for" in repr_str
        assert str(len(mock_sswei_data)) in repr_str
