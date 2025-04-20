"""
Tests for the CLI module of the Snow Drought Index package.

This module contains tests for the command-line interface functionality.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from snowdroughtindex.cli import (
    parse_args, parse_thresholds, fill_gaps_command, calculate_sswei_command,
    classify_drought_command, plot_sswei_command, run_workflow_command, main
)


class TestCLI(unittest.TestCase):
    """
    Test the CLI functionality.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        
        # Create a sample SSWEI data file
        self.sample_sswei_data = pd.DataFrame({
            'season_year': [2000, 2001, 2002, 2003, 2004],
            'integrated_swe': [100, 80, 120, 90, 110],
            'SWEI': [0.5, -1.2, 1.0, -0.8, 0.7]
        })
        
        self.sample_sswei_file = os.path.join(self.temp_path, 'sample_sswei.csv')
        self.sample_sswei_data.to_csv(self.sample_sswei_file, index=False)
    
    def tearDown(self):
        """
        Clean up test fixtures.
        """
        # Close the temporary directory
        self.temp_dir.cleanup()
    
    def test_parse_thresholds(self):
        """
        Test parsing threshold arguments.
        """
        # Test with valid thresholds
        thresholds_list = ['exceptional=-2.0', 'extreme=-1.5', 'severe=-1.0']
        expected = {
            'exceptional': -2.0,
            'extreme': -1.5,
            'severe': -1.0
        }
        result = parse_thresholds(thresholds_list)
        self.assertEqual(result, expected)
        
        # Test with invalid threshold value
        thresholds_list = ['exceptional=invalid', 'extreme=-1.5']
        expected = {'extreme': -1.5}
        result = parse_thresholds(thresholds_list)
        self.assertEqual(result, expected)
        
        # Test with empty list
        thresholds_list = []
        expected = {}
        result = parse_thresholds(thresholds_list)
        self.assertEqual(result, expected)
        
        # Test with None
        thresholds_list = None
        expected = {}
        result = parse_thresholds(thresholds_list)
        self.assertEqual(result, expected)
    
    @patch('snowdroughtindex.cli.SWEDataset')
    @patch('snowdroughtindex.cli.Configuration')
    def test_fill_gaps_command(self, mock_config, mock_dataset):
        """
        Test the fill-gaps command.
        """
        # Create mock objects
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.nc'
        args.output_file = 'output.nc'
        args.window_days = 15
        args.min_obs_corr = 10
        args.min_obs_cdf = 5
        args.min_corr = 0.7
        args.parallel = True
        args.n_jobs = 4
        args.memory_efficient = True
        
        # Call the command
        fill_gaps_command(args)
        
        # Check that the configuration was set correctly
        mock_config_instance.set_performance_params.assert_called_once_with(
            parallel=True,
            n_jobs=4,
            memory_efficient=True
        )
        
        # Check that the dataset was created and used correctly
        mock_dataset.assert_called_once_with(config=mock_config_instance)
        mock_dataset_instance.load_from_file.assert_called_once_with('input.nc')
        mock_dataset_instance.preprocess.assert_called_once()
        mock_dataset_instance.gap_fill.assert_called_once_with(
            window_days=15,
            min_obs_corr=10,
            min_obs_cdf=5,
            min_corr=0.7
        )
        mock_dataset_instance.save.assert_called_once_with('output.nc')
    
    @patch('snowdroughtindex.cli.SWEDataset')
    @patch('snowdroughtindex.cli.SSWEI')
    @patch('snowdroughtindex.cli.Configuration')
    def test_calculate_sswei_command(self, mock_config, mock_sswei, mock_dataset):
        """
        Test the calculate-sswei command.
        """
        # Create mock objects
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_sswei_instance = MagicMock()
        mock_sswei.return_value = mock_sswei_instance
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.nc'
        args.output_file = 'output.csv'
        args.start_month = 12
        args.end_month = 3
        args.min_years = 10
        args.distribution = 'gamma'
        args.reference_period = [1980, 2010]
        args.parallel = True
        args.n_jobs = 4
        args.memory_efficient = True
        
        # Call the command
        calculate_sswei_command(args)
        
        # Check that the configuration was set correctly
        mock_config_instance.set_performance_params.assert_called_once_with(
            parallel=True,
            n_jobs=4,
            memory_efficient=True
        )
        
        # Check that the dataset was created and used correctly
        mock_dataset.assert_called_once_with(config=mock_config_instance)
        mock_dataset_instance.load_from_file.assert_called_once_with('input.nc')
        mock_dataset_instance.preprocess.assert_called_once()
        
        # Check that the SSWEI object was created and used correctly
        mock_sswei.assert_called_once_with(mock_dataset_instance, config=mock_config_instance)
        mock_sswei_instance.calculate_sswei.assert_called_once_with(
            start_month=12,
            end_month=3,
            min_years=10,
            distribution='gamma',
            reference_period=(1980, 2010)
        )
        mock_sswei_instance.save_results.assert_called_once_with('output.csv')
    
    @patch('snowdroughtindex.cli.pd.read_csv')
    @patch('snowdroughtindex.cli.SSWEI')
    def test_classify_drought_command(self, mock_sswei, mock_read_csv):
        """
        Test the classify-drought command.
        """
        # Create mock objects
        mock_sswei_instance = MagicMock()
        mock_sswei.return_value = mock_sswei_instance
        
        mock_sswei_data = MagicMock()
        mock_read_csv.return_value = mock_sswei_data
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.csv'
        args.output_file = 'output.csv'
        args.thresholds = ['exceptional=-2.0', 'extreme=-1.5']
        
        # Call the command
        classify_drought_command(args)
        
        # Check that the data was loaded correctly
        mock_read_csv.assert_called_once_with('input.csv')
        
        # Check that the SSWEI object was created and used correctly
        mock_sswei.assert_called_once()
        self.assertEqual(mock_sswei_instance.sswei_data, mock_sswei_data)
        
        # Check that drought classification was performed with the correct thresholds
        expected_thresholds = {'exceptional': -2.0, 'extreme': -1.5}
        mock_sswei_instance.classify_drought.assert_called_once_with(thresholds=expected_thresholds)
        
        # Check that the results were saved
        mock_sswei_instance.save_results.assert_called_once_with('output.csv')
    
    @patch('snowdroughtindex.cli.pd.read_csv')
    @patch('snowdroughtindex.cli.SSWEI')
    def test_plot_sswei_command(self, mock_sswei, mock_read_csv):
        """
        Test the plot-sswei command.
        """
        # Create mock objects
        mock_sswei_instance = MagicMock()
        mock_sswei.return_value = mock_sswei_instance
        
        mock_sswei_data = MagicMock()
        mock_read_csv.return_value = mock_sswei_data
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.axes = [mock_ax]
        mock_sswei_instance.plot_sswei_timeseries.return_value = mock_fig
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.csv'
        args.output_file = 'output.png'
        args.title = 'Custom Title'
        args.figsize = [12, 8]
        args.dpi = 150
        
        # Call the command
        plot_sswei_command(args)
        
        # Check that the data was loaded correctly
        mock_read_csv.assert_called_once_with('input.csv')
        
        # Check that the SSWEI object was created and used correctly
        mock_sswei.assert_called_once()
        self.assertEqual(mock_sswei_instance.sswei_data, mock_sswei_data)
        
        # Check that the plot was created with the correct parameters
        mock_sswei_instance.plot_sswei_timeseries.assert_called_once_with(figsize=(12, 8))
        
        # Check that the title was set
        mock_ax.set_title.assert_called_once_with('Custom Title')
        
        # Check that the plot was saved with the correct parameters
        mock_fig.savefig.assert_called_once_with('output.png', dpi=150)
    
    @patch('snowdroughtindex.cli.os.makedirs')
    @patch('snowdroughtindex.cli.SWEDataset')
    @patch('snowdroughtindex.cli.SSWEI')
    @patch('snowdroughtindex.cli.DroughtAnalysis')
    @patch('snowdroughtindex.cli.Configuration')
    def test_run_workflow_command_sswei(self, mock_config, mock_drought_analysis, 
                                       mock_sswei, mock_dataset, mock_makedirs):
        """
        Test the run-workflow command with the sswei workflow.
        """
        # Create mock objects
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_sswei_instance = MagicMock()
        mock_sswei.return_value = mock_sswei_instance
        
        # Create mock figures
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        mock_fig3 = MagicMock()
        
        mock_sswei_instance.plot_sswei_timeseries.return_value = mock_fig1
        mock_sswei_instance.plot_drought_classification_heatmap.return_value = mock_fig2
        mock_sswei_instance.plot_drought_severity_distribution.return_value = mock_fig3
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.nc'
        args.output_dir = 'output_dir'
        args.config_file = None
        args.workflow = 'sswei'
        
        # Call the command
        run_workflow_command(args)
        
        # Check that the output directory was created
        mock_makedirs.assert_called_once_with('output_dir', exist_ok=True)
        
        # Check that the dataset was created and used correctly
        mock_dataset.assert_called_once_with(config=mock_config_instance)
        mock_dataset_instance.load_from_file.assert_called_once_with('input.nc')
        mock_dataset_instance.preprocess.assert_called_once()
        mock_dataset_instance.gap_fill.assert_called_once()
        
        # Check that the SSWEI object was created and used correctly
        mock_sswei.assert_called_once_with(mock_dataset_instance, config=mock_config_instance)
        mock_sswei_instance.calculate_sswei.assert_called_once()
        mock_sswei_instance.classify_drought.assert_called_once()
        
        # Check that the plots were created and saved
        mock_sswei_instance.plot_sswei_timeseries.assert_called_once()
        mock_sswei_instance.plot_drought_classification_heatmap.assert_called_once()
        mock_sswei_instance.plot_drought_severity_distribution.assert_called_once()
        
        # Check that the figures were saved
        mock_fig1.savefig.assert_called_once()
        mock_fig2.savefig.assert_called_once()
        mock_fig3.savefig.assert_called_once()
    
    @patch('snowdroughtindex.cli.os.makedirs')
    @patch('snowdroughtindex.cli.SWEDataset')
    @patch('snowdroughtindex.cli.SSWEI')
    @patch('snowdroughtindex.cli.DroughtAnalysis')
    @patch('snowdroughtindex.cli.Configuration')
    def test_run_workflow_command_drought_analysis(self, mock_config, mock_drought_analysis, 
                                                 mock_sswei, mock_dataset, mock_makedirs):
        """
        Test the run-workflow command with the drought-analysis workflow.
        """
        # Create mock objects
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_sswei_instance = MagicMock()
        mock_sswei.return_value = mock_sswei_instance
        
        # Create mock figures
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        mock_fig3 = MagicMock()
        mock_fig4 = MagicMock()
        mock_fig5 = MagicMock()
        
        mock_sswei_instance.plot_sswei_timeseries.return_value = mock_fig1
        mock_sswei_instance.plot_drought_classification_heatmap.return_value = mock_fig2
        mock_sswei_instance.plot_drought_severity_distribution.return_value = mock_fig3
        mock_sswei_instance.plot_drought_characteristics.return_value = mock_fig4
        mock_sswei_instance.plot_drought_trends.return_value = mock_fig5
        
        # Create mock data
        mock_drought_chars = pd.DataFrame({
            'start_year': [2000, 2005],
            'end_year': [2002, 2007],
            'duration': [3, 3],
            'severity': [-1.5, -1.2]
        })
        mock_sswei_instance.calculate_drought_characteristics.return_value = mock_drought_chars
        
        mock_trend_data = pd.DataFrame({
            'window_end': [2010, 2015, 2020],
            'drought_frequency': [0.3, 0.4, 0.5],
            'mean_severity': [-1.2, -1.3, -1.4]
        })
        mock_sswei_instance.analyze_drought_trends.return_value = mock_trend_data
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.nc'
        args.output_dir = 'output_dir'
        args.config_file = None
        args.workflow = 'drought-analysis'
        
        # Call the command
        run_workflow_command(args)
        
        # Check that the output directory was created
        mock_makedirs.assert_called_once_with('output_dir', exist_ok=True)
        
        # Check that the dataset was created and used correctly
        mock_dataset.assert_called_once_with(config=mock_config_instance)
        mock_dataset_instance.load_from_file.assert_called_once_with('input.nc')
        mock_dataset_instance.preprocess.assert_called_once()
        mock_dataset_instance.gap_fill.assert_called_once()
        
        # Check that the SSWEI object was created and used correctly
        mock_sswei.assert_called_once_with(mock_dataset_instance, config=mock_config_instance)
        mock_sswei_instance.calculate_sswei.assert_called_once()
        mock_sswei_instance.classify_drought.assert_called_once()
        
        # Check that the drought analysis was performed
        mock_sswei_instance.calculate_drought_characteristics.assert_called_once()
        mock_sswei_instance.analyze_drought_trends.assert_called_once()
        
        # Check that the plots were created and saved
        mock_sswei_instance.plot_sswei_timeseries.assert_called_once()
        mock_sswei_instance.plot_drought_classification_heatmap.assert_called_once()
        mock_sswei_instance.plot_drought_severity_distribution.assert_called_once()
        mock_sswei_instance.plot_drought_characteristics.assert_called_once()
        mock_sswei_instance.plot_drought_trends.assert_called_once()
        
        # Check that the figures were saved
        mock_fig1.savefig.assert_called_once()
        mock_fig2.savefig.assert_called_once()
        mock_fig3.savefig.assert_called_once()
        mock_fig4.savefig.assert_called_once()
        mock_fig5.savefig.assert_called_once()
    
    @patch('snowdroughtindex.cli.os.makedirs')
    @patch('snowdroughtindex.cli.SWEDataset')
    @patch('snowdroughtindex.cli.SSWEI')
    @patch('snowdroughtindex.cli.DroughtAnalysis')
    @patch('snowdroughtindex.cli.Configuration')
    def test_run_workflow_command_elevation_analysis(self, mock_config, mock_drought_analysis, 
                                                   mock_sswei, mock_dataset, mock_makedirs):
        """
        Test the run-workflow command with the elevation-analysis workflow.
        """
        # Create mock objects
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_sswei_instance = MagicMock()
        mock_sswei.return_value = mock_sswei_instance
        
        mock_drought_analysis_instance = MagicMock()
        mock_drought_analysis.return_value = mock_drought_analysis_instance
        
        # Create mock figures
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        mock_fig3 = MagicMock()
        mock_fig4 = MagicMock()
        mock_fig5 = MagicMock()
        mock_fig6 = MagicMock()
        
        mock_sswei_instance.plot_sswei_timeseries.return_value = mock_fig1
        mock_sswei_instance.plot_drought_classification_heatmap.return_value = mock_fig2
        mock_sswei_instance.plot_drought_severity_distribution.return_value = mock_fig3
        mock_drought_analysis_instance.plot_elevation_band_comparison.return_value = mock_fig4
        mock_drought_analysis_instance.plot_drought_synchronicity.return_value = (mock_fig5, mock_fig6)
        
        # Create mock data
        mock_comparison_df = pd.DataFrame({
            'Elevation_Band': ['1000-1500m', '1500-2000m', '2000-2500m'],
            'Drought_Frequency': [0.3, 0.4, 0.5],
            'Mean_Severity': [-1.2, -1.3, -1.4]
        })
        mock_drought_analysis_instance.compare_elevation_bands.return_value = mock_comparison_df
        
        mock_sync_data = pd.DataFrame({
            'Year': [2000, 2001, 2002],
            'Datasets_in_Drought': [1, 2, 0],
            'Percent_in_Drought': [33.3, 66.7, 0.0]
        })
        mock_drought_analysis_instance.analyze_drought_synchronicity.return_value = mock_sync_data
        
        # Create mock arguments
        args = MagicMock()
        args.input_file = 'input.nc'
        args.output_dir = 'output_dir'
        args.config_file = None
        args.workflow = 'elevation-analysis'
        
        # Call the command
        run_workflow_command(args)
        
        # Check that the output directory was created
        mock_makedirs.assert_called_once_with('output_dir', exist_ok=True)
        
        # Check that the dataset was created and used correctly
        mock_dataset.assert_called_once_with(config=mock_config_instance)
        mock_dataset_instance.load_from_file.assert_called_once_with('input.nc')
        mock_dataset_instance.preprocess.assert_called_once()
        mock_dataset_instance.gap_fill.assert_called_once()
        
        # Check that the SSWEI object was created and used correctly
        mock_sswei.assert_called_once_with(mock_dataset_instance, config=mock_config_instance)
        mock_sswei_instance.calculate_sswei.assert_called_once()
        mock_sswei_instance.classify_drought.assert_called_once()
        
        # Check that the DroughtAnalysis object was created and used correctly
        mock_drought_analysis.assert_called_once()
        mock_drought_analysis_instance.add_dataset.assert_called_once_with("All Stations", mock_dataset_instance)
        mock_drought_analysis_instance.calculate_sswei.assert_called_once()
        mock_drought_analysis_instance.compare_elevation_bands.assert_called_once()
        mock_drought_analysis_instance.analyze_drought_synchronicity.assert_called_once()
        
        # Check that the plots were created and saved
        mock_sswei_instance.plot_sswei_timeseries.assert_called_once()
        mock_sswei_instance.plot_drought_classification_heatmap.assert_called_once()
        mock_sswei_instance.plot_drought_severity_distribution.assert_called_once()
        mock_drought_analysis_instance.plot_elevation_band_comparison.assert_called_once()
        mock_drought_analysis_instance.plot_drought_synchronicity.assert_called_once()
        
        # Check that the figures were saved
        mock_fig1.savefig.assert_called_once()
        mock_fig2.savefig.assert_called_once()
        mock_fig3.savefig.assert_called_once()
        mock_fig4.savefig.assert_called_once()
        mock_fig5.savefig.assert_called_once()
        mock_fig6.savefig.assert_called_once()
    
    @patch('snowdroughtindex.cli.parse_args')
    @patch('snowdroughtindex.cli.fill_gaps_command')
    @patch('snowdroughtindex.cli.calculate_sswei_command')
    @patch('snowdroughtindex.cli.classify_drought_command')
    @patch('snowdroughtindex.cli.plot_sswei_command')
    @patch('snowdroughtindex.cli.run_workflow_command')
    def test_main(self, mock_run_workflow, mock_plot_sswei, mock_classify_drought,
                 mock_calculate_sswei, mock_fill_gaps, mock_parse_args):
        """
        Test the main function.
        """
        # Test with fill-gaps command
        mock_args = MagicMock()
        mock_args.command = 'fill-gaps'
        mock_parse_args.return_value = mock_args
        
        main()
        
        mock_fill_gaps.assert_called_once_with(mock_args)
        mock_calculate_sswei.assert_not_called()
        mock_classify_drought.assert_not_called()
        mock_plot_sswei.assert_not_called()
        mock_run_workflow.assert_not_called()
        
        # Reset mocks
        mock_fill_gaps.reset_mock()
        
        # Test with calculate-sswei command
        mock_args.command = 'calculate-sswei'
        main()
        
        mock_fill_gaps.assert_not_called()
        mock_calculate_sswei.assert_called_once_with(mock_args)
        mock_classify_drought.assert_not_called()
        mock_plot_sswei.assert_not_called()
        mock_run_workflow.assert_not_called()
        
        # Reset mocks
        mock_calculate_sswei.reset_mock()
        
        # Test with classify-drought command
        mock_args.command = 'classify-drought'
        main()
        
        mock_fill_gaps.assert_not_called()
        mock_calculate_sswei.assert_not_called()
        mock_classify_drought.assert_called_once_with(mock_args)
        mock_plot_sswei.assert_not_called()
        mock_run_workflow.assert_not_called()
        
        # Reset mocks
        mock_classify_drought.reset_mock()
        
        # Test with plot-sswei command
        mock_args.command = 'plot-sswei'
        main()
        
        mock_fill_gaps.assert_not_called()
        mock_calculate_sswei.assert_not_called()
        mock_classify_drought.assert_not_called()
        mock_plot_sswei.assert_called_once_with(mock_args)
        mock_run_workflow.assert_not_called()
        
        # Reset mocks
        mock_plot_sswei.reset_mock()
        
        # Test with run-workflow command
        mock_args.command = 'run-workflow'
        main()
        
        mock_fill_gaps.assert_not_called()
        mock_calculate_sswei.assert_not_called()
        mock_classify_drought.assert_not_called()
        mock_plot_sswei.assert_not_called()
        mock_run_workflow.assert_called_once_with(mock_args)
        
        # Reset mocks
        mock_run_workflow.reset_mock()
        
        # Test with no command
        mock_args.command = None
        
        # Patch sys.exit to avoid exiting the test
        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)
        
        mock_fill_gaps.assert_not_called()
        mock_calculate_sswei.assert_not_called()
        mock_classify_drought.assert_not_called()
        mock_plot_sswei.assert_not_called()
        mock_run_workflow.assert_not_called()


if __name__ == '__main__':
    unittest.main()
