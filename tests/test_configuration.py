"""
Unit tests for the Configuration class.
"""

import pytest
import os
import json
import yaml
import tempfile
from pathlib import Path

from snowdroughtindex.core.configuration import Configuration

class TestConfiguration:
    """
    Test class for the Configuration class.
    """
    
    def test_init(self):
        """
        Test the initialization of the Configuration class.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Check that the config attribute is a dictionary
        assert isinstance(config.config, dict)
        
        # Check that the default sections are present
        expected_sections = ['gap_filling', 'sswei', 'drought_classification', 'visualization', 'paths']
        for section in expected_sections:
            assert section in config.config
        
        # Check some specific default values
        assert config.get('gap_filling', 'window_days') == 15
        assert config.get('sswei', 'distribution') == 'gamma'
        assert config.get('drought_classification', 'moderate') == -0.5
    
    def test_init_with_config_dict(self):
        """
        Test initialization with a configuration dictionary.
        """
        # Create a configuration dictionary
        config_dict = {
            'gap_filling': {
                'window_days': 20,
                'min_corr': 0.8
            },
            'custom_section': {
                'param1': 'value1',
                'param2': 42
            }
        }
        
        # Initialize with the configuration dictionary
        config = Configuration(config_dict=config_dict)
        
        # Check that the values from the dictionary are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('gap_filling', 'min_corr') == 0.8
        assert config.get('custom_section', 'param1') == 'value1'
        assert config.get('custom_section', 'param2') == 42
        
        # Check that default values for other parameters are preserved
        assert config.get('sswei', 'distribution') == 'gamma'
        assert config.get('drought_classification', 'moderate') == -0.5
    
    def test_get(self):
        """
        Test the get method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get a parameter
        window_days = config.get('gap_filling', 'window_days')
        assert window_days == 15
        
        # Get a section
        gap_filling = config.get('gap_filling')
        assert isinstance(gap_filling, dict)
        assert gap_filling['window_days'] == 15
        
        # Get a parameter with a default value
        custom_param = config.get('custom_section', 'custom_param', default='default_value')
        assert custom_param == 'default_value'
        
        # Get a non-existent section
        non_existent = config.get('non_existent_section')
        assert non_existent is None
        
        # Get a non-existent parameter with a default value
        non_existent_param = config.get('gap_filling', 'non_existent_param', default='default_value')
        assert non_existent_param == 'default_value'
    
    def test_set(self):
        """
        Test the set method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Set a parameter in an existing section
        config.set('gap_filling', 'window_days', 20)
        assert config.get('gap_filling', 'window_days') == 20
        
        # Set a parameter in a new section
        config.set('custom_section', 'custom_param', 'custom_value')
        assert config.get('custom_section', 'custom_param') == 'custom_value'
        
        # Set a new parameter in an existing section
        config.set('gap_filling', 'new_param', 'new_value')
        assert config.get('gap_filling', 'new_param') == 'new_value'
        
        # Check that the method returns the Configuration object
        result = config.set('another_section', 'another_param', 'another_value')
        assert result is config
    
    def test_update(self):
        """
        Test the update method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Create a configuration dictionary
        config_dict = {
            'gap_filling': {
                'window_days': 20,
                'min_corr': 0.8
            },
            'custom_section': {
                'param1': 'value1',
                'param2': 42
            }
        }
        
        # Update the configuration
        config.update(config_dict)
        
        # Check that the values from the dictionary are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('gap_filling', 'min_corr') == 0.8
        assert config.get('custom_section', 'param1') == 'value1'
        assert config.get('custom_section', 'param2') == 42
        
        # Check that default values for other parameters are preserved
        assert config.get('sswei', 'distribution') == 'gamma'
        assert config.get('drought_classification', 'moderate') == -0.5
        
        # Check that the method returns the Configuration object
        result = config.update({'another_section': {'another_param': 'another_value'}})
        assert result is config
    
    def test_load_from_file_json(self, tmp_path):
        """
        Test loading configuration from a JSON file.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory path.
        """
        # Create a configuration dictionary
        config_dict = {
            'gap_filling': {
                'window_days': 20,
                'min_corr': 0.8
            },
            'custom_section': {
                'param1': 'value1',
                'param2': 42
            }
        }
        
        # Create a JSON file
        json_file = tmp_path / "config.json"
        with open(json_file, 'w') as f:
            json.dump(config_dict, f)
        
        # Initialize with default settings
        config = Configuration()
        
        # Load from the JSON file
        config.load_from_file(str(json_file))
        
        # Check that the values from the file are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('gap_filling', 'min_corr') == 0.8
        assert config.get('custom_section', 'param1') == 'value1'
        assert config.get('custom_section', 'param2') == 42
        
        # Check that the method returns the Configuration object
        result = config.load_from_file(str(json_file))
        assert result is config
        
        # Test with a non-existent file
        with pytest.raises(ValueError):
            config.load_from_file("non_existent_file.json")
        
        # Test with an invalid JSON file
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json")
        
        with pytest.raises(ValueError):
            config.load_from_file(str(invalid_json_file))
    
    def test_load_from_file_yaml(self, tmp_path):
        """
        Test loading configuration from a YAML file.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory path.
        """
        # Create a configuration dictionary
        config_dict = {
            'gap_filling': {
                'window_days': 20,
                'min_corr': 0.8
            },
            'custom_section': {
                'param1': 'value1',
                'param2': 42
            }
        }
        
        # Create a YAML file
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Initialize with default settings
        config = Configuration()
        
        # Load from the YAML file
        config.load_from_file(str(yaml_file))
        
        # Check that the values from the file are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('gap_filling', 'min_corr') == 0.8
        assert config.get('custom_section', 'param1') == 'value1'
        assert config.get('custom_section', 'param2') == 42
        
        # Check that the method returns the Configuration object
        result = config.load_from_file(str(yaml_file))
        assert result is config
        
        # Test with an invalid YAML file
        invalid_yaml_file = tmp_path / "invalid.yaml"
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: file:")
        
        with pytest.raises(ValueError):
            config.load_from_file(str(invalid_yaml_file))
        
        # Test with an unsupported file format
        unsupported_file = tmp_path / "config.txt"
        with open(unsupported_file, 'w') as f:
            f.write("unsupported file format")
        
        with pytest.raises(ValueError):
            config.load_from_file(str(unsupported_file))
    
    def test_save_to_file_json(self, tmp_path):
        """
        Test saving configuration to a JSON file.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory path.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Set some values
        config.set('gap_filling', 'window_days', 20)
        config.set('custom_section', 'custom_param', 'custom_value')
        
        # Save to a JSON file
        json_file = tmp_path / "config.json"
        config.save_to_file(str(json_file))
        
        # Check that the file exists
        assert json_file.exists()
        
        # Load the file and check the values
        with open(json_file, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config['gap_filling']['window_days'] == 20
        assert saved_config['custom_section']['custom_param'] == 'custom_value'
        
        # Test with an unsupported file format
        with pytest.raises(ValueError):
            config.save_to_file(str(tmp_path / "config.txt"))
        
        # Test with a directory that doesn't exist
        nested_dir = tmp_path / "nested" / "dir"
        nested_file = nested_dir / "config.json"
        
        # This should create the directory and save the file
        config.save_to_file(str(nested_file))
        assert nested_file.exists()
    
    def test_save_to_file_yaml(self, tmp_path):
        """
        Test saving configuration to a YAML file.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory path.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Set some values
        config.set('gap_filling', 'window_days', 20)
        config.set('custom_section', 'custom_param', 'custom_value')
        
        # Save to a YAML file
        yaml_file = tmp_path / "config.yaml"
        config.save_to_file(str(yaml_file))
        
        # Check that the file exists
        assert yaml_file.exists()
        
        # Load the file and check the values
        with open(yaml_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['gap_filling']['window_days'] == 20
        assert saved_config['custom_section']['custom_param'] == 'custom_value'
        
        # Test with a YML extension
        yml_file = tmp_path / "config.yml"
        config.save_to_file(str(yml_file))
        assert yml_file.exists()
    
    def test_load_from_env_vars(self, monkeypatch):
        """
        Test loading configuration from environment variables.
        
        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Initialize with default settings
        config = Configuration(use_env_vars=False)
        
        # Set environment variables
        monkeypatch.setenv('SNOWDROUGHTINDEX_GAP_FILLING_WINDOW_DAYS', '20')
        monkeypatch.setenv('SNOWDROUGHTINDEX_SSWEI_DISTRIBUTION', 'normal')
        monkeypatch.setenv('SNOWDROUGHTINDEX_CUSTOM_SECTION_CUSTOM_PARAM', 'env_value')
        
        # Load from environment variables
        config.load_from_env_vars()
        
        # Check that the values from the environment variables are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('sswei', 'distribution') == 'normal'
        assert config.get('custom_section', 'custom_param') == 'env_value'
        
        # Test with boolean values
        monkeypatch.setenv('SNOWDROUGHTINDEX_CUSTOM_SECTION_BOOL_TRUE', 'true')
        monkeypatch.setenv('SNOWDROUGHTINDEX_CUSTOM_SECTION_BOOL_FALSE', 'false')
        
        # Load from environment variables
        config.load_from_env_vars()
        
        # Check that the boolean values are correctly converted
        assert config.get('custom_section', 'bool_true') is True
        assert config.get('custom_section', 'bool_false') is False
        
        # Test with numeric values
        monkeypatch.setenv('SNOWDROUGHTINDEX_CUSTOM_SECTION_INT_VALUE', '42')
        monkeypatch.setenv('SNOWDROUGHTINDEX_CUSTOM_SECTION_FLOAT_VALUE', '3.14')
        
        # Load from environment variables
        config.load_from_env_vars()
        
        # Check that the numeric values are correctly converted
        assert config.get('custom_section', 'int_value') == 42
        assert config.get('custom_section', 'float_value') == 3.14
        
        # Check that the method returns the Configuration object
        result = config.load_from_env_vars()
        assert result is config
    
    def test_load_from_cli_args(self, monkeypatch):
        """
        Test loading configuration from command-line arguments.
        
        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        """
        # Mock sys.argv to simulate command-line arguments
        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--gap_filling.window_days=20',
            '--sswei.distribution=normal',
            '--custom_section.custom_param=cli_value'
        ])
        
        # Initialize with command-line argument integration
        config = Configuration(use_cli_args=True)
        
        # Check that the values from the command-line arguments are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('sswei', 'distribution') == 'normal'
        
        # Note: custom_section.custom_param won't be set because it's not in DEFAULT_CONFIG
        # and the argparse setup only adds arguments for parameters in DEFAULT_CONFIG
        
        # Initialize without command-line argument integration
        config = Configuration(use_cli_args=False)
        
        # Check that the values from the command-line arguments are not set
        assert config.get('gap_filling', 'window_days') == 15
        assert config.get('sswei', 'distribution') == 'gamma'
        
        # Check that the method returns the Configuration object
        result = config.load_from_cli_args()
        assert result is config
    
    def test_init_with_config_file(self, tmp_path):
        """
        Test initialization with a configuration file.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory path.
        """
        # Create a configuration dictionary
        config_dict = {
            'gap_filling': {
                'window_days': 20,
                'min_corr': 0.8
            },
            'custom_section': {
                'param1': 'value1',
                'param2': 42
            }
        }
        
        # Create a JSON file
        json_file = tmp_path / "config.json"
        with open(json_file, 'w') as f:
            json.dump(config_dict, f)
        
        # Initialize with the configuration file
        config = Configuration(config_file=str(json_file))
        
        # Check that the values from the file are set
        assert config.get('gap_filling', 'window_days') == 20
        assert config.get('gap_filling', 'min_corr') == 0.8
        assert config.get('custom_section', 'param1') == 'value1'
        assert config.get('custom_section', 'param2') == 42
    
    def test_get_gap_filling_params(self):
        """
        Test the get_gap_filling_params method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get gap filling parameters
        gap_filling_params = config.get_gap_filling_params()
        
        # Check that the result is a dictionary
        assert isinstance(gap_filling_params, dict)
        
        # Check that the dictionary contains the expected parameters
        assert 'window_days' in gap_filling_params
        assert 'min_obs_corr' in gap_filling_params
        assert 'min_obs_cdf' in gap_filling_params
        assert 'min_corr' in gap_filling_params
        assert 'min_obs_KGE' in gap_filling_params
        
        # Check that the values are correct
        assert gap_filling_params['window_days'] == 15
        assert gap_filling_params['min_corr'] == 0.7
    
    def test_get_sswei_params(self):
        """
        Test the get_sswei_params method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get SSWEI parameters
        sswei_params = config.get_sswei_params()
        
        # Check that the result is a dictionary
        assert isinstance(sswei_params, dict)
        
        # Check that the dictionary contains the expected parameters
        assert 'start_month' in sswei_params
        assert 'end_month' in sswei_params
        assert 'min_years' in sswei_params
        assert 'distribution' in sswei_params
        assert 'reference_period' in sswei_params
        
        # Check that the values are correct
        assert sswei_params['start_month'] == 12
        assert sswei_params['end_month'] == 3
        assert sswei_params['distribution'] == 'gamma'
    
    def test_get_drought_classification_thresholds(self):
        """
        Test the get_drought_classification_thresholds method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get drought classification thresholds
        thresholds = config.get_drought_classification_thresholds()
        
        # Check that the result is a dictionary
        assert isinstance(thresholds, dict)
        
        # Check that the dictionary contains the expected thresholds
        assert 'exceptional' in thresholds
        assert 'extreme' in thresholds
        assert 'severe' in thresholds
        assert 'moderate' in thresholds
        
        # Check that the values are correct
        assert thresholds['exceptional'] == -2.0
        assert thresholds['extreme'] == -1.5
        assert thresholds['severe'] == -1.0
        assert thresholds['moderate'] == -0.5
    
    def test_get_visualization_settings(self):
        """
        Test the get_visualization_settings method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get visualization settings
        viz_settings = config.get_visualization_settings()
        
        # Check that the result is a dictionary
        assert isinstance(viz_settings, dict)
        
        # Check that the dictionary contains the expected settings
        assert 'figsize' in viz_settings
        assert 'colors' in viz_settings
        assert 'dpi' in viz_settings
        assert 'fontsize' in viz_settings
        assert 'linewidth' in viz_settings
        assert 'markersize' in viz_settings
        
        # Check that the values are correct
        assert viz_settings['dpi'] == 100
        assert viz_settings['fontsize'] == 12
        assert viz_settings['linewidth'] == 1.5
        assert viz_settings['markersize'] == 6
        
        # Check nested dictionaries
        assert 'small' in viz_settings['figsize']
        assert 'medium' in viz_settings['figsize']
        assert 'large' in viz_settings['figsize']
        
        assert 'exceptional' in viz_settings['colors']
        assert 'extreme' in viz_settings['colors']
        assert 'severe' in viz_settings['colors']
        assert 'moderate' in viz_settings['colors']
        assert 'normal' in viz_settings['colors']
    
    def test_get_paths(self):
        """
        Test the get_paths method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get paths
        paths = config.get_paths()
        
        # Check that the result is a dictionary
        assert isinstance(paths, dict)
        
        # Check that the dictionary contains the expected paths
        assert 'data_dir' in paths
        assert 'output_dir' in paths
        assert 'sample_data' in paths
        
        # Check that the values are correct
        assert paths['data_dir'] == 'data'
        assert paths['output_dir'] == 'output'
        assert paths['sample_data'] == 'data/sample'
    
    def test_repr(self):
        """
        Test the __repr__ method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get the string representation
        repr_str = repr(config)
        
        # Check that the representation contains the expected information
        assert "Configuration" in repr_str
        assert "sections=" in repr_str
        
        # Check that all sections are listed
        for section in config.config.keys():
            assert section in repr_str
    
    def test_str(self):
        """
        Test the __str__ method.
        """
        # Initialize with default settings
        config = Configuration()
        
        # Get the string representation
        str_repr = str(config)
        
        # Check that the representation is a JSON string
        config_dict = json.loads(str_repr)
        
        # Check that the dictionary contains the expected sections
        assert 'gap_filling' in config_dict
        assert 'sswei' in config_dict
        assert 'drought_classification' in config_dict
        assert 'visualization' in config_dict
        assert 'paths' in config_dict
