"""
Configuration class for the Snow Drought Index package.

This module contains the Configuration class, which provides a centralized way to manage
parameters for gap filling, SSWEI calculation, and visualization settings.
"""

import os
import json
import yaml
import argparse
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy

class Configuration:
    """
    A class for managing configuration parameters for the Snow Drought Index package.
    
    This class provides a centralized way to manage parameters for gap filling,
    SSWEI calculation, and visualization settings. It supports loading configuration
    from files (YAML/JSON), command-line parameter overrides, and environment variable
    integration.
    
    Attributes
    ----------
    config : dict
        Dictionary containing all configuration parameters.
    """
    
    # Default configuration parameters
    DEFAULT_CONFIG = {
        # Gap filling parameters
        "gap_filling": {
            "window_days": 15,
            "min_obs_corr": 10,
            "min_obs_cdf": 5,
            "min_corr": 0.7,
            "min_obs_KGE": 5
        },
        
        # SSWEI calculation parameters
        "sswei": {
            "start_month": 12,  # December
            "end_month": 3,     # March
            "min_years": 10,
            "distribution": "gamma",
            "reference_period": None  # Use entire period
        },
        
        # Drought classification thresholds
        "drought_classification": {
            "exceptional": -2.0,
            "extreme": -1.5,
            "severe": -1.0,
            "moderate": -0.5
        },
        
        # Visualization settings
        "visualization": {
            "figsize": {
                "small": [8, 6],
                "medium": [12, 8],
                "large": [16, 10]
            },
            "colors": {
                "exceptional": "#800000",  # Dark red
                "extreme": "#FF0000",      # Red
                "severe": "#FF6600",       # Orange-red
                "moderate": "#FFA500",     # Orange
                "normal": "#008000"        # Green
            },
            "dpi": 100,
            "fontsize": 12,
            "linewidth": 1.5,
            "markersize": 6
        },
        
        # Performance optimization settings
        "performance": {
            "parallel": False,
            "n_jobs": -1,  # Use all available cores
            "lazy_loading": False,
            "chunks": {
                "time": 100,
                "lat": 50,
                "lon": 50
            },
            "memory_efficient": False,
            "enable_caching": False,
            "cache_dir": "./.cache",
            "dask_client": None
        },
        
        # Data paths
        "paths": {
            "data_dir": "data",
            "output_dir": "output",
            "sample_data": "data/sample"
        }
    }
    
    def __init__(self, config_file: Optional[str] = None, 
                 config_dict: Optional[Dict[str, Any]] = None,
                 use_env_vars: bool = True,
                 use_cli_args: bool = False):
        """
        Initialize a Configuration object.
        
        Parameters
        ----------
        config_file : str, optional
            Path to a configuration file (YAML or JSON), by default None.
        config_dict : dict, optional
            Dictionary containing configuration parameters, by default None.
        use_env_vars : bool, optional
            Whether to use environment variables for configuration, by default True.
        use_cli_args : bool, optional
            Whether to use command-line arguments for configuration, by default False.
        """
        # Start with default configuration
        self.config = deepcopy(self.DEFAULT_CONFIG)
        
        # Update with configuration from file if provided
        if config_file is not None:
            self.load_from_file(config_file)
        
        # Update with configuration from dictionary if provided
        if config_dict is not None:
            self.update(config_dict)
        
        # Update with environment variables if enabled
        if use_env_vars:
            self.load_from_env_vars()
        
        # Update with command-line arguments if enabled
        if use_cli_args:
            self.load_from_cli_args()
    
    def load_from_file(self, config_file: str) -> 'Configuration':
        """
        Load configuration from a file.
        
        Parameters
        ----------
        config_file : str
            Path to a configuration file (YAML or JSON).
            
        Returns
        -------
        Configuration
            The Configuration object with updated parameters.
            
        Raises
        ------
        ValueError
            If the file format is not supported or the file cannot be read.
        """
        if not os.path.exists(config_file):
            raise ValueError(f"Configuration file not found: {config_file}")
        
        file_ext = os.path.splitext(config_file)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            self.update(config_dict)
            return self
        
        except Exception as e:
            raise ValueError(f"Error loading configuration from file: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """
        Save configuration to a file.
        
        Parameters
        ----------
        config_file : str
            Path to save the configuration file (YAML or JSON).
            
        Raises
        ------
        ValueError
            If the file format is not supported or the file cannot be written.
        """
        file_ext = os.path.splitext(config_file)[1].lower()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            if file_ext == '.json':
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        except Exception as e:
            raise ValueError(f"Error saving configuration to file: {e}")
    
    def load_from_env_vars(self) -> 'Configuration':
        """
        Load configuration from environment variables.
        
        Environment variables should be in the format:
        SNOWDROUGHTINDEX_SECTION_PARAMETER=value
        
        For example:
        SNOWDROUGHTINDEX_GAP_FILLING_WINDOW_DAYS=20
        
        Returns
        -------
        Configuration
            The Configuration object with updated parameters.
        """
        prefix = "SNOWDROUGHTINDEX_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Remove prefix and split by underscore
                parts = env_var[len(prefix):].lower().split('_')
                
                # Need at least section and parameter
                if len(parts) < 2:
                    continue
                
                # First part is section, last part is parameter
                section = parts[0]
                param = '_'.join(parts[1:])
                
                # Convert value to appropriate type
                try:
                    # Try to convert to int
                    value = int(value)
                except ValueError:
                    try:
                        # Try to convert to float
                        value = float(value)
                    except ValueError:
                        # Check for boolean values
                        if value.lower() in ['true', 'yes', '1']:
                            value = True
                        elif value.lower() in ['false', 'no', '0']:
                            value = False
                        # Otherwise, keep as string
                
                # Update configuration
                if section in self.config and param in self.config[section]:
                    self.config[section][param] = value
                elif section in self.config:
                    # Create new parameter in existing section
                    self.config[section][param] = value
                else:
                    # Create new section and parameter
                    self.config[section] = {param: value}
        
        return self
    
    def load_from_cli_args(self) -> 'Configuration':
        """
        Load configuration from command-line arguments.
        
        Command-line arguments should be in the format:
        --section.parameter=value
        
        For example:
        --gap_filling.window_days=20
        
        Returns
        -------
        Configuration
            The Configuration object with updated parameters.
        """
        parser = argparse.ArgumentParser(description='Snow Drought Index Configuration')
        
        # Add arguments for each configuration parameter
        for section, params in self.DEFAULT_CONFIG.items():
            if isinstance(params, dict):
                for param, value in params.items():
                    arg_name = f"--{section}.{param}"
                    parser.add_argument(arg_name, type=type(value) if value is not None else str)
        
        # Parse known arguments (ignore unknown ones)
        args, _ = parser.parse_known_args()
        
        # Convert namespace to dictionary
        args_dict = vars(args)
        
        # Update configuration
        for arg_name, value in args_dict.items():
            if value is not None:
                section, param = arg_name.split('.')
                if section in self.config:
                    self.config[section][param] = value
        
        return self
    
    def update(self, config_dict: Dict[str, Any]) -> 'Configuration':
        """
        Update configuration with values from a dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration parameters.
            
        Returns
        -------
        Configuration
            The Configuration object with updated parameters.
        """
        for section, params in config_dict.items():
            if isinstance(params, dict):
                # Create section if it doesn't exist
                if section not in self.config:
                    self.config[section] = {}
                
                # Update parameters in section
                for param, value in params.items():
                    self.config[section][param] = value
            else:
                # Set value directly
                self.config[section] = params
        
        return self
    
    def get(self, section: str, parameter: Optional[str] = None, 
           default: Any = None) -> Any:
        """
        Get a configuration parameter.
        
        Parameters
        ----------
        section : str
            Configuration section.
        parameter : str, optional
            Configuration parameter within the section, by default None.
            If None, returns the entire section.
        default : Any, optional
            Default value to return if the parameter is not found, by default None.
            
        Returns
        -------
        Any
            The configuration parameter value, or the default value if not found.
        """
        if section not in self.config:
            return default
        
        if parameter is None:
            return self.config[section]
        
        if parameter not in self.config[section]:
            return default
        
        return self.config[section][parameter]
    
    def set(self, section: str, parameter: str, value: Any) -> 'Configuration':
        """
        Set a configuration parameter.
        
        Parameters
        ----------
        section : str
            Configuration section.
        parameter : str
            Configuration parameter within the section.
        value : Any
            Value to set.
            
        Returns
        -------
        Configuration
            The Configuration object with updated parameters.
        """
        # Create section if it doesn't exist
        if section not in self.config:
            self.config[section] = {}
        
        # Set parameter value
        self.config[section][parameter] = value
        
        return self
    
    def get_gap_filling_params(self) -> Dict[str, Any]:
        """
        Get gap filling parameters.
        
        Returns
        -------
        dict
            Dictionary containing gap filling parameters.
        """
        return self.get('gap_filling', default={})
    
    def get_sswei_params(self) -> Dict[str, Any]:
        """
        Get SSWEI calculation parameters.
        
        Returns
        -------
        dict
            Dictionary containing SSWEI calculation parameters.
        """
        return self.get('sswei', default={})
    
    def get_drought_classification_thresholds(self) -> Dict[str, float]:
        """
        Get drought classification thresholds.
        
        Returns
        -------
        dict
            Dictionary containing drought classification thresholds.
        """
        return self.get('drought_classification', default={})
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """
        Get visualization settings.
        
        Returns
        -------
        dict
            Dictionary containing visualization settings.
        """
        return self.get('visualization', default={})
    
    def get_paths(self) -> Dict[str, str]:
        """
        Get data paths.
        
        Returns
        -------
        dict
            Dictionary containing data paths.
        """
        return self.get('paths', default={})
    
    def get_performance_params(self) -> Dict[str, Any]:
        """
        Get performance optimization parameters.
        
        Returns
        -------
        dict
            Dictionary containing performance optimization parameters.
        """
        return self.get('performance', default={})
    
    def set_performance_params(self, parallel: bool = None, n_jobs: int = None,
                              lazy_loading: bool = None, chunks: Dict[str, int] = None,
                              memory_efficient: bool = None, enable_caching: bool = None,
                              cache_dir: str = None, dask_client: Any = None) -> 'Configuration':
        """
        Set performance optimization parameters.
        
        Parameters
        ----------
        parallel : bool, optional
            Whether to use parallel processing, by default None (no change).
        n_jobs : int, optional
            Number of jobs for parallel processing, by default None (no change).
            Use -1 to use all available cores.
        lazy_loading : bool, optional
            Whether to use lazy loading for large datasets, by default None (no change).
        chunks : dict, optional
            Chunk sizes for lazy loading, by default None (no change).
            Example: {'time': 100, 'lat': 50, 'lon': 50}
        memory_efficient : bool, optional
            Whether to use memory-efficient algorithms, by default None (no change).
        enable_caching : bool, optional
            Whether to enable result caching, by default None (no change).
        cache_dir : str, optional
            Directory for caching results, by default None (no change).
        dask_client : Any, optional
            Dask client for distributed computing, by default None (no change).
            
        Returns
        -------
        Configuration
            The Configuration object with updated parameters.
        """
        performance_params = self.get_performance_params()
        
        # Update parameters if provided
        if parallel is not None:
            performance_params['parallel'] = parallel
        
        if n_jobs is not None:
            performance_params['n_jobs'] = n_jobs
        
        if lazy_loading is not None:
            performance_params['lazy_loading'] = lazy_loading
        
        if chunks is not None:
            performance_params['chunks'] = chunks
        
        if memory_efficient is not None:
            performance_params['memory_efficient'] = memory_efficient
        
        if enable_caching is not None:
            performance_params['enable_caching'] = enable_caching
        
        if cache_dir is not None:
            performance_params['cache_dir'] = cache_dir
        
        if dask_client is not None:
            performance_params['dask_client'] = dask_client
        
        # Update configuration
        self.config['performance'] = performance_params
        
        return self
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Configuration object.
        
        Returns
        -------
        str
            String representation of the Configuration object.
        """
        return f"Configuration(sections={list(self.config.keys())})"
    
    def __str__(self) -> str:
        """
        Return a string representation of the configuration.
        
        Returns
        -------
        str
            String representation of the configuration.
        """
        return json.dumps(self.config, indent=2)
