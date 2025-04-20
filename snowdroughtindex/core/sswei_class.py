"""
SSWEI class for the Snow Drought Index package.

This module contains the SSWEI class, which encapsulates methods for calculating
the Standardized Snow Water Equivalent Index (SSWEI) and classifying drought conditions.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import gc
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from snowdroughtindex.core import sswei
from snowdroughtindex.core import drought_classification
from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.configuration import Configuration
from snowdroughtindex.utils import visualization

class SSWEI:
    """
    A class for calculating the Standardized Snow Water Equivalent Index (SSWEI)
    and classifying drought conditions.
    
    This class provides methods for calculating SSWEI from SWE data, classifying
    drought conditions based on SSWEI values, and visualizing the results.
    
    Attributes
    ----------
    data : pandas.DataFrame
        The SWE data used for SSWEI calculation.
    sswei_data : pandas.DataFrame
        The calculated SSWEI values.
    drought_classifications : pandas.DataFrame
        The drought classifications based on SSWEI values.
    config : Configuration
        Configuration object with parameters for data processing and performance optimization.
    """
    
    def __init__(self, data: Optional[Union[pd.DataFrame, SWEDataset]] = None,
                config: Optional[Configuration] = None,
                parallel: bool = False,
                n_jobs: int = -1,
                memory_efficient: bool = False,
                enable_caching: bool = False,
                cache_dir: str = './.cache'):
        """
        Initialize an SSWEI object.
        
        Parameters
        ----------
        data : pandas.DataFrame or SWEDataset, optional
            The SWE data to use for SSWEI calculation. If None, data must be loaded
            using one of the load methods. If a SWEDataset is provided, its data
            attribute will be used.
        config : Configuration, optional
            Configuration object with parameters for data processing and performance optimization.
            If None, a default configuration will be used.
        parallel : bool, optional
            Whether to use parallel processing, by default False.
        n_jobs : int, optional
            Number of jobs for parallel processing, by default -1 (all available cores).
        memory_efficient : bool, optional
            Whether to use memory-efficient algorithms, by default False.
        enable_caching : bool, optional
            Whether to enable result caching, by default False.
        cache_dir : str, optional
            Directory for caching results, by default './.cache'.
        """
        # Set data
        if isinstance(data, SWEDataset):
            self.data = data.data
            # If SWEDataset has a configuration, use it
            if config is None and hasattr(data, 'config'):
                config = data.config
        else:
            self.data = data
        
        # Set configuration
        if config is None:
            self.config = Configuration()
        else:
            self.config = config
        
        # Set performance parameters
        performance_params = {
            'parallel': parallel,
            'n_jobs': n_jobs,
            'memory_efficient': memory_efficient,
            'enable_caching': enable_caching,
            'cache_dir': cache_dir
        }
        
        self.config.set_performance_params(**performance_params)
        
        # Create cache directory if caching is enabled
        if enable_caching and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        self.sswei_data = None
        self.drought_classifications = None
        self.thresholds = drought_classification.DEFAULT_THRESHOLDS.copy()
        
        # Cache for storing intermediate results
        self._cache = {}
    
    def load_data(self, file_path: str) -> 'SSWEI':
        """
        Load SWE data from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the file containing SWE data.
            
        Returns
        -------
        SSWEI
            The SSWEI object with loaded data.
        """
        dataset = SWEDataset()
        dataset.load_from_file(file_path)
        self.data = dataset.data
        return self
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> 'SSWEI':
        """
        Set custom thresholds for drought classification.
        
        Parameters
        ----------
        thresholds : dict
            Dictionary of threshold values for different drought classifications.
            
        Returns
        -------
        SSWEI
            The SSWEI object with updated thresholds.
        """
        self.thresholds.update(thresholds)
        return self
    
    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key based on the input parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Input parameters to hash.
            
        Returns
        -------
        str
            Cache key.
        """
        # Convert parameters to a string and hash it
        param_str = str(sorted(kwargs.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key.
            
        Returns
        -------
        Any
            Cached value, or None if not found.
        """
        # Get performance parameters
        perf_params = self.config.get_performance_params()
        enable_caching = perf_params.get('enable_caching', False)
        cache_dir = perf_params.get('cache_dir', './.cache')
        
        if not enable_caching:
            return None
        
        # Check in-memory cache first
        if cache_key in self._cache:
            print(f"Using in-memory cached result for {cache_key}")
            return self._cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                print(f"Using disk cached result for {cache_key}")
                # Store in memory for faster access next time
                self._cache[cache_key] = result
                return result
            except Exception as e:
                print(f"Error loading cached result: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, value: Any) -> None:
        """
        Save a value to the cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key.
        value : Any
            Value to cache.
        """
        # Get performance parameters
        perf_params = self.config.get_performance_params()
        enable_caching = perf_params.get('enable_caching', False)
        cache_dir = perf_params.get('cache_dir', './.cache')
        
        if not enable_caching:
            return
        
        # Save to in-memory cache
        self._cache[cache_key] = value
        
        # Save to disk cache
        try:
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            print(f"Saved result to cache: {cache_key}")
        except Exception as e:
            print(f"Error saving result to cache: {e}")
    
    def calculate_sswei(self, 
                       start_month: int, 
                       end_month: int, 
                       min_years: int = 10,
                       distribution: str = 'gamma',
                       reference_period: Optional[Tuple[int, int]] = None,
                       parallel: bool = None,
                       n_jobs: int = None,
                       memory_efficient: bool = None) -> 'SSWEI':
        """
        Calculate the Standardized Snow Water Equivalent Index (SSWEI).
        
        Parameters
        ----------
        start_month : int
            Starting month of the season (1-12).
        end_month : int
            Ending month of the season (1-12).
        min_years : int, optional
            Minimum number of years required for calculation, by default 10.
        distribution : str, optional
            Probability distribution to use, by default 'gamma'.
            Options: 'gamma', 'normal'.
        reference_period : tuple, optional
            Reference period (start_year, end_year) for standardization.
            If None, the entire period is used.
        parallel : bool, optional
            Whether to use parallel processing, by default None (use configuration value).
        n_jobs : int, optional
            Number of jobs for parallel processing, by default None (use configuration value).
        memory_efficient : bool, optional
            Whether to use memory-efficient algorithms, by default None (use configuration value).
            
        Returns
        -------
        SSWEI
            The SSWEI object with calculated SSWEI values.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before calculating SSWEI.")
        
        # Get performance parameters from configuration
        perf_params = self.config.get_performance_params()
        
        # Use provided parameters or get from configuration
        if parallel is None:
            parallel = perf_params.get('parallel', False)
        
        if n_jobs is None:
            n_jobs = perf_params.get('n_jobs', -1)
        
        if memory_efficient is None:
            memory_efficient = perf_params.get('memory_efficient', False)
        
        # Print performance settings
        if parallel:
            print(f"Running SSWEI calculation with parallel processing (n_jobs={n_jobs})")
        if memory_efficient:
            print("Using memory-efficient algorithms for SSWEI calculation")
        
        # Generate cache key
        cache_params = {
            'data_hash': hash(str(self.data.shape) + str(self.data.index[0]) + str(self.data.index[-1])),
            'start_month': start_month,
            'end_month': end_month,
            'min_years': min_years,
            'distribution': distribution,
            'reference_period': reference_period
        }
        cache_key = self._generate_cache_key(**cache_params)
        
        # Check cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.sswei_data = cached_result
            # Classify drought conditions
            self.classify_drought()
            return self
        
        # Prepare data for SSWEI calculation
        seasonal_data = sswei.prepare_season_data(
            self.data, start_month, end_month, min_years
        )
        
        # Calculate integrated SWE
        if memory_efficient:
            # Process in smaller chunks to reduce memory usage
            # This is a simplified example - actual implementation would depend on the data structure
            integrated_swe = pd.DataFrame()
            chunk_size = max(1, len(seasonal_data) // 4)
            
            for i in range(0, len(seasonal_data), chunk_size):
                chunk = seasonal_data.iloc[i:i+chunk_size]
                chunk_integrated = sswei.integrate_season(chunk)
                integrated_swe = pd.concat([integrated_swe, chunk_integrated])
                
                # Free memory
                del chunk
                gc.collect()
        else:
            integrated_swe = sswei.integrate_season(seasonal_data)
        
        # Calculate SSWEI with or without parallel processing
        if parallel:
            # Define a function to process a subset of data
            def process_subset(subset_data):
                return sswei.calculate_sswei(subset_data, distribution, reference_period)
            
            # Split data into chunks for parallel processing
            n_jobs = min(n_jobs if n_jobs > 0 else mp.cpu_count(), mp.cpu_count())
            chunk_size = max(1, len(integrated_swe) // n_jobs)
            chunks = [integrated_swe.iloc[i:i+chunk_size] for i in range(0, len(integrated_swe), chunk_size)]
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_subset, chunks))
            
            # Combine results
            self.sswei_data = pd.concat(results)
        else:
            # Calculate SSWEI normally
            self.sswei_data = sswei.calculate_sswei(
                integrated_swe, distribution, reference_period
            )
        
        # Save to cache
        self._save_to_cache(cache_key, self.sswei_data)
        
        # Classify drought conditions
        self.classify_drought()
        
        return self
    
    def classify_drought(self, thresholds: Optional[Dict[str, float]] = None) -> 'SSWEI':
        """
        Classify drought conditions based on SSWEI values.
        
        Parameters
        ----------
        thresholds : dict, optional
            Dictionary of threshold values for different drought classifications.
            If None, the thresholds set in the object are used.
            
        Returns
        -------
        SSWEI
            The SSWEI object with drought classifications.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before classifying drought conditions.")
        
        if thresholds is not None:
            self.thresholds.update(thresholds)
        
        # Apply drought classification to SSWEI values
        self.sswei_data['Drought_Classification'] = self.sswei_data['SWEI'].apply(
            lambda x: drought_classification.classify_drought(x, self.thresholds)
        )
        
        return self
    
    def calculate_drought_characteristics(self) -> pd.DataFrame:
        """
        Calculate drought characteristics such as duration, severity, and intensity.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing drought characteristics.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before analyzing drought characteristics.")
        
        drought_chars = drought_classification.calculate_drought_characteristics(
            self.sswei_data,
            year_column='season_year',
            swei_column='SWEI'
        )
        
        return drought_chars
    
    def analyze_drought_trends(self, window_size: int = 10) -> pd.DataFrame:
        """
        Analyze drought trends over time using a moving window approach.
        
        Parameters
        ----------
        window_size : int, optional
            Size of the moving window in years, by default 10.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing drought trend analysis.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before analyzing drought trends.")
        
        trend_data = drought_classification.analyze_drought_trends(
            self.sswei_data,
            year_column='season_year',
            swei_column='SWEI',
            window_size=window_size
        )
        
        return trend_data
    
    def plot_sswei_timeseries(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot SSWEI time series with drought classification color coding.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before plotting.")
        
        fig = visualization.plot_sswei_timeseries(
            self.sswei_data,
            year_column='season_year',
            swei_column='SWEI',
            classification_column='Drought_Classification',
            figsize=figsize
        )
        
        return fig
    
    def plot_drought_characteristics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot drought characteristics including duration, severity, and intensity.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        drought_chars = self.calculate_drought_characteristics()
        
        if drought_chars.empty:
            print("No drought events to plot.")
            return None
        
        fig = drought_classification.plot_drought_characteristics(
            drought_chars,
            figsize=figsize
        )
        
        return fig
    
    def plot_drought_trends(self, window_size: int = 10, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot drought trends over time.
        
        Parameters
        ----------
        window_size : int, optional
            Size of the moving window in years, by default 10.
        figsize : tuple, optional
            Figure size, by default (12, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        trend_data = self.analyze_drought_trends(window_size)
        
        fig = drought_classification.plot_drought_trends(
            trend_data,
            figsize=figsize
        )
        
        return fig
    
    def plot_drought_classification_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a heatmap of drought classifications by decade.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before plotting.")
        
        fig = visualization.plot_drought_classification_heatmap(
            self.sswei_data,
            figsize=figsize
        )
        
        return fig
    
    def plot_drought_severity_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the distribution of drought severity values.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (10, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before plotting.")
        
        fig = visualization.plot_drought_severity_distribution(
            self.sswei_data,
            figsize=figsize
        )
        
        return fig
    
    def save_results(self, file_path: str) -> None:
        """
        Save SSWEI results to a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to save the results to.
        """
        if self.sswei_data is None:
            raise ValueError("SSWEI must be calculated before saving results.")
        
        self.sswei_data.to_csv(file_path, index=False)
        print(f"SSWEI results saved to {file_path}")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the SSWEI object.
        
        Returns
        -------
        str
            String representation of the SSWEI object.
        """
        if self.sswei_data is None:
            return "SSWEI(data=<loaded>, sswei=<not calculated>)"
        else:
            return f"SSWEI(data=<loaded>, sswei=<calculated for {len(self.sswei_data)} years>)"
