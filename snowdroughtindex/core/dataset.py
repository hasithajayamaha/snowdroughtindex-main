"""
SWEDataset class for the Snow Drought Index package.

This module contains the SWEDataset class, which encapsulates methods for loading,
preprocessing, and gap filling SWE data.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import lru_cache

from snowdroughtindex.core import data_preparation, gap_filling
from snowdroughtindex.core.configuration import Configuration

class SWEDataset:
    """
    A class for managing Snow Water Equivalent (SWE) data.
    
    This class provides methods for loading, preprocessing, and gap filling SWE data.
    It encapsulates the functionality from the data_preparation and gap_filling modules
    in an object-oriented interface.
    
    Attributes
    ----------
    data : xarray.Dataset or pandas.DataFrame
        The SWE data.
    stations : pandas.DataFrame
        Information about the SWE stations.
    config : Configuration
        Configuration object with parameters for data processing and performance optimization.
    """
    
    def __init__(self, data: Optional[Union[xr.Dataset, pd.DataFrame]] = None, 
                 stations: Optional[pd.DataFrame] = None,
                 config: Optional[Configuration] = None,
                 parallel: bool = False,
                 n_jobs: int = -1,
                 lazy_loading: bool = False,
                 chunks: Optional[Dict[str, int]] = None,
                 memory_efficient: bool = False,
                 enable_caching: bool = False,
                 cache_dir: str = './.cache',
                 dask_client: Any = None):
        """
        Initialize a SWEDataset object.
        
        Parameters
        ----------
        data : xarray.Dataset or pandas.DataFrame, optional
            The SWE data. If None, data must be loaded using one of the load methods.
        stations : pandas.DataFrame, optional
            Information about the SWE stations. If None, station information will be
            extracted from the data if possible.
        config : Configuration, optional
            Configuration object with parameters for data processing and performance optimization.
            If None, a default configuration will be used.
        parallel : bool, optional
            Whether to use parallel processing, by default False.
        n_jobs : int, optional
            Number of jobs for parallel processing, by default -1 (all available cores).
        lazy_loading : bool, optional
            Whether to use lazy loading for large datasets, by default False.
        chunks : dict, optional
            Chunk sizes for lazy loading, by default None.
            Example: {'time': 100, 'lat': 50, 'lon': 50}
        memory_efficient : bool, optional
            Whether to use memory-efficient algorithms, by default False.
        enable_caching : bool, optional
            Whether to enable result caching, by default False.
        cache_dir : str, optional
            Directory for caching results, by default './.cache'.
        dask_client : Any, optional
            Dask client for distributed computing, by default None.
        """
        self.data = data
        self.stations = stations
        self.data_type_flags = None
        self.donor_stations = None
        
        # Set configuration
        if config is None:
            self.config = Configuration()
        else:
            self.config = config
        
        # Set performance parameters
        performance_params = {
            'parallel': parallel,
            'n_jobs': n_jobs,
            'lazy_loading': lazy_loading,
            'memory_efficient': memory_efficient,
            'enable_caching': enable_caching,
            'cache_dir': cache_dir,
            'dask_client': dask_client
        }
        
        if chunks is not None:
            performance_params['chunks'] = chunks
        
        self.config.set_performance_params(**performance_params)
        
        # Create cache directory if caching is enabled
        if enable_caching and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def load_from_file(self, file_path: str, chunks: Optional[Dict[str, int]] = None) -> 'SWEDataset':
        """
        Load SWE data from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the file containing SWE data.
        chunks : dict, optional
            Chunk sizes for lazy loading, by default None.
            If None, uses the chunks from the configuration.
            Example: {'time': 100, 'lat': 50, 'lon': 50}
            
        Returns
        -------
        SWEDataset
            The SWEDataset object with loaded data.
        """
        # Get performance parameters
        perf_params = self.config.get_performance_params()
        lazy_loading = perf_params.get('lazy_loading', False)
        
        # Use provided chunks or get from configuration
        if chunks is None and lazy_loading:
            chunks = perf_params.get('chunks', {'time': 100, 'lat': 50, 'lon': 50})
        
        # Load data with or without lazy loading
        if lazy_loading and file_path.endswith(('.nc', '.netcdf')):
            # Use xarray's open_dataset with dask for lazy loading
            self.data = xr.open_dataset(file_path, chunks=chunks)
            print(f"Loaded data with lazy loading (chunks: {chunks})")
        else:
            # Use standard data loading
            self.data = data_preparation.load_swe_data(file_path)
        
        return self
    
    def preprocess(self) -> 'SWEDataset':
        """
        Preprocess the SWE data.
        
        This method converts the data to a pandas DataFrame if it is an xarray Dataset,
        and performs any necessary preprocessing steps.
        
        Returns
        -------
        SWEDataset
            The SWEDataset object with preprocessed data.
        """
        if isinstance(self.data, xr.Dataset):
            self.data = data_preparation.preprocess_swe(self.data)
        
        # Ensure the DataFrame has a time index
        if isinstance(self.data, pd.DataFrame) and 'time' in self.data.columns:
            self.data = self.data.set_index('time')
        
        return self
    
    def extract_stations_in_basin(self, basin_id: str, basins: pd.DataFrame, 
                                  buffer_km: float = 0) -> 'SWEDataset':
        """
        Extract stations within a specified basin.
        
        Parameters
        ----------
        basin_id : str
            ID of the basin to extract stations from.
        basins : pandas.DataFrame
            DataFrame containing basin information.
        buffer_km : float, optional
            Buffer around the basin in kilometers, by default 0.
            
        Returns
        -------
        SWEDataset
            A new SWEDataset object containing only the stations within the basin.
        """
        if self.stations is None:
            raise ValueError("Station information is required for basin extraction.")
        
        stations_in_basin, _ = data_preparation.extract_stations_in_basin(
            self.stations, basins, basin_id, buffer_km
        )
        
        # Filter data to include only stations in the basin
        if isinstance(self.data, xr.Dataset):
            basin_data = self.data.sel(station_id=stations_in_basin.station_id.values)
        else:  # pandas.DataFrame
            station_cols = [col for col in self.data.columns 
                           if col in stations_in_basin.station_id.values]
            basin_data = self.data[station_cols]
        
        return SWEDataset(basin_data, stations_in_basin)
    
    def extract_monthly_data(self, month: int, plot: bool = False) -> 'SWEDataset':
        """
        Extract data for the first day of a given month.
        
        Parameters
        ----------
        month : int
            Month to extract data for (1-12).
        plot : bool, optional
            Whether to plot the evolution of the selection criteria, by default False.
            
        Returns
        -------
        SWEDataset
            A new SWEDataset object containing only the data for the specified month.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame for monthly extraction.")
        
        month_data = data_preparation.extract_monthly_data(self.data, month, plot)
        return SWEDataset(month_data, self.stations)
    
    def gap_fill(self, window_days: int = 15, min_obs_corr: int = 10, 
                min_obs_cdf: int = 5, min_corr: float = 0.7,
                parallel: bool = None, n_jobs: int = None,
                memory_efficient: bool = None) -> 'SWEDataset':
        """
        Perform gap filling on the SWE data.
        
        Parameters
        ----------
        window_days : int, optional
            Number of days to select data for around a certain day of year, by default 15.
        min_obs_corr : int, optional
            Minimum number of overlapping observations required to calculate correlation, by default 10.
        min_obs_cdf : int, optional
            Minimum number of stations required to calculate a station's cdf, by default 5.
        min_corr : float, optional
            Minimum correlation value required to keep a donor station, by default 0.7.
        parallel : bool, optional
            Whether to use parallel processing, by default None (use configuration value).
        n_jobs : int, optional
            Number of jobs for parallel processing, by default None (use configuration value).
        memory_efficient : bool, optional
            Whether to use memory-efficient algorithms, by default None (use configuration value).
            
        Returns
        -------
        SWEDataset
            The SWEDataset object with gap-filled data.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame for gap filling.")
        
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
            print(f"Running gap filling with parallel processing (n_jobs={n_jobs})")
        if memory_efficient:
            print("Using memory-efficient algorithms for gap filling")
        
        # Implement memory-efficient approach
        if memory_efficient:
            # Process data in smaller chunks to reduce memory usage
            # First, identify stations with gaps
            stations_with_gaps = []
            for col in self.data.columns:
                if self.data[col].isna().any():
                    stations_with_gaps.append(col)
            
            # Process stations in batches
            batch_size = max(1, len(stations_with_gaps) // (4 * (n_jobs if parallel else 1)))
            all_gapfilled_data = self.data.copy()
            all_data_type_flags = pd.DataFrame(index=self.data.index, columns=self.data.columns)
            all_donor_stations = {}
            
            for i in range(0, len(stations_with_gaps), batch_size):
                batch_stations = stations_with_gaps[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(stations_with_gaps) + batch_size - 1)//batch_size}: {len(batch_stations)} stations")
                
                # Create a subset of data with only the stations in this batch and their potential donors
                batch_data = self.data[batch_stations].copy()
                
                # Run gap filling on this batch
                batch_gapfilled, batch_flags, batch_donors = gap_filling.qm_gap_filling(
                    batch_data, window_days, min_obs_corr, min_obs_cdf, min_corr,
                    parallel=parallel, n_jobs=n_jobs
                )
                
                # Update results
                for col in batch_stations:
                    all_gapfilled_data[col] = batch_gapfilled[col]
                    all_data_type_flags[col] = batch_flags[col]
                    if col in batch_donors:
                        all_donor_stations[col] = batch_donors[col]
                
                # Free memory
                del batch_data, batch_gapfilled, batch_flags, batch_donors
                gc.collect()
            
            self.data = all_gapfilled_data
            self.data_type_flags = all_data_type_flags
            self.donor_stations = all_donor_stations
        else:
            # Run standard gap filling with or without parallel processing
            gapfilled_data, data_type_flags, donor_stations = gap_filling.qm_gap_filling(
                self.data, window_days, min_obs_corr, min_obs_cdf, min_corr,
                parallel=parallel, n_jobs=n_jobs
            )
            
            self.data = gapfilled_data
            self.data_type_flags = data_type_flags
            self.donor_stations = donor_stations
        
        return self
    
    def evaluate_gap_filling(self, iterations: int = 3, artificial_gap_perc: int = 20,
                            window_days: int = 15, min_obs_corr: int = 10, 
                            min_obs_cdf: int = 5, min_corr: float = 0.7,
                            min_obs_KGE: int = 5, plot: bool = False) -> Dict[str, np.ndarray]:
        """
        Evaluate gap filling performance using artificial gaps.
        
        Parameters
        ----------
        iterations : int, optional
            Number of iterations for artificial gap filling, by default 3.
        artificial_gap_perc : int, optional
            Percentage of data to remove for artificial gap filling, by default 20.
        window_days : int, optional
            Number of days to select data for around a certain day of year, by default 15.
        min_obs_corr : int, optional
            Minimum number of overlapping observations required to calculate correlation, by default 10.
        min_obs_cdf : int, optional
            Minimum number of stations required to calculate a station's cdf, by default 5.
        min_corr : float, optional
            Minimum correlation value required to keep a donor station, by default 0.7.
        min_obs_KGE : int, optional
            Minimum number of observations for KGE calculation, by default 5.
        plot : bool, optional
            Whether to plot the evaluation results, by default False.
            
        Returns
        -------
        dict
            Dictionary containing the evaluation results.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame for gap filling evaluation.")
        
        if plot:
            evaluation, fig = gap_filling.artificial_gap_filling(
                self.data, iterations, artificial_gap_perc, window_days, 
                min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag=1
            )
            return evaluation, fig
        else:
            evaluation = gap_filling.artificial_gap_filling(
                self.data, iterations, artificial_gap_perc, window_days, 
                min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag=0
            )
            return evaluation
    
    def plot_data_availability(self, gapfilled_data: Optional['SWEDataset'] = None, 
                              figsize: Tuple[int, int] = (14, 8)):
        """
        Plot the percentage of SWE stations available on the first day of each month of each year.
        
        Parameters
        ----------
        gapfilled_data : SWEDataset, optional
            Gap-filled SWE data for comparison, by default None.
        figsize : tuple, optional
            Figure size, by default (14, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        from snowdroughtindex.utils.visualization import plot_data_availability
        
        if self.stations is None:
            raise ValueError("Station information is required for data availability plotting.")
        
        if gapfilled_data is not None:
            return plot_data_availability(
                self.stations, self.data, gapfilled_data.data, figsize
            )
        else:
            return plot_data_availability(
                self.stations, self.data, None, figsize
            )
    
    def plot_gap_filling_evaluation(self, evaluation: Dict[str, np.ndarray], 
                                   figsize: Tuple[int, int] = (9, 5)):
        """
        Plot evaluation results for the artificial gap filling.
        
        Parameters
        ----------
        evaluation : dict
            Dictionary containing the artificial gap filling evaluation results.
        figsize : tuple, optional
            Figure size, by default (9, 5).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        from snowdroughtindex.utils.visualization import plot_gap_filling_evaluation
        return plot_gap_filling_evaluation(evaluation, figsize)
    
    def to_xarray(self) -> xr.Dataset:
        """
        Convert the data to an xarray Dataset.
        
        Returns
        -------
        xarray.Dataset
            The data as an xarray Dataset.
        """
        if isinstance(self.data, xr.Dataset):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return xr.Dataset.from_dataframe(self.data)
        else:
            raise ValueError("Data must be a pandas DataFrame or xarray Dataset.")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the data to a pandas DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            The data as a pandas DataFrame.
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, xr.Dataset):
            return self.data.to_dataframe()
        else:
            raise ValueError("Data must be a pandas DataFrame or xarray Dataset.")
    
    def save(self, file_path: str, format: str = 'netcdf') -> None:
        """
        Save the data to a file.
        
        Parameters
        ----------
        file_path : str
            Path to save the data to.
        format : str, optional
            Format to save the data in, by default 'netcdf'.
            Options: 'netcdf', 'csv'.
        """
        if format.lower() == 'netcdf':
            self.to_xarray().to_netcdf(file_path)
        elif format.lower() == 'csv':
            self.to_dataframe().to_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'netcdf' or 'csv'.")
    
    def calculate_daily_mean(self) -> pd.DataFrame:
        """
        Calculate the daily mean SWE across all stations.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the daily mean SWE values.
        """
        df = self.to_dataframe()
        
        # Identify SWE columns (excluding metadata columns)
        swe_columns = [col for col in df.columns if col not in ['station_id', 'lat', 'lon', 'elevation']]
        
        # Calculate daily mean SWE across all stations
        daily_mean = pd.DataFrame({
            'date': df.index,
            'mean_SWE': df[swe_columns].mean(axis=1)
        })
        
        return daily_mean
    
    def __repr__(self) -> str:
        """
        Return a string representation of the SWEDataset object.
        
        Returns
        -------
        str
            String representation of the SWEDataset object.
        """
        if self.data is None:
            return "SWEDataset(data=None)"
        
        if isinstance(self.data, xr.Dataset):
            return f"SWEDataset(data=<xarray.Dataset with {len(self.data.variables)} variables>)"
        elif isinstance(self.data, pd.DataFrame):
            return f"SWEDataset(data=<pandas.DataFrame with shape {self.data.shape}>)"
        else:
            return f"SWEDataset(data=<{type(self.data).__name__}>)"
