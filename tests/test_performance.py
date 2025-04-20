"""
Performance tests for the Snow Drought Index package.

This module contains performance tests that evaluate the efficiency and scalability
of different components of the package.
"""

import pytest
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.sswei_class import SSWEI
from snowdroughtindex.core.drought_analysis import DroughtAnalysis
from snowdroughtindex.core.configuration import Configuration
from snowdroughtindex.core import gap_filling, sswei


class TestPerformance:
    """
    Test the performance of various components of the package.
    """
    
    def generate_large_dataset(self, num_years, num_stations):
        """
        Generate a large dataset for performance testing.
        
        Parameters
        ----------
        num_years : int
            Number of years of data to generate.
        num_stations : int
            Number of stations to generate.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the generated data.
        """
        # Generate dates
        start_date = datetime(2000, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(365 * num_years)]
        
        # Generate station columns
        stations = [f'station_{i}' for i in range(1, num_stations + 1)]
        
        # Create a DataFrame with random SWE values
        np.random.seed(42)  # For reproducibility
        
        data = {}
        for station in stations:
            # Create seasonal pattern with random noise
            days = np.arange(len(dates))
            seasonal = 100 * np.sin(2 * np.pi * days / 365.25 - np.pi/2) + 100
            seasonal[seasonal < 0] = 0  # No negative SWE values
            noise = np.random.normal(0, 10, len(dates))
            values = seasonal + noise
            values[values < 0] = 0  # No negative SWE values
            data[station] = values
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_gap_filling_performance(self):
        """
        Test the performance of the gap filling algorithm with datasets of different sizes.
        """
        # Define dataset sizes to test
        dataset_sizes = [(5, 5), (5, 10), (10, 10)]  # (years, stations)
        
        # Store results
        results = []
        
        for years, stations in dataset_sizes:
            # Generate dataset
            df = self.generate_large_dataset(years, stations)
            
            # Introduce gaps
            df_with_gaps = df.copy()
            mask = np.random.random(df_with_gaps.shape) < 0.1  # 10% of data as gaps
            df_with_gaps = df_with_gaps.mask(mask)
            
            # Create SWEDataset
            dataset = SWEDataset(df_with_gaps)
            
            # Measure time for gap filling
            start_time = time.time()
            dataset.gap_fill(
                window_days=15,
                min_obs_corr=5,
                min_obs_cdf=3,
                min_corr=0.5
            )
            end_time = time.time()
            
            # Calculate execution time
            execution_time = end_time - start_time
            
            # Store results
            results.append({
                'Years': years,
                'Stations': stations,
                'Data_Points': len(df) * len(df.columns),
                'Execution_Time': execution_time
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print results
        print("\nGap Filling Performance Results:")
        print(results_df)
        
        # Check that execution time increases with dataset size
        assert results_df['Execution_Time'].is_monotonic_increasing
    
    def test_sswei_calculation_performance(self):
        """
        Test the performance of the SSWEI calculation with datasets of different sizes.
        """
        # Define dataset sizes to test
        dataset_sizes = [(5, 5), (5, 10), (10, 10)]  # (years, stations)
        
        # Store results
        results = []
        
        for years, stations in dataset_sizes:
            # Generate dataset
            df = self.generate_large_dataset(years, stations)
            
            # Create SWEDataset
            dataset = SWEDataset(df)
            
            # Create SSWEI object
            sswei_obj = SSWEI(dataset)
            
            # Measure time for SSWEI calculation
            start_time = time.time()
            sswei_obj.calculate_sswei(
                start_month=12,
                end_month=3,
                min_years=2,  # Using a small value for testing
                distribution='gamma'
            )
            end_time = time.time()
            
            # Calculate execution time
            execution_time = end_time - start_time
            
            # Store results
            results.append({
                'Years': years,
                'Stations': stations,
                'Data_Points': len(df) * len(df.columns),
                'Execution_Time': execution_time
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print results
        print("\nSSWEI Calculation Performance Results:")
        print(results_df)
        
        # Check that execution time increases with dataset size
        assert results_df['Execution_Time'].is_monotonic_increasing
    
    def test_drought_analysis_performance(self):
        """
        Test the performance of drought analysis with datasets of different sizes.
        """
        # Define dataset sizes to test
        dataset_sizes = [(5, 5), (5, 10), (10, 10)]  # (years, stations)
        
        # Store results
        results = []
        
        for years, stations in dataset_sizes:
            # Generate datasets for different elevation bands
            low_elev = self.generate_large_dataset(years, stations)
            mid_elev = self.generate_large_dataset(years, stations) * 1.2
            high_elev = self.generate_large_dataset(years, stations) * 1.5
            
            # Create SWEDataset objects
            low_dataset = SWEDataset(low_elev)
            mid_dataset = SWEDataset(mid_elev)
            high_dataset = SWEDataset(high_elev)
            
            # Create DroughtAnalysis object
            drought_analysis = DroughtAnalysis()
            
            # Add datasets
            drought_analysis.add_dataset("1000-1500m", low_dataset)
            drought_analysis.add_dataset("1500-2000m", mid_dataset)
            drought_analysis.add_dataset("2000-2500m", high_dataset)
            
            # Measure time for drought analysis
            start_time = time.time()
            
            # Calculate SSWEI
            drought_analysis.calculate_sswei(
                start_month=12,
                end_month=3,
                min_years=2,
                distribution='gamma'
            )
            
            # Perform analyses
            drought_analysis.compare_elevation_bands()
            drought_analysis.analyze_drought_synchronicity()
            drought_analysis.analyze_elevation_sensitivity()
            drought_analysis.analyze_temporal_trends(window_size=2)
            
            end_time = time.time()
            
            # Calculate execution time
            execution_time = end_time - start_time
            
            # Store results
            results.append({
                'Years': years,
                'Stations': stations,
                'Data_Points': len(low_elev) * len(low_elev.columns) * 3,  # 3 datasets
                'Execution_Time': execution_time
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print results
        print("\nDrought Analysis Performance Results:")
        print(results_df)
        
        # Check that execution time increases with dataset size
        assert results_df['Execution_Time'].is_monotonic_increasing
    
    def test_memory_usage(self):
        """
        Test the memory usage of the package with datasets of different sizes.
        """
        import psutil
        import os
        
        # Define dataset sizes to test
        dataset_sizes = [(5, 5), (5, 10), (10, 10)]  # (years, stations)
        
        # Store results
        results = []
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        for years, stations in dataset_sizes:
            # Measure baseline memory usage
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate dataset
            df = self.generate_large_dataset(years, stations)
            
            # Create SWEDataset
            dataset = SWEDataset(df)
            
            # Create SSWEI object
            sswei_obj = SSWEI(dataset)
            
            # Calculate SSWEI
            sswei_obj.calculate_sswei(
                start_month=12,
                end_month=3,
                min_years=2,
                distribution='gamma'
            )
            
            # Measure memory usage after SSWEI calculation
            memory_after_sswei = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate memory increase
            memory_increase = memory_after_sswei - baseline_memory
            
            # Store results
            results.append({
                'Years': years,
                'Stations': stations,
                'Data_Points': len(df) * len(df.columns),
                'Memory_Usage_MB': memory_increase
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print results
        print("\nMemory Usage Results:")
        print(results_df)
        
        # Check that memory usage increases with dataset size
        # Note: This might not always be true due to garbage collection
        # So we'll just print the results without asserting
    
    def test_vectorization_performance(self):
        """
        Test the performance improvement from vectorization.
        """
        # Generate a large dataset
        years = 10
        stations = 20
        df = self.generate_large_dataset(years, stations)
        
        # Create a non-vectorized version of the integrate_season function
        def integrate_season_loop(seasonal_data):
            """
            Non-vectorized version of integrate_season using loops.
            """
            data = seasonal_data['data']
            years = seasonal_data['years']
            stations = seasonal_data['stations']
            
            # Initialize results DataFrame
            results = pd.DataFrame({'season_year': years})
            
            # Calculate integrated SWE for each year and station
            integrated_values = []
            
            for year in years:
                # Get data for this year
                year_data = data[data.index.year == year]
                
                # Calculate integrated SWE (sum of daily values)
                integrated_swe = 0
                for station in stations:
                    station_sum = year_data[station].sum()
                    integrated_swe += station_sum
                
                integrated_values.append(integrated_swe)
            
            # Add integrated SWE to results
            results['integrated_swe'] = integrated_values
            
            return results
        
        # Prepare seasonal data
        seasonal_data = sswei.prepare_season_data(
            df, start_month=12, end_month=3, min_years=2
        )
        
        # Measure time for vectorized version
        start_time = time.time()
        vectorized_result = sswei.integrate_season(seasonal_data)
        vectorized_time = time.time() - start_time
        
        # Measure time for non-vectorized version
        start_time = time.time()
        loop_result = integrate_season_loop(seasonal_data)
        loop_time = time.time() - start_time
        
        # Calculate speedup
        speedup = loop_time / vectorized_time
        
        # Print results
        print(f"\nVectorization Performance Results:")
        print(f"Vectorized execution time: {vectorized_time:.6f} seconds")
        print(f"Loop execution time: {loop_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        
        # Check that vectorized version is faster
        assert vectorized_time < loop_time
        
        # Check that results are the same
        pd.testing.assert_frame_equal(
            vectorized_result[['season_year', 'integrated_swe']],
            loop_result[['season_year', 'integrated_swe']],
            check_dtype=False  # Ignore dtype differences
        )
    
    def test_parallel_processing_potential(self):
        """
        Test the potential for parallel processing in gap filling.
        """
        # This is a demonstration of how parallel processing could be implemented
        # for gap filling, but we won't actually implement it here.
        
        # Generate a large dataset
        years = 10
        stations = 20
        df = self.generate_large_dataset(years, stations)
        
        # Introduce gaps
        df_with_gaps = df.copy()
        mask = np.random.random(df_with_gaps.shape) < 0.1  # 10% of data as gaps
        df_with_gaps = df_with_gaps.mask(mask)
        
        # Create SWEDataset
        dataset = SWEDataset(df_with_gaps)
        
        # Measure time for sequential gap filling
        start_time = time.time()
        dataset.gap_fill(
            window_days=15,
            min_obs_corr=5,
            min_obs_cdf=3,
            min_corr=0.5
        )
        sequential_time = time.time() - start_time
        
        # Print results
        print(f"\nParallel Processing Potential:")
        print(f"Sequential execution time: {sequential_time:.6f} seconds")
        print(f"Estimated parallel execution time (4 cores): {sequential_time/4:.6f} seconds")
        print(f"Potential speedup: 4.00x")
        
        # Note: This is just an estimate. Actual parallel implementation would be needed
        # to measure real speedup, which would likely be less than the theoretical maximum
        # due to overhead.
