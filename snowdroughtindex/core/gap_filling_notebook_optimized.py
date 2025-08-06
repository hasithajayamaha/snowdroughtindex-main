"""
Optimized gap filling module for the Snow Drought Index package.

This module contains optimized functions for filling gaps in SWE data using quantile mapping
and evaluating the performance of gap filling methods. Focuses on performance improvements
while maintaining the original logic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
import sys
from sklearn.metrics import mean_squared_error
import datetime
from pathlib import Path
from scipy.interpolate import interp1d
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def artificial_gap_filling_optimized(original_data, iterations, artificial_gap_perc, window_days, min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag, n_jobs=None):
    """
    Optimized version of artificial gap filling with significant performance improvements.
    
    Key optimizations:
    1. Pre-computation of all correlation data and time windows
    2. Vectorized operations where possible
    3. Efficient data caching and lookup structures
    4. Batch processing of similar operations
    5. Optional parallel processing for iterations
    6. Memory-efficient data structures
    
    Parameters are the same as the original function, with additional:
    - n_jobs: Number of parallel jobs (None for auto-detection, 1 for sequential)
    """
    
    # Suppress warnings for cleaner output
    pd.set_option("mode.chained_assignment", None)
    
    print("Starting optimized artificial gap filling...")
    
    # Set up the figure (unchanged)
    if flag == 1:
        ncols = 3
        fig, axs = plt.subplots(4, ncols, sharex=False, sharey=False, figsize=(8,10))
        plot_col = -1
        row = 0

    # Identify stations for gap filling
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]
    
    # Initialize evaluation dictionary
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan

    # Pre-compute all necessary data structures
    print("Pre-computing data structures...")
    precomputed_data = _precompute_data_structures(original_data, cols, window_days, min_obs_corr)
    
    # Determine number of jobs for parallel processing
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), iterations)
    elif n_jobs == 1:
        n_jobs = 1
    
    print(f"Using {n_jobs} parallel jobs for {iterations} iterations")
    
    # Process iterations in parallel if n_jobs > 1
    if n_jobs > 1 and iterations > 1:
        evaluation = _process_iterations_parallel(
            original_data, cols, iterations, artificial_gap_perc, 
            min_obs_cdf, min_corr, min_obs_KGE, precomputed_data, 
            n_jobs, flag, axs if flag == 1 else None
        )
    else:
        evaluation = _process_iterations_sequential(
            original_data, cols, iterations, artificial_gap_perc,
            min_obs_cdf, min_corr, min_obs_KGE, precomputed_data,
            flag, axs if flag == 1 else None
        )
    
    print("Optimized artificial gap filling completed!")
    
    if flag == 1:
        plt.tight_layout()
        return evaluation, fig
    else:
        return evaluation

def _precompute_data_structures(original_data, cols, window_days, min_obs_corr):
    """Pre-compute all data structures needed for gap filling."""
    
    # Add DOY to original data
    original_data_with_doy = original_data.copy()
    original_data_with_doy['doy'] = original_data_with_doy.index.dayofyear
    
    # Calculate correlations (this is the most expensive operation)
    print("Calculating correlations...")
    corr = calculate_stations_doy_corr_optimized(original_data_with_doy, window_days, min_obs_corr)
    
    # Pre-compute monthly DOY values and windows
    monthly_data = {}
    for mo in range(1, 13):
        doy = int(datetime.datetime(2010, mo, 1).strftime('%j'))
        window_startdoy = (doy - window_days) % 365
        window_startdoy = 365 if window_startdoy == 0 else window_startdoy
        window_enddoy = (doy + window_days) % 365
        window_enddoy = 366 if window_enddoy == 0 or window_enddoy == 365 else window_enddoy
        monthly_data[mo] = {
            'doy': doy,
            'window_startdoy': window_startdoy,
            'window_enddoy': window_enddoy
        }
    
    # Pre-compute station data with DOY for each station
    station_data_cache = {}
    station_window_cache = {}
    
    for s in cols:
        station_data = original_data[s].dropna()
        if len(station_data) > 0:
            station_df = pd.DataFrame({s: station_data})
            station_df['doy'] = station_df.index.dayofyear
            station_data_cache[s] = station_df
            
            # Pre-compute windowed data for each month
            station_window_cache[s] = {}
            for mo in range(1, 13):
                window_startdoy = monthly_data[mo]['window_startdoy']
                window_enddoy = monthly_data[mo]['window_enddoy']
                
                if window_startdoy > window_enddoy:
                    mask = (station_df['doy'] >= window_startdoy) | (station_df['doy'] <= window_enddoy)
                else:
                    mask = (station_df['doy'] >= window_startdoy) & (station_df['doy'] <= window_enddoy)
                
                station_window_cache[s][mo] = station_df[mask]
    
    return {
        'corr': corr,
        'monthly_data': monthly_data,
        'station_data_cache': station_data_cache,
        'station_window_cache': station_window_cache,
        'original_data_with_doy': original_data_with_doy
    }

def _process_iterations_sequential(original_data, cols, iterations, artificial_gap_perc,
                                 min_obs_cdf, min_corr, min_obs_KGE, precomputed_data,
                                 flag, axs):
    """Process iterations sequentially with progress bar."""
    
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan
    
    total_combinations = 12 * len(cols) * iterations
    
    # Create progress bar for sequential processing
    pbar = tqdm(total=total_combinations, desc="Sequential Processing", 
                unit="combinations", ncols=100)
    
    try:
        # Process each month-iteration-station combination
        for mo in range(1, 13):
            # Controls for plotting
            if flag == 1:
                plot_col = (mo - 1) % 3
                row = (mo - 1) // 3
            
            for i in range(iterations):
                for elem, s in enumerate(cols):
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'Month': mo, 
                        'Iteration': i+1, 
                        'Station': f"{elem+1}/{len(cols)}"
                    })
                    
                    # Process single combination
                    result = _process_single_combination(
                        original_data, s, mo, i, artificial_gap_perc,
                        min_obs_cdf, min_corr, min_obs_KGE, precomputed_data
                    )
                    
                    if result is not None:
                        results_df, metrics_dict = result
                        
                        # Store metrics
                        for metric in metrics:
                            if metric in metrics_dict:
                                evaluation[metric][mo-1, elem, i] = metrics_dict[metric]
                        
                        # Plot if requested
                        if flag == 1 and not results_df.empty:
                            axs[row, plot_col].scatter(results_df['obs'], results_df['pre'], color='b', alpha=.3)
                            axs[row, plot_col].set_title('month' + str(mo))
                            if row == 3 and plot_col == 0:
                                axs[row, plot_col].set_xlabel('observed')
                                axs[row, plot_col].set_ylabel('infilling')
    
    finally:
        pbar.close()
    
    return evaluation

def _process_iterations_parallel(original_data, cols, iterations, artificial_gap_perc,
                               min_obs_cdf, min_corr, min_obs_KGE, precomputed_data,
                               n_jobs, flag, axs):
    """Process iterations in parallel with progress bar."""
    
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan
    
    # Create all combinations to process
    combinations = []
    for mo in range(1, 13):
        for i in range(iterations):
            for elem, s in enumerate(cols):
                combinations.append((mo, i, elem, s))
    
    print(f"Processing {len(combinations)} combinations in parallel...")
    
    # Process combinations in parallel
    process_func = partial(
        _process_single_combination_wrapper,
        original_data=original_data,
        artificial_gap_perc=artificial_gap_perc,
        min_obs_cdf=min_obs_cdf,
        min_corr=min_corr,
        min_obs_KGE=min_obs_KGE,
        precomputed_data=precomputed_data
    )
    
    # Create progress bar for parallel processing
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks and track with progress bar
        with tqdm(total=len(combinations), desc="Parallel Processing", 
                  unit="combinations", ncols=100) as pbar:
            
            # Use executor.map with tqdm wrapper
            results = []
            for result in executor.map(process_func, combinations):
                results.append(result)
                pbar.update(1)
                
                # Update postfix with current progress info
                completed = len(results)
                pbar.set_postfix({
                    'Workers': n_jobs,
                    'Completed': f"{completed}/{len(combinations)}"
                })
    
    # Collect results
    for idx, (mo, i, elem, s) in enumerate(combinations):
        result = results[idx]
        if result is not None:
            results_df, metrics_dict = result
            
            # Store metrics
            for metric in metrics:
                if metric in metrics_dict:
                    evaluation[metric][mo-1, elem, i] = metrics_dict[metric]
            
            # Plot if requested (sequential for thread safety)
            if flag == 1 and not results_df.empty:
                plot_col = (mo - 1) % 3
                row = (mo - 1) // 3
                axs[row, plot_col].scatter(results_df['obs'], results_df['pre'], color='b', alpha=.3)
                axs[row, plot_col].set_title('month' + str(mo))
                if row == 3 and plot_col == 0:
                    axs[row, plot_col].set_xlabel('observed')
                    axs[row, plot_col].set_ylabel('infilling')
    
    return evaluation

def _process_single_combination_wrapper(combination, original_data, artificial_gap_perc,
                                      min_obs_cdf, min_corr, min_obs_KGE, precomputed_data):
    """Wrapper for parallel processing."""
    mo, i, elem, s = combination
    return _process_single_combination(
        original_data, s, mo, i, artificial_gap_perc,
        min_obs_cdf, min_corr, min_obs_KGE, precomputed_data
    )

def _process_single_combination(original_data, station, month, iteration, artificial_gap_perc,
                              min_obs_cdf, min_corr, min_obs_KGE, precomputed_data):
    """Process a single month-station-iteration combination."""
    
    # Skip if station has no cached data
    if station not in precomputed_data['station_window_cache']:
        return None
    
    # Get pre-computed data for this station and month
    data_window = precomputed_data['station_window_cache'][station][month]
    
    if len(data_window) == 0:
        return None
    
    # Calculate number of observations to remove
    n = int(len(data_window) * artificial_gap_perc / 100)
    
    if n == 0:
        return None
    
    # Create artificial gaps
    if artificial_gap_perc == 100:
        dates_to_remove = data_window.index
    else:
        dates_to_remove = data_window.index[random.sample(range(len(data_window)), n)]
    
    # Create artificial gaps data
    artificial_gaps_data = original_data.copy()
    artificial_gaps_data.loc[dates_to_remove, station] = np.nan
    artificial_gaps_data = artificial_gaps_data.loc[dates_to_remove]
    
    # Gap fill the data
    gapfilled_data = artificial_gaps_data[station].copy()
    data_window_target = data_window[station]
    
    # Add DOY to artificial gaps data
    artificial_gaps_data['doy'] = artificial_gaps_data.index.dayofyear
    
    # Get month info
    month_info = precomputed_data['monthly_data'][month]
    doy = month_info['doy']
    
    # Process each date for gap filling
    time_index = data_window.dropna().index
    
    for d in time_index:
        date_doy = data_window.dropna().loc[d, 'doy']
        
        # Get stations with data for this date
        data_window_allstations = artificial_gaps_data.dropna(axis=1, how='all')
        non_missing_stations = [c for c in data_window_allstations.columns if c != 'doy']
        
        if len(non_missing_stations) == 0:
            continue
        
        data_window_allstations['days_to_date'] = abs((d - data_window_allstations.index).days)
        
        # Check if we have enough target data
        if len(data_window_target) >= min_obs_cdf:
            # Get correlation data
            corr = precomputed_data['corr']
            if date_doy not in corr or station not in corr[date_doy].columns:
                continue
            
            station_corr = corr[date_doy][station].dropna()
            
            # Filter by stations with data and minimum correlation
            valid_corr = station_corr[station_corr.index.isin(non_missing_stations)]
            potential_donors = valid_corr[valid_corr >= min_corr]
            potential_donors = potential_donors[potential_donors.index != station]
            
            if len(potential_donors) > 0:
                # Sort donors by correlation (highest first)
                potential_donors_sorted = potential_donors.sort_values(ascending=False)
                
                # Try each donor station
                for donor_station in potential_donors_sorted.index:
                    # Get donor data from cache if available
                    if donor_station in precomputed_data['station_window_cache']:
                        donor_window_data = precomputed_data['station_window_cache'][donor_station][month]
                        data_window_donor = donor_window_data[donor_station]
                        
                        if len(data_window_donor) >= min_obs_cdf:
                            # Get the closest donor value
                            donor_data_in_window = data_window_allstations[donor_station].dropna()
                            if len(donor_data_in_window) > 0:
                                valid_indices = donor_data_in_window.index
                                closest_idx = data_window_allstations.loc[valid_indices, 'days_to_date'].idxmin()
                                value_donor = data_window_allstations.loc[closest_idx, donor_station]
                                
                                # Perform quantile mapping
                                value_target = quantile_mapping_optimized(
                                    data_window_donor, data_window_target, value_donor, min_obs_cdf
                                )
                                
                                if value_target is not None:
                                    gapfilled_data.loc[d] = value_target
                                    break
    
    # Combine observed & predicted data
    results = gapfilled_data.to_frame(name='pre')
    results['obs'] = original_data[station].loc[dates_to_remove]
    results = results.dropna()
    
    if results.empty:
        return None
    
    # Calculate evaluation metrics
    metrics_dict = {}
    try:
        rmse = np.sqrt(mean_squared_error(results['obs'], results['pre']))
        kge_prime_prime = KGE_Tang2021_optimized(results['obs'].values, results['pre'].values, min_obs_KGE)
        
        metrics_dict['RMSE'] = rmse
        metrics_dict["KGE''"] = kge_prime_prime['KGE']
        metrics_dict["KGE''_corr"] = kge_prime_prime['r']
        metrics_dict["KGE''_bias"] = kge_prime_prime['beta']
        metrics_dict["KGE''_var"] = kge_prime_prime['alpha']
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
    
    return results, metrics_dict

def calculate_stations_doy_corr_optimized(stations_obs, window_days, min_obs_corr):
    """Highly optimized correlation calculation with significant performance improvements."""
    
    stations_doy_corr = {}
    
    # Pre-compute DOY data once
    stations_obs_doy = stations_obs.copy()
    if 'doy' not in stations_obs_doy.columns:
        stations_obs_doy['doy'] = stations_obs_doy.index.dayofyear
    
    # Get unique DOYs and pre-compute all window bounds
    unique_doys = sorted(stations_obs_doy['doy'].unique())
    print(f"Calculating correlations for {len(unique_doys)} unique DOYs...")
    
    # Pre-compute all window bounds to avoid repeated calculations
    window_bounds = {}
    for doy in unique_doys:
        window_start = (doy - window_days) % 366
        window_start = 366 if window_start == 0 else window_start
        window_end = (doy + window_days) % 366
        window_end = 366 if window_end == 0 else window_end
        window_bounds[doy] = (window_start, window_end)
    
    # Convert to numpy arrays for faster operations
    doy_values = stations_obs_doy['doy'].values
    data_columns = [col for col in stations_obs_doy.columns if col != 'doy']
    data_only = stations_obs_doy[data_columns]
    
    # Pre-compute masks for all DOYs to avoid repeated boolean operations
    print("Pre-computing window masks...")
    window_masks = {}
    for doy in unique_doys:
        window_start, window_end = window_bounds[doy]
        if window_start > window_end:
            mask = (doy_values >= window_start) | (doy_values <= window_end)
        else:
            mask = (doy_values >= window_start) & (doy_values <= window_end)
        window_masks[doy] = mask
    
    # Use parallel processing for correlation calculations if beneficial
    n_cores = min(mp.cpu_count(), len(unique_doys))
    use_parallel = len(unique_doys) > 50 and n_cores > 1
    
    if use_parallel:
        print(f"Using parallel processing with {n_cores} cores for correlation calculation...")
        
        # Prepare data for parallel processing
        correlation_tasks = []
        for doy in unique_doys:
            mask = window_masks[doy]
            data_window = data_only.iloc[mask]
            correlation_tasks.append((doy, data_window, min_obs_corr, data_columns))
        
        # Process correlations in parallel
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            with tqdm(total=len(correlation_tasks), desc="Parallel Correlation Calc", 
                      unit="DOYs", ncols=100) as pbar:
                
                # Submit all tasks
                future_to_doy = {
                    executor.submit(_calculate_single_correlation, task): task[0] 
                    for task in correlation_tasks
                }
                
                # Collect results as they complete
                for future in future_to_doy:
                    doy = future_to_doy[future]
                    try:
                        corr_result = future.result()
                        stations_doy_corr[doy] = corr_result
                    except Exception as e:
                        print(f"Error calculating correlation for DOY {doy}: {e}")
                        # Create empty correlation matrix as fallback
                        stations_doy_corr[doy] = pd.DataFrame(
                            index=data_columns, columns=data_columns
                        ).fillna(np.nan)
                    
                    pbar.update(1)
                    pbar.set_postfix({'DOY': doy, 'Workers': n_cores})
    
    else:
        # Sequential processing with optimizations
        with tqdm(total=len(unique_doys), desc="Sequential Correlation Calc", 
                  unit="DOYs", ncols=100) as pbar:
            
            for doy in unique_doys:
                pbar.update(1)
                pbar.set_postfix({
                    'Current DOY': doy,
                    'Window Days': window_days
                })
                
                # Use pre-computed mask
                mask = window_masks[doy]
                data_window = data_only.iloc[mask]
                
                # Calculate correlations with optimizations
                if len(data_window) > min_obs_corr:
                    # Use faster correlation method for large datasets
                    if len(data_window) > 1000:
                        # For large datasets, use numpy-based correlation which is faster
                        corr = _fast_correlation_calculation(data_window, min_obs_corr)
                    else:
                        # For smaller datasets, use pandas correlation
                        corr = data_window.corr(method='spearman', min_periods=min_obs_corr)
                    
                    stations_doy_corr[doy] = corr
                else:
                    # Create empty correlation matrix if insufficient data
                    stations_doy_corr[doy] = pd.DataFrame(
                        index=data_columns, columns=data_columns
                    ).fillna(np.nan)
    
    print(f"Correlation calculation completed for {len(stations_doy_corr)} DOYs")
    return stations_doy_corr

def _calculate_single_correlation(task):
    """Helper function for parallel correlation calculation."""
    doy, data_window, min_obs_corr, data_columns = task
    
    if len(data_window) > min_obs_corr:
        # Use optimized correlation calculation
        if len(data_window) > 1000:
            return _fast_correlation_calculation(data_window, min_obs_corr)
        else:
            return data_window.corr(method='spearman', min_periods=min_obs_corr)
    else:
        # Return empty correlation matrix
        return pd.DataFrame(index=data_columns, columns=data_columns).fillna(np.nan)

def _fast_correlation_calculation(data, min_obs_corr):
    """Fast correlation calculation using numpy for large datasets."""
    try:
        # Convert to numpy array for faster computation
        data_array = data.values
        
        # Remove rows with all NaN values
        valid_rows = ~np.isnan(data_array).all(axis=1)
        if valid_rows.sum() < min_obs_corr:
            return pd.DataFrame(index=data.columns, columns=data.columns).fillna(np.nan)
        
        data_clean = data_array[valid_rows]
        
        # Calculate correlation matrix using numpy
        # For very large datasets, we can use a more memory-efficient approach
        n_cols = data_clean.shape[1]
        corr_matrix = np.full((n_cols, n_cols), np.nan)
        
        for i in range(n_cols):
            for j in range(i, n_cols):
                # Get valid pairs for these two columns
                col_i = data_clean[:, i]
                col_j = data_clean[:, j]
                
                # Find valid pairs (both not NaN)
                valid_pairs = ~(np.isnan(col_i) | np.isnan(col_j))
                
                if valid_pairs.sum() >= min_obs_corr:
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        # Calculate Spearman correlation
                        from scipy.stats import spearmanr
                        corr_val, _ = spearmanr(col_i[valid_pairs], col_j[valid_pairs])
                        corr_matrix[i, j] = corr_val
                        corr_matrix[j, i] = corr_val  # Symmetric matrix
        
        # Convert back to pandas DataFrame
        return pd.DataFrame(corr_matrix, index=data.columns, columns=data.columns)
        
    except Exception as e:
        print(f"Fast correlation calculation failed, falling back to pandas: {e}")
        # Fallback to pandas correlation
        return data.corr(method='spearman', min_periods=min_obs_corr)

def quantile_mapping_optimized(data_donor, data_target, value_donor, min_obs_cdf):
    """Optimized quantile mapping with better performance."""
    
    # Remove duplicates and sort (vectorized operations)
    sorted_data_donor = np.sort(data_donor.drop_duplicates().values)
    sorted_data_target = np.sort(data_target.drop_duplicates().values)
    
    # Check minimum observations requirement
    if len(sorted_data_donor) < min_obs_cdf or len(sorted_data_target) < min_obs_cdf:
        return None
    
    # Find donor value position using vectorized search
    if value_donor in sorted_data_donor:
        rank_donor_obs = np.where(sorted_data_donor == value_donor)[0][0]
    else:
        rank_donor_obs = np.argmin(np.abs(sorted_data_donor - value_donor))
    
    # Calculate cumulative probability
    cumul_prob_donor_obs = (rank_donor_obs + 1) / len(sorted_data_donor)
    
    # Calculate target cumulative probabilities
    cumul_prob_target = np.arange(1, len(sorted_data_target) + 1) / len(sorted_data_target)
    
    # Interpolate to get target value
    value_target = np.interp(cumul_prob_donor_obs, cumul_prob_target, sorted_data_target)
    
    # Ensure non-negative values
    value_target = max(0, round(float(value_target), 2))
    
    return value_target

def KGE_Tang2021_optimized(obs, pre, min_obs_KGE):
    """Optimized KGE calculation with vectorized operations."""
    
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pre))
    obs_clean = obs[valid_mask]
    pre_clean = pre[valid_mask]
    
    if len(obs_clean) < min_obs_KGE:
        return {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
    
    # Vectorized calculations
    pre_mean = np.mean(pre_clean)
    obs_mean = np.mean(obs_clean)
    pre_std = np.std(pre_clean)
    obs_std = np.std(obs_clean)
    
    # Calculate correlation
    if pre_std == 0:
        r = 0
    else:
        r = np.corrcoef(obs_clean, pre_clean)[0, 1]
        if np.isnan(r):
            r = 0
    
    # Calculate alpha and beta
    alpha = pre_std / obs_std if obs_std != 0 else np.nan
    beta = (pre_mean - obs_mean) / obs_std if obs_std != 0 else np.nan
    
    # Calculate KGE
    if np.isnan(alpha) or np.isnan(beta):
        KGE = np.nan
    else:
        KGE = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + beta ** 2)
    
    return {'KGE': KGE, 'r': r, 'alpha': alpha, 'beta': beta}

# Import other functions from original module (unchanged)
def data_availability_monthly_plots_1(SWE_stations, original_SWE_data, gapfilled_SWE_data, flag):
    """Import from original module - unchanged."""
    from .gap_filling_notebook import data_availability_monthly_plots_1 as original_func
    return original_func(SWE_stations, original_SWE_data, gapfilled_SWE_data, flag)

def data_availability_monthly_plots_2(SWE_data):
    """Import from original module - unchanged."""
    from .gap_filling_notebook import data_availability_monthly_plots_2 as original_func
    return original_func(SWE_data)

def qm_gap_filling(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr):
    """Import from original module - unchanged for now."""
    from .gap_filling_notebook import qm_gap_filling as original_func
    return original_func(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr)

def quantile_mapping(data_donor, data_target, value_donor, min_obs_cdf, flag):
    """Import from original module - unchanged."""
    from .gap_filling_notebook import quantile_mapping as original_func
    return original_func(data_donor, data_target, value_donor, min_obs_cdf, flag)

def KGE_Tang2021(obs, pre, min_obs_KGE):
    """Import from original module - unchanged."""
    from .gap_filling_notebook import KGE_Tang2021 as original_func
    return original_func(obs, pre, min_obs_KGE)

def calculate_stations_doy_corr(stations_obs, window_days, min_obs_corr):
    """Import from original module - unchanged."""
    from .gap_filling_notebook import calculate_stations_doy_corr as original_func
    return original_func(stations_obs, window_days, min_obs_corr)
