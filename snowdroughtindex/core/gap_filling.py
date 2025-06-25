"""
Gap filling module for the Snow Drought Index package.

This module contains functions for filling gaps in SWE data using quantile mapping
and evaluating the performance of gap filling methods.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import random
import datetime
from matplotlib.figure import Figure
import psutil
import gc
import matplotlib.patches as mpatches

def calculate_stations_doy_corr(
    stations_obs: pd.DataFrame,
    window_days: int,
    min_obs_corr: int,
    chunk_size: int = 1000
) -> Dict[int, pd.DataFrame]:
    """
    Calculate stations' correlations for each day of the year (doy) with a window centered around the doy.
    Memory-optimized version using chunking and numpy arrays.
    
    Parameters
    ----------
    stations_obs : pandas.DataFrame
        DataFrame of all SWE & precipitation stations observations.
    window_days : int
        Number of days to select data for around a certain doy, to calculate correlations.
    min_obs_corr : int
        Minimum number of overlapping observations required to calculate the correlation between 2 stations.
    chunk_size : int, optional
        Number of stations to process in each chunk. Default is 1000.
        
    Returns
    -------
    dict
        Dictionary containing a pandas.DataFrame of stations correlations for each day of year.
    """
    # Convert DataFrame to numpy array for memory efficiency
    stations_data = stations_obs.to_numpy(dtype=np.float32)
    station_names = stations_obs.columns
    dates = stations_obs.index
    
    # Calculate day of year for each date
    doys = np.array([d.dayofyear for d in dates], dtype=np.int16)
    
    # Initialize output dictionary
    stations_doy_corr = {}
    
    # Process stations in chunks to manage memory
    n_stations = len(station_names)
    for chunk_start in range(0, n_stations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_stations)
        chunk_stations = station_names[chunk_start:chunk_end]
        chunk_data = stations_data[:, chunk_start:chunk_end]
        
        # Process each day of year
        for doy in range(1, 367):
            # Calculate window indices
            window_start = (doy - window_days) % 366
            window_end = (doy + window_days) % 366
            
            # Handle year boundary cases
            if window_start > window_end:
                mask = (doys >= window_start) | (doys <= window_end)
            else:
                mask = (doys >= window_start) & (doys <= window_end)
            
            # Get data for this window
            window_data = chunk_data[mask]
            
            # Calculate correlations
            corr_matrix = np.corrcoef(window_data, rowvar=False)
            
            # Create DataFrame for this chunk
            chunk_corr = pd.DataFrame(
                corr_matrix,
                index=chunk_stations,
                columns=chunk_stations,
                dtype=np.float32
            )
            
            # Apply minimum observations filter
            valid_pairs = np.sum(~np.isnan(window_data), axis=0) >= min_obs_corr
            chunk_corr = chunk_corr.loc[valid_pairs, valid_pairs]
            
            # Update or create correlation DataFrame for this doy
            if doy in stations_doy_corr:
                stations_doy_corr[doy] = pd.concat([
                    stations_doy_corr[doy],
                    chunk_corr
                ], axis=1)
            else:
                stations_doy_corr[doy] = chunk_corr
        
        # Free memory
        del chunk_data, chunk_corr
        gc.collect()
        
        # Monitor memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory usage after chunk {chunk_start//chunk_size + 1}: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    return stations_doy_corr

def quantile_mapping(
    data_donor: pd.Series,
    data_target: pd.Series,
    value_donor: float,
    min_obs_cdf: int,
    flag: int = 0
) -> Optional[float]:
    """
    Calculate target station's gap filling value from donor station's value using quantile mapping.
    
    Parameters
    ----------
    data_donor : pandas.DataFrame
        DataFrame of donor station observations used to build empirical cdf.
    data_target : pandas.DataFrame
        DataFrame of target station observations used to build empirical cdf.
    value_donor : float
        Donor station value used in the quantile mapping.
    min_obs_cdf : int
        Minimum number of unique observations required to calculate a station's cdf.
    flag : int, optional
        Flag to plot (1) or not (0) the donor and target stations' cdfs, by default 0.
        
    Returns
    -------
    float or None
        Target station value calculated using quantile mapping, or None if there are not enough observations.
    """
    # Build the donor station's empirical cdf
    sorted_data_donor = data_donor.drop_duplicates().sort_values(ignore_index=True)

    # Build the target station's empirical cdf
    sorted_data_target = data_target.drop_duplicates().sort_values(ignore_index=True)

    # Calculate the donor & target stations' cdfs if they both have at least min_obs_cdf unique observations
    if (len(sorted_data_donor) >= min_obs_cdf) & (len(sorted_data_target) >= min_obs_cdf):
        # Calculate the cumulative probability corresponding to the donor value
        rank_donor_obs = sorted_data_donor[sorted_data_donor == value_donor].index[0]
        total_obs_donor = len(sorted_data_donor)
        cumul_prob_donor_obs = (rank_donor_obs + 1) / total_obs_donor

        # Calculate the cumulative probability corresponding to the target value
        cumul_prob_target = np.arange(1, len(sorted_data_target)+1) / (len(sorted_data_target))

        # Inter-/extrapolate linearly to get the target value corresponding to the donor station's cumulative probability
        inverted_edf = interp1d(cumul_prob_target, sorted_data_target, fill_value="extrapolate")
        value_target = round(float(inverted_edf(cumul_prob_donor_obs)), 2)

        # Set any potential negative values from interpolation/extrapolation to zero
        if value_target < 0:
            value_target = 0

        # If requested, plot the target & donor stations' cdfs
        if flag == 1:
            plt.figure()
            plt.plot(sorted_data_donor, np.arange(1, len(sorted_data_donor)+1) / (len(sorted_data_donor)), label='donor')
            plt.plot(sorted_data_target, cumul_prob_target, label='target')
            plt.scatter(value_donor, cumul_prob_donor_obs)
            plt.legend()

        return value_target

    # If either/both the target & donor stations have < min_obs_cdf observations do nothing
    else:
        return None

def qm_gap_filling(
    original_data: pd.DataFrame,
    window_days: int,
    min_obs_corr: int,
    min_obs_cdf: int,
    min_corr: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform gap filling for all missing observations (when possible) using quantile mapping.
    
    For each target station and each date for which data is missing, we identify a donor station as the station with:
    - data for this date,
    - a cdf for this doy,
    - and the best correlation to the target station (correlation >= min_corr for this doy).
    
    Parameters
    ----------
    original_data : pandas.DataFrame
        DataFrame of original stations' observations dataset, which will be gap filled.
    window_days : int
        Number of days to select data for around a certain doy, to calculate correlations.
    min_obs_corr : int
        Minimum number of overlapping observations required to calculate the correlation between 2 stations.
    min_obs_cdf : int
        Minimum number of stations required to calculate a station's cdf.
    min_corr : float
        Minimum correlation value required to keep a donor station.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame of gap filled stations' observations.
    pandas.DataFrame
        DataFrame with information about the type of data (estimates or observations) in the gap filled dataset.
    pandas.DataFrame
        DataFrame with information about the donor station used to fill each of the gaps.
    """
    # Create a duplicate of the dataset to gap fill
    gapfilled_data = original_data.copy()

    # Remove P & external SWE stations (buffer) from the dataframe
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Keep only gap filled SWE stations (without P stations & external SWE stations)
    gapfilled_data = gapfilled_data[cols]

    # Add doy to the Pandas DataFrame
    original_data['doy'] = original_data.index.dayofyear

    # Set empty dataframes to keep track of data type and donor station ids
    data_type_flags = pd.DataFrame(data=0, index=original_data.index, columns=cols)
    donor_stationIDs = pd.DataFrame(data=np.nan, index=original_data.index, columns=cols)

    # Calculate correlations between stations that have overlapping observations
    corr = calculate_stations_doy_corr(original_data, window_days, min_obs_corr)

    # Identify dates for gap filling
    time_index = original_data.index

    # Loop over dates for gap filling
    for d in time_index:
        # Calculate the doy corresponding to the date
        # Note: doy 365 and 366 are bundled together
        doy = original_data.loc[d, 'doy']

        # Calculate the start and end dates of the time window for the gap filling steps
        window_startdate = d - pd.Timedelta(days=window_days)
        window_enddate = d + pd.Timedelta(days=window_days)

        # Get IDs of all stations with data for this date (and within time window)
        data_window = original_data[window_startdate:window_enddate].dropna(axis=1, how='all')
        non_missing_stations = [c for c in data_window.columns if 'doy' not in c]
        data_window['days_to_date'] = abs((d - data_window.index).days)

        # Calculate the start & end doys of the time window for quantile mapping, with special rules around the start & end of the calendar year
        window_startdoy = (data_window['doy'].iloc[0]) % 366
        window_startdoy = 366 if window_startdoy == 0 else window_startdoy
        window_enddoy = (data_window['doy'].iloc[-1]) % 366
        window_enddoy = 366 if window_enddoy == 0 else window_enddoy

        # Loop over stations to gap fill
        for target_station in cols:
            # If station has no data, proceed with the gap filling
            if np.isnan(original_data.loc[d, target_station]):
                # Select target data within time window for this doy from all years
                if window_startdoy > window_enddoy:
                    data_window_target = original_data[target_station].dropna()[
                        (original_data['doy'] >= window_startdoy) | (original_data['doy'] <= window_enddoy)]
                else:
                    data_window_target = original_data[target_station].dropna()[
                        (original_data['doy'] >= window_startdoy) & (original_data['doy'] <= window_enddoy)]

                # We can continue if there are enough target data to build cdf
                if len(data_window_target.index) >= min_obs_cdf:
                    # Get ids of all stations with correlations >= a minimum correlation for this doy, not including the target station itself
                    non_missing_corr = corr[doy][target_station].dropna()
                    non_missing_corr = non_missing_corr[non_missing_corr.index.isin(non_missing_stations)]
                    potential_donor_stations = non_missing_corr[non_missing_corr >= min_corr].index.values
                    potential_donor_stations = [c for c in potential_donor_stations if target_station not in c]

                    # If there is at least one potential donor station, proceed
                    if len(potential_donor_stations) > 0:
                        # Sort the donor stations from highest to lowest value
                        potential_donor_stations_sorted = corr[doy].loc[potential_donor_stations, target_station].dropna().sort_values(
                            ascending=False).index.values

                        # Loop over sorted donor stations until I find one with enough data to build a cdf
                        for donor_station in potential_donor_stations_sorted:
                            # Select data within time window for this doy from all years
                            if window_startdoy > window_enddoy:
                                data_window_donor = original_data[donor_station].dropna()[
                                    (original_data['doy'] >= window_startdoy) | (original_data['doy'] <= window_enddoy)]
                            else:
                                data_window_donor = original_data[donor_station].dropna()[
                                    (original_data['doy'] >= window_startdoy) & (original_data['doy'] <= window_enddoy)]

                            # We can continue if there are enough donor data to build cdf
                            if len(data_window_donor.index) >= min_obs_cdf:
                                # If the donor station has multiple values within the window, we keep the closest donor station value to the date we are gap filling
                                sorted_data_window = data_window.sort_values(by=['days_to_date'])
                                value_donor = sorted_data_window[donor_station].dropna()[0]

                                # Perform the gap filling using quantile mapping
                                value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                if value_target is not None:
                                    gapfilled_data.loc[d, target_station] = value_target
                                    data_type_flags.loc[d, target_station] = 1
                                    donor_stationIDs.loc[d, target_station] = donor_station

                                break

                            else:
                                continue

    return gapfilled_data, data_type_flags, donor_stationIDs

def KGE_Tang2021(
    obs: np.ndarray,
    pre: np.ndarray,
    min_obs_KGE: int
) -> Dict[str, float]:
    """
    Calculate the modified Kling-Gupta Efficiency (KGE") and its 3 components.
    
    The KGE measures the correlation, bias and variability of the simulated values against the observed values.
    KGE" was proposed by Tang et al. (2021) to solve issues arising with 0 values in the KGE or KGE'.
    
    Parameters
    ----------
    obs : numpy.ndarray
        Array of observations to evaluate.
    pre : numpy.ndarray
        Array of predictions/simulations to evaluate.
    min_obs_KGE : int
        Minimum number of stations required to calculate a station's cdf.
        
    Returns
    -------
    dict
        Dictionary containing the final KGE'' value as well as all elements of the KGE''.
    """
    ind_nan = np.isnan(obs) | np.isnan(pre)
    obs = obs[~ind_nan]
    pre = pre[~ind_nan]

    if len(obs) >= min_obs_KGE:
        pre_mean = np.mean(pre, axis=0, dtype=np.float64)
        obs_mean = np.mean(obs, axis=0, dtype=np.float64)
        pre_std = np.std(pre, axis=0)
        obs_std = np.std(obs, dtype=np.float64)

        # Check to see if all forecast values are the same. If they are r cannot be calculated and is set to 0
        if pre_std == 0:
            r = 0
        else:
            r = np.sum((pre - pre_mean) * (obs - obs_mean), axis=0, dtype=np.float64) / \
                np.sqrt(np.sum((pre - pre_mean) ** 2, axis=0, dtype=np.float64) *
                        np.sum((obs - obs_mean) ** 2, dtype=np.float64))

        alpha = pre_std / obs_std
        beta = (pre_mean - obs_mean) / obs_std
        KGE = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta) ** 2)

        KGEgroup = {'KGE': KGE, 'r': r, 'alpha': alpha, 'beta': beta}
    else:
        KGEgroup = {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}

    return KGEgroup

def artificial_gap_filling(
    original_data: pd.DataFrame,
    iterations: int,
    artificial_gap_perc: float,
    window_days: int,
    min_obs_corr: int,
    min_obs_cdf: int,
    min_corr: float,
    min_obs_KGE: int,
    flag: int = 0
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Figure]]:
    """
    Create random artificial gaps in the original dataset for each month & station, and run the gap filling function to assess its performance.
    
    Parameters
    ----------
    original_data : pandas.DataFrame
        DataFrame of original stations' observations dataset, to which data will be removed for artificial gap filling.
    iterations : int
        Number of times to repeat the artificial gap filling (we remove data at random each time in the original dataset).
    artificial_gap_perc : float
        Percentage between 1 and 100 for the amount of data to remove for each station & month's first day.
    window_days : int
        Number of days to select data for around a certain doy, to calculate correlations.
    min_obs_corr : int
        Minimum number of overlapping observations required to calculate the correlation between 2 stations.
    min_obs_cdf : int
        Minimum number of stations required to calculate a station's cdf.
    min_corr : float
        Minimum correlation value required to keep a donor station.
    min_obs_KGE : int
        Minimum number of stations required to calculate a station's cdf.
    flag : int, optional
        Flag to plot the gap filled values vs the observed values (1) or not (0), by default 0.
        
    Returns
    -------
    dict
        Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration.
    matplotlib.figure.Figure, optional
        A figure of the gap filled vs. the actual SWE observations for each first day of the month, if flag=1.
    """
    # Suppresses the "SettingWithCopyWarning"
    pd.set_option("mode.chained_assignment", None)

    # Set up the figure
    if flag == 1:
        ncols = 3
        fig, axs = plt.subplots(4, ncols, sharex=False, sharey=False, figsize=(8, 10))
        plot_col = -1
        row = 0

    # Identify stations for gap filling (without P & external SWE stations (buffer) as we don't do any gap filling for these)
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Create an empty dictionary to store the metric values for each month, station & iteration
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan

    # Calculate correlations between stations that have overlapping observations
    corr = calculate_stations_doy_corr(original_data, window_days, min_obs_corr)

    # Loop over months
    for mo in range(1, 12+1):
        # Controls for plotting on right subplot
        if flag == 1:
            plot_col += 1
            if plot_col == ncols:
                row += 1
                plot_col = 0

        # Loop over iterations
        for i in range(iterations):
            # Initialize counter to assign results to the right station
            elem = -1

            # Looping over stations
            for s in cols:
                # Update counter to assign results to the right station
                elem += 1

                # Duplicate original data to create artificial gaps from this
                artificial_gaps_data = original_data.copy()

                # Remove all missing values for a given station for which to perform gap filling
                station_nomissing_values = pd.DataFrame(artificial_gaps_data[s].dropna())

                # Add DOY to select data to gap fill within a time window around first day of month
                station_nomissing_values['doy'] = station_nomissing_values.index.dayofyear

                # Calculate the doy corresponding to the date - using 2010 as common year (not leap year)
                doy = int(datetime.datetime(2010, mo, 1).strftime('%j'))

                # Calculate the start & end doys of the time window for quantile mapping, with caution around the start & end of the calendar year
                window_startdoy = (doy-window_days) % 365
                window_startdoy = 365 if window_startdoy == 0 else window_startdoy
                window_enddoy = (doy+window_days) % 365
                window_enddoy = 366 if window_enddoy == 0 or window_enddoy == 365 else window_enddoy

                # Select data within time window
                if window_startdoy > window_enddoy:
                    data_window = station_nomissing_values[(station_nomissing_values['doy'] >= window_startdoy) | (
                        station_nomissing_values['doy'] <= window_enddoy)]
                else:
                    data_window = station_nomissing_values[(station_nomissing_values['doy'] >= window_startdoy) & (
                        station_nomissing_values['doy'] <= window_enddoy)]

                # Select target data within this time window
                data_window_target = data_window[s]

                # Calculate the number of observations to remove for this station & month's first day
                n = int(len(data_window.index) * artificial_gap_perc / 100)

                # If the number of observations is above zero we can proceed with the gap filling
                if n > 0:
                    # Randomly select n dates from the station's data (no duplicates) and remove them from the original dataset - if 100% is removed then all dates will be selected
                    if artificial_gap_perc == 100:
                        dates_to_remove = data_window.index
                    else:
                        dates_to_remove = data_window.index[random.sample(range(0, len(data_window.index)), n)]
                    artificial_gaps_data[s].loc[dates_to_remove] = np.nan
                    artificial_gaps_data = artificial_gaps_data.loc[dates_to_remove]

                    # Keep only SWE station to gap fill
                    gapfilled_data = artificial_gaps_data[s].copy()

                    # Identify dates for gap filling
                    time_index = data_window.dropna().index

                    # Loop over dates for gap filling
                    for d in time_index:
                        # Get the doy corresponding to the date
                        doy = data_window.dropna().loc[d, 'doy']

                        # Get IDs of all stations with data for this date (and within time window)
                        data_window_allstations = artificial_gaps_data.dropna(axis=1, how='all')
                        non_missing_stations = [c for c in data_window_allstations.columns]
                        data_window_allstations['days_to_date'] = abs((d - data_window_allstations.index).days)

                        # We can continue if there are enough target data to build cdf
                        if len(data_window_target.index) >= min_obs_cdf:
                            # Get ids of all stations with correlations >= a minimum correlation for this doy, not including the target station itself
                            non_missing_corr = corr[doy][s].dropna()
                            non_missing_corr = non_missing_corr[non_missing_corr.index.isin(non_missing_stations)]
                            potential_donor_stations = non_missing_corr[non_missing_corr >= min_corr].index.values
                            potential_donor_stations = [c for c in potential_donor_stations if s not in c]

                            # If there is at least one potential donor station, proceed
                            if len(potential_donor_stations) > 0:
                                # Sort the donor stations from highest to lowest value
                                potential_donor_stations_sorted = corr[doy].loc[potential_donor_stations, s].dropna().sort_values(
                                    ascending=False).index.values

                                # Loop over sorted donor stations until I find one with enough data to build a cdf
                                for donor_station in potential_donor_stations_sorted:
                                    # Select data within time window for this doy from all years
                                    data_window_donor = data_window_allstations[donor_station].dropna()

                                    # We can continue if there are enough donor data to build cdf
                                    if len(data_window_donor.index) >= min_obs_cdf:
                                        # If the donor station has multiple values within the window, we keep the closest donor station value to the date we are gap filling
                                        sorted_data_window = data_window_allstations.sort_values(by=['days_to_date'])
                                        value_donor = sorted_data_window[donor_station].dropna()[0]

                                        # Perform the gap filling using quantile mapping
                                        value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                        if value_target is not None:
                                            gapfilled_data.loc[d] = value_target

                                        break

                                    else:
                                        continue

                    # Combine observed & predicted data into a single Pandas dataframe
                    results = gapfilled_data.to_frame(name='pre')
                    results['obs'] = original_data[s].loc[dates_to_remove]
                    results = results.dropna()

                    # Plot the gap filled vs the observed values
                    if flag == 1:
                        axs[row, plot_col].scatter(results['obs'], results['pre'], color='b', alpha=.3)
                        axs[row, plot_col].set_title('month'+str(mo))
                        if row == 3 and plot_col == 0:
                            axs[row, plot_col].set_xlabel('observed')
                            axs[row, plot_col].set_ylabel('infilling')

                    # If there are no predicted values set the metrics to nan
                    if results.empty:
                        for m in metrics:
                            evaluation[m][mo-1, elem, i] = np.nan

                    # Otherwise proceed with evaluating the gap filling performance
                    else:
                        rmse = mean_squared_error(results['obs'], results['pre'], squared=False)
                        kge_prime_prime = KGE_Tang2021(results['obs'].values, results['pre'].values, min_obs_KGE)
                        evaluation['RMSE'][mo-1, elem, i] = rmse
                        evaluation["KGE''"][mo-1, elem, i] = kge_prime_prime['KGE']
                        evaluation["KGE''_corr"][mo-1, elem, i] = kge_prime_prime['r']
                        evaluation["KGE''_bias"][mo-1, elem, i] = kge_prime_prime['beta']
                        evaluation["KGE''_var"][mo-1, elem, i] = kge_prime_prime['alpha']

                # Else if the number of observations is zero we go to the next station
                else:
                    continue

    if flag == 1:
        plt.tight_layout()
        return evaluation, fig
    else:
        return evaluation

def plots_artificial_gap_evaluation(
    evaluation_scores: Dict[str, np.ndarray]
) -> Figure:
    """
    Plot evaluation results for the artificial gap filling.
    
    Parameters
    ----------
    evaluation_scores : dict
        Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure of the various evaluation metrics for all stations, iterations & each month's first day.
    """
    # Initialize figure
    ncols = 3
    fig, axs = plt.subplots(2, ncols, sharex=True, sharey=False, figsize=(9, 5))
    elem = -1
    row = 0

    # Define metrics used & their optimal values
    metrics = list(evaluation_scores.keys())
    metrics_optimal_values = {'RMSE': 0, "KGE''": 1, "KGE''_corr": 1, "KGE''_bias": 0, "KGE''_var": 1}
    units = {'RMSE': 'mm', "KGE''": '-', "KGE''_corr": '-', "KGE''_bias": '-', "KGE''_var": '-'}

    # Loop over metrics
    for m in metrics:
        # Controls for plotting on right subplot
        elem += 1
        if elem == ncols:
            row += 1
            elem = 0

        # Loop over iterations
        for i in range(evaluation_scores[m].shape[2]):
            # Plot boxplot for each month
            for mo in range(1, 12+1):
                nonan = evaluation_scores[m][mo-1, :, i][~np.isnan(evaluation_scores[m][mo-1, :, i])]
                bp = axs[row, elem].boxplot(nonan, positions=[mo], patch_artist=True, showfliers=False, widths=.7)
                plt.setp(bp['boxes'], color='b', alpha=.5)
                plt.setp(bp['whiskers'], color='b')
                plt.setp(bp['medians'], color='k')

        # Add elements to the plot
        axs[row, elem].plot(np.arange(0, 13+1), [metrics_optimal_values[m]]*14, color='grey', ls='--', label='best values')
        axs[row, elem].set_xlim([0, 13])
        axs[row, elem].set_xticks(np.arange(1, 12+1))
        axs[row, elem].set_ylabel(m+' ['+units[m]+']', fontweight='bold')
        axs[row, elem].tick_params(axis='y', labelsize=8)

        if row == 1:
            axs[row, elem].set_xticklabels(np.arange(1, 12+1), fontsize=8)

    axs[1, 0].legend(fontsize=8)
    axs[1, 0].set_xlabel('months (1st of)', fontweight='bold')
    fig.delaxes(axs[1][2])
    plt.tight_layout()

    return fig

###

def data_availability_monthly_plots_1(SWE_stations, original_SWE_data, gapfilled_SWE_data, flag):

    """Calculating and plotting the % of SWE stations available on the first day of each month of each year.

    Keyword arguments:
    ------------------
    - SWE_stations: Pandas GeoDataFrame of all SWE stations
    - original_SWE_data: xarray DataArray of the original SWE observations
    - gapfilled_SWE_data: xarray DataArray of the SWE observations after gap filling
    - flag: Flag to indicate if gap filled data was provided (1) or not (0). In the case that it is provided, a comparison plot will be made to compare data availability in the original data vs the gap filled data

    Returns:
    --------
    - Bar chart timeseries of SWE stations available on the first day of each month of each year

    """

    # Initialize plot
    fig, axs = plt.subplots(6, 2, sharex=True, sharey=True, figsize=(14,8))
    elem = -1
    column = 0

    # Loop over months
    for m in range(1,12+1):

        # controls for plotting on right subplot (i.e., month)
        elem += 1
        if elem == 6:
            column += 1
            elem = 0

        # for SWE data with gap filling
        if flag == 1:

            # extract data on the first of the month m
            data_month_gapfilled = gapfilled_SWE_data.sel(station_id=SWE_stations.station_id.values, time=( (gapfilled_SWE_data['time.month'] == m) & (gapfilled_SWE_data['time.day'] == 1) ))

            # count the % of stations with data on those dates
            data_month_gapfilled_count = data_month_gapfilled.count(dim='station_id') / len(SWE_stations) * 100

            # plot bar chart of available data
            axs[elem,column].bar(data_month_gapfilled_count['time.year'], data_month_gapfilled_count.data, color='r', alpha=.5)

        # same process as above but for original SWE data
        data_month = original_SWE_data.sel(station_id=SWE_stations.station_id.values, time=( (original_SWE_data['time.month'] == m) & (original_SWE_data['time.day'] == 1) ))
        data_month_count = data_month.count(dim='station_id') / len(SWE_stations) * 100
        axs[elem,column].bar(data_month_count['time.year'], data_month_count.data, color='b')

        # add plot labels
        if elem == 5 and column == 0:
            axs[elem,column].set_ylabel('% of SWE stations \n with data in basin')
        month_name = datetime.datetime.strptime(str(m), "%m").strftime("%b")
        axs[elem,column].set_title('1st '+month_name, fontweight='bold')

        if flag == 1:
            bluepatch = mpatches.Patch(color='b', label='original data')
            redpatch = mpatches.Patch(color='r', alpha=.5, label='after gap filling')
            plt.legend(handles=[bluepatch, redpatch])

    plt.tight_layout()

    return fig

###

def data_availability_monthly_plots_2(SWE_data):

    """Creating bar chart subplots of the days with SWE observations around the 1st day of each month.

    Keyword arguments:
    ------------------
    - SWE_data: Pandas DataFrame containing the SWE stations observations

    Returns:
    --------
    - Bar chart subplots of the days with SWE observations around the 1st day of each month

    """

    # Initialize plot
    fig, axs = plt.subplots(6, 2, sharex=False, sharey=True, figsize=(8,16))
    elem = -1
    column = 0

    # Add day of year (doy) to test basin SWE observations Pandas DataFrame
    SWE_data_with_doy = SWE_data.copy()
    SWE_data_with_doy['doy'] = SWE_data_with_doy.index.dayofyear

    # Remove automatic stations as they distract the analysis
    manual_stations = [s for s in SWE_data_with_doy.columns if s[-1] != 'P']
    SWE_data_with_doy_manual = SWE_data_with_doy[manual_stations]

    # Define the doys of 1st of each month
    doys_first_month = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    # Loop over months
    for m in range(1,12+1):

        # controls for plotting on right subplot
        elem += 1
        if elem == 6:
            column += 1
            elem = 0

        # calculate the start & end of the data selection window, with caution around the start & end of the calendar year
        window_start = (doys_first_month[m-1]-15)%366
        if window_start == 0:
            window_start = 365
        window_end = (doys_first_month[m-1]+15)%366
        if window_end == 0 or window_end == 365:
            window_end = 366

        # select SWE observations within window
        if window_start > window_end:
            data_window = SWE_data_with_doy_manual[(SWE_data_with_doy_manual['doy']>=window_start) | (SWE_data_with_doy_manual['doy'] <= window_end)]
        else:
            data_window = SWE_data_with_doy_manual[(SWE_data_with_doy_manual['doy']>=window_start) & (SWE_data_with_doy_manual['doy'] <= window_end)]

        # drop dates or stations with no data at all
        data_window = data_window.dropna(axis=0, how='all')
        data_window = data_window.dropna(axis=1, how='all')

        # count total number of stations with data on each doy
        stations_cols = [c for c in data_window.columns if 'doy' not in c]
        data_stations_window = data_window[stations_cols]
        data_count_window = data_stations_window.count(axis=1)

        # create xticks to plot the data for each doy
        if window_start > window_end:
            xticks = list(np.arange(window_start,365+1))+list(np.arange(1,window_end+1))
        else:
            xticks = list(np.arange(window_start,window_end+1))
        xticks_plot = np.arange(len(xticks))

        # save the data for the right doy
        data_count_plot = [0]*len(xticks)
        for x in range(len(data_window.index)):
            doy = data_window.iloc[x]['doy']
            if doy == 366:
                doy = 365
            data_count_plot[xticks.index(doy)] += data_count_window.iloc[x]

        # plot data
        axs[elem,column].bar(xticks_plot, data_count_plot, color='b')
        axs[elem,column].set_xticks([xticks_plot[0],xticks_plot[15],xticks_plot[-1]])
        axs[elem,column].set_xticklabels([xticks[0],doys_first_month[m-1],xticks[-1]])

        # add plot labels
        if elem == 5 and column == 0:
            axs[elem,column].set_ylabel('# of SWE obs.')
            axs[elem,column].set_xlabel('DOY')

        if elem == 5 and column == 1:
            axs[elem,column].set_xlabel('DOY')

        month_name = datetime.datetime.strptime(str(m), "%m").strftime("%b")
        axs[elem,column].set_title('1st '+month_name+' +/- 15 days', fontweight='bold')

    plt.tight_layout()

    return fig

###
