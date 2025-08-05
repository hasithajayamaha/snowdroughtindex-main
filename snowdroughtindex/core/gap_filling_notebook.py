"""
Gap filling module extracted from notebook logic for the Snow Drought Index package.

This module contains functions for filling gaps in SWE data using quantile mapping
and evaluating the performance of gap filling methods. Logic prioritizes the notebook's implementation.
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

def artificial_gap_filling(original_data, iterations, artificial_gap_perc, window_days, min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag):
    """Creating random artificial gaps in the original dataset for each month & station, and running the gap filling function to assess its performance.

    Keyword arguments:
    ------------------
    - original_data: Pandas DataFrame of original stations' observations dataset, to which data will be removed for artificial gap filling
    - iterations: Positive integer denoting the number of times we want to repeat the artificial gap filling (we remove data at random each time in the original dataset)
    - artificial_gap_perc: Percentage between 1 and 100 for the amount of data to remove for each station & month's first day
    - window_days: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
    - min_obs_corr: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations
    - min_obs_cdf: Positive integer for the minimum number of stations required to calculate a station's cdf
    - min_corr: Value between 0 and 1 for the minimum correlation value required to keep a donor station
    - min_obs_KGE: Positive integer for the minimum number of stations required to calculate a station's cdf
    - flag: Integer to plot the gap filled values vs the observed values (1) or not (0)

    Returns:
    --------
    - evaluation: Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration
    - fig (optional): A figure of the gap filled vs. the actual SWE observations for each first day of the month
    """
    # suppresses the "SettingWithCopyWarning"
    pd.set_option("mode.chained_assignment", None)

    # Set up the figure
    if flag == 1:
        ncols = 3
        fig, axs = plt.subplots(4, ncols, sharex=False, sharey=False, figsize=(8,10))
        plot_col = -1
        row = 0

    # Identify stations for gap filling (without P & external SWE stations (buffer) as we don't do any gap filling for these)
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Create an empty dictionary to store the metric values for each month, station & iteration
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan

    print("Calculating correlations for artificial gap filling...")
    # Calculate correlations between stations that have overlapping observations (only once)
    original_data_with_doy = original_data.copy()
    original_data_with_doy['doy'] = original_data_with_doy.index.dayofyear
    corr = calculate_stations_doy_corr(original_data_with_doy, window_days, min_obs_corr)

    # Pre-compute monthly DOY values and windows
    monthly_doys = {}
    monthly_windows = {}
    for mo in range(1, 13):
        doy = int(datetime.datetime(2010, mo, 1).strftime('%j'))
        window_startdoy = (doy - window_days) % 365
        window_startdoy = 365 if window_startdoy == 0 else window_startdoy
        window_enddoy = (doy + window_days) % 365
        window_enddoy = 366 if window_enddoy == 0 or window_enddoy == 365 else window_enddoy
        monthly_doys[mo] = doy
        monthly_windows[mo] = (window_startdoy, window_enddoy)

    # Pre-compute station data with DOY for each station
    station_data_cache = {}
    for s in cols:
        station_data = original_data[s].dropna()
        if len(station_data) > 0:
            station_df = pd.DataFrame({s: station_data})
            station_df['doy'] = station_df.index.dayofyear
            station_data_cache[s] = station_df

    print(f"Processing {12 * len(cols) * iterations} combinations...")
    total_combinations = 12 * len(cols) * iterations
    processed = 0

    # loop over months
    for mo in range(1, 13):
        doy = monthly_doys[mo]
        window_startdoy, window_enddoy = monthly_windows[mo]

        # controls for plotting on right subplot
        if flag == 1:
            plot_col += 1
            if plot_col == ncols:
                row += 1
                plot_col = 0

        # loop over iterations
        for i in range(iterations):
            # loop over stations
            for elem, s in enumerate(cols):
                processed += 1
                if processed % 50 == 0:  # Progress indicator
                    print(f"Progress: {processed}/{total_combinations} ({100*processed/total_combinations:.1f}%)")

                # Skip if station has no data
                if s not in station_data_cache:
                    continue

                station_nomissing_values = station_data_cache[s].copy()

                # select data within time window
                if window_startdoy > window_enddoy:
                    data_window = station_nomissing_values[
                        (station_nomissing_values['doy'] >= window_startdoy) | 
                        (station_nomissing_values['doy'] <= window_enddoy)
                    ]
                else:
                    data_window = station_nomissing_values[
                        (station_nomissing_values['doy'] >= window_startdoy) & 
                        (station_nomissing_values['doy'] <= window_enddoy)
                    ]

                # Select target data within this time window
                data_window_target = data_window[s]

                # calculate the number of observations to remove for this station & month's first day
                n = int(len(data_window.index) * artificial_gap_perc / 100)

                # if the number of observations is above zero we can proceed with the gap filling
                if n > 0:
                    # Create artificial gaps data for this specific test
                    artificial_gaps_data = original_data.copy()

                    # randomly select n dates from the station's data (no duplicates) and remove them from the original dataset
                    if artificial_gap_perc == 100:
                        dates_to_remove = data_window.index
                    else:
                        dates_to_remove = data_window.index[random.sample(range(0, len(data_window.index)), n)]
                    
                    artificial_gaps_data.loc[dates_to_remove, s] = np.nan
                    artificial_gaps_data = artificial_gaps_data.loc[dates_to_remove]

                    # Keep only SWE station to gap fill
                    gapfilled_data = artificial_gaps_data[s].copy()

                    # Identify dates for gap filling
                    time_index = data_window.dropna().index

                    # Pre-compute days_to_date for all dates in the artificial gaps data
                    artificial_gaps_data['doy'] = artificial_gaps_data.index.dayofyear

                    # Loop over dates for gap filling
                    for d in time_index:
                        # Get the doy corresponding to the date
                        date_doy = data_window.dropna().loc[d, 'doy']

                        # Get IDs of all stations with data for this date (and within time window)
                        data_window_allstations = artificial_gaps_data.dropna(axis=1, how='all')
                        non_missing_stations = [c for c in data_window_allstations.columns if c != 'doy']
                        
                        if len(non_missing_stations) == 0:
                            continue

                        data_window_allstations['days_to_date'] = abs((d - data_window_allstations.index).days)

                        # We can continue if there are enough target data to build cdf
                        if len(data_window_target.index) >= min_obs_cdf:
                            # Get correlation data for this DOY and station
                            if date_doy not in corr or s not in corr[date_doy].columns:
                                continue

                            station_corr = corr[date_doy][s].dropna()
                            
                            # Filter by stations with data and minimum correlation
                            valid_corr = station_corr[station_corr.index.isin(non_missing_stations)]
                            potential_donors = valid_corr[valid_corr >= min_corr]
                            potential_donors = potential_donors[potential_donors.index != s]

                            if len(potential_donors) > 0:
                                # Sort donors by correlation (highest first)
                                potential_donors_sorted = potential_donors.sort_values(ascending=False)

                                # Try each donor station
                                for donor_station in potential_donors_sorted.index:
                                    # Select data within time window for this doy from all years
                                    data_window_donor = data_window_allstations[donor_station].dropna()

                                    # We can continue if there are enough donor data to build cdf
                                    if len(data_window_donor.index) >= min_obs_cdf:
                                        # Get the closest donor value
                                        closest_idx = data_window_allstations.loc[data_window_donor.index, 'days_to_date'].idxmin()
                                        value_donor = data_window_allstations.loc[closest_idx, donor_station]

                                        # Perform the gap filling using quantile mapping
                                        value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                        if value_target is not None:
                                            gapfilled_data.loc[d] = value_target
                                            break

                    # combine observed & predicted data into a single Pandas dataframe
                    results = gapfilled_data.to_frame(name='pre')
                    results['obs'] = original_data[s].loc[dates_to_remove]
                    results = results.dropna()

                    # plot the gap filled vs the observed values
                    if flag == 1 and not results.empty:
                        axs[row, plot_col].scatter(results['obs'], results['pre'], color='b', alpha=.3)
                        axs[row, plot_col].set_title('month' + str(mo))
                        if row == 3 and plot_col == 0:
                            axs[row, plot_col].set_xlabel('observed')
                            axs[row, plot_col].set_ylabel('infilling')

                    # if there are no predicted values set the metrics to nan
                    if results.empty:
                        for m in metrics:
                            evaluation[m][mo-1, elem, i] = np.nan
                    else:
                        # Calculate evaluation metrics
                        rmse = np.sqrt(mean_squared_error(results['obs'], results['pre']))
                        kge_prime_prime = KGE_Tang2021(results['obs'].values, results['pre'].values, min_obs_KGE)
                        evaluation['RMSE'][mo-1, elem, i] = rmse
                        evaluation["KGE''"][mo-1, elem, i] = kge_prime_prime['KGE']
                        evaluation["KGE''_corr"][mo-1, elem, i] = kge_prime_prime['r']
                        evaluation["KGE''_bias"][mo-1, elem, i] = kge_prime_prime['beta']
                        evaluation["KGE''_var"][mo-1, elem, i] = kge_prime_prime['alpha']

    print("Artificial gap filling completed!")
    
    if flag == 1:
        plt.tight_layout()
        return evaluation, fig
    else:
        return evaluation

def KGE_Tang2021(obs, pre, min_obs_KGE):
    """Calculates the modified Kling-Gupta Efficiency (KGE") and its 3 components.
    The KGE measures the correlation, bias and variability of the simulated values against the observed values.
    KGE" was proposed by Tang et al. (2021) to solve issues arising with 0 values in the KGE or KGE'.
    For more info, see https://doi.org/10.1175/jcli-d-21-0067.1
    KGE" range: -Inf to 1. Perfect score: 1. Units: Unitless.
    Correlation (r): Perfect score is 1.
    Bias ratio (beta): Perfect score is 0.
    Variability ratio (alpha):  Perfect score is 1.

    Keyword arguments:
    ------------------
    - obs: Numpy Array of observations to evaluate
    - pre: Numpy Array of predictions/simulations to evaluate
    - min_obs_KGE: Positive integer for the minimum number of stations required to calculate a station's cdf

    Returns:
    --------
    - KGEgroup: Dictionary containing the final KGE'' value as well as all elements of the KGE''
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
        # For more info: https://doi.org/10.5194/hess-23-4323-2019 (Section 2)
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
            data_month_gapfilled = gapfilled_SWE_data.sel(station_id=SWE_stations['station_id'].values, time=( (gapfilled_SWE_data['time.month'] == m) & (gapfilled_SWE_data['time.day'] == 1) ))

            # count the % of stations with data on those dates
            data_month_gapfilled_count = data_month_gapfilled.count(dim='station_id') / len(SWE_stations) * 100

            # plot bar chart of available data
            axs[elem,column].bar(data_month_gapfilled_count['time.year'], data_month_gapfilled_count.data, color='r', alpha=.5)

        # same process as above but for original SWE data
        data_month = original_SWE_data.sel(station_id=SWE_stations['station_id'].values, time=( (original_SWE_data['time.month'] == m) & (original_SWE_data['time.day'] == 1) ))
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

def calculate_stations_doy_corr(stations_obs, window_days, min_obs_corr):

    """Calculating stations' correlations for each day of the year (doy; with a X-day window centered around the doy).

    Keyword arguments:
    ------------------
    - stations_obs: Pandas DataFrame of all SWE & P stations observations
    - window_days: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
    - min_obs_corr: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations

    Returns:
    --------
    - stations_doy_corr: Dictionary containing a Pandas DataFrame of stations correlations for each day of year

    """

    # Set up the dictionary to save all correlations
    stations_doy_corr = {}

    # Duplicate the stations observations Pandas DataFrame and add doy column
    stations_obs_doy = stations_obs.copy()
    stations_obs_doy['doy'] = stations_obs_doy.index.dayofyear

    # Loop over days of the year
    for doy in range(1,366+1):

        # calculate the start & end of the data selection window, with caution around the start & end of the calendar year
        window_start = (doy-window_days)%366
        window_start = 366 if window_start == 0 else window_start
        window_end = (doy+window_days)%366
        window_end = 366 if window_end == 0 else window_end

        # select data for the window of interest
        if window_start > window_end:
            data_window = stations_obs_doy[(stations_obs_doy['doy']>=window_start) | (stations_obs_doy['doy'] <= window_end)]
        else:
            data_window = stations_obs_doy[(stations_obs_doy['doy']>=window_start) & (stations_obs_doy['doy'] <= window_end)]

        # calculate the Pearson product-moment correlations between stations if the minimum number of observations criterium is met
        data_window = data_window.drop(columns=['doy'])
        corr = data_window.corr(method='spearman', min_periods=min_obs_corr)
        # np.fill_diagonal(corr.values, np.nan)

        # save correlation for the doy to the dictionary
        stations_doy_corr[doy] = corr

    return stations_doy_corr

def qm_gap_filling(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr):

    """Performing the gap filling for all missing observations (when possible) using quantile mapping.
    For each target station and each date for which date is missing, we identify a donor stations as the station with:
    - data for this date,
    - a cdf for this doy,
    - and the best correlation to the target station (correlation >= min_corr for this doy).

    Keyword arguments:
    ------------------
    - original_data: Pandas DataFrame of original stations' observations dataset, which will be gap filled
    - window_days: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
    - min_obs_corr: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations
    - min_obs_cdf: Positive integer for the minimum number of stations required to calculate a station's cdf
    - min_corr: Value between 0 and 1 for the minimum correlation value required to keep a donor station

    Returns:
    --------
    - gapfilled_data: Pandas DataFrame of gap filled stations' observations
    - data_type_flags: Pandas DataFrame with information about the type of data (estimates or observations) in the gap filled dataset
    - donor_stationIDs: Pandas DataFrame with information about the donor station used to fill each of the gaps

    """

    # Create a duplicate of the dataset to gap fill
    gapfilled_data = original_data.copy()

    # Remove P & external SWE stations (buffer) from the dataframe
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Keep only gap filled SWE stations (without P stations & external SWE stations)
    gapfilled_data = gapfilled_data[cols]

    # Add doy to the Pandas DataFrame
    original_data_with_doy = original_data.copy()
    original_data_with_doy['doy'] = original_data_with_doy.index.dayofyear

    # Set empty dataframes to keep track of data type and donor station ids
    data_type_flags = pd.DataFrame(data=0, index=original_data.index, columns=cols)
    donor_stationIDs = pd.DataFrame(data="", index=original_data.index, columns=cols, dtype=object)

    # Calculate correlations between stations that have overlapping observations
    print("Calculating correlations...")
    corr = calculate_stations_doy_corr(original_data_with_doy, window_days, min_obs_corr)

    # Pre-compute missing data mask for efficiency
    missing_mask = gapfilled_data.isna()
    
    # Get all dates that have missing data
    dates_with_missing = missing_mask.any(axis=1)
    dates_to_process = original_data.index[dates_with_missing]
    
    print(f"Processing {len(dates_to_process)} dates with missing data...")
    
    # Pre-compute DOY data for all dates
    doy_data = original_data_with_doy['doy'].to_dict()
    
    # Pre-compute time windows for each unique DOY to avoid repeated calculations
    unique_doys = set(doy_data.values())
    doy_windows = {}
    for doy in unique_doys:
        window_startdoy = (doy - window_days) % 366
        window_startdoy = 366 if window_startdoy == 0 else window_startdoy
        window_enddoy = (doy + window_days) % 366
        window_enddoy = 366 if window_enddoy == 0 else window_enddoy
        doy_windows[doy] = (window_startdoy, window_enddoy)

    # Pre-compute target data windows for each station and DOY combination
    print("Pre-computing target data windows...")
    target_data_cache = {}
    for target_station in cols:
        target_data_cache[target_station] = {}
        station_data = original_data_with_doy[target_station].dropna()
        station_doys = original_data_with_doy.loc[station_data.index, 'doy']
        
        for doy in unique_doys:
            window_startdoy, window_enddoy = doy_windows[doy]
            if window_startdoy > window_enddoy:
                mask = (station_doys >= window_startdoy) | (station_doys <= window_enddoy)
            else:
                mask = (station_doys >= window_startdoy) & (station_doys <= window_enddoy)
            target_data_cache[target_station][doy] = station_data[mask]

    # Process dates in batches for better memory efficiency
    batch_size = min(1000, len(dates_to_process))
    total_filled = 0
    
    for batch_start in range(0, len(dates_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(dates_to_process))
        batch_dates = dates_to_process[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(dates_to_process)-1)//batch_size + 1}")
        
        for d in batch_dates:
            doy = doy_data[d]
            window_startdoy, window_enddoy = doy_windows[doy]
            
            # Calculate the start and end dates of the time window for the gap filling steps
            window_startdate = d - pd.Timedelta(days=window_days)
            window_enddate = d + pd.Timedelta(days=window_days)

            # Get IDs of all stations with data for this date (and within time window)
            data_window = original_data_with_doy[window_startdate:window_enddate].dropna(axis=1, how='all')
            non_missing_stations = [c for c in data_window.columns if c != 'doy']
            
            if len(non_missing_stations) == 0:
                continue
                
            # Pre-compute days_to_date for this window
            days_to_date = abs((d - data_window.index).days)
            data_window = data_window.copy()
            data_window['days_to_date'] = days_to_date

            # Get stations that need gap filling for this date
            stations_to_fill = [station for station in cols if missing_mask.loc[d, station]]
            
            for target_station in stations_to_fill:
                # Use cached target data
                data_window_target = target_data_cache[target_station].get(doy, pd.Series(dtype=float))
                
                # We can continue if there are enough target data to build cdf
                if len(data_window_target) >= min_obs_cdf:
                    # Get correlation data for this DOY and target station
                    if doy not in corr or target_station not in corr[doy].columns:
                        continue
                        
                    station_corr = corr[doy][target_station].dropna()
                    
                    # Filter by stations with data and minimum correlation
                    valid_corr = station_corr[station_corr.index.isin(non_missing_stations)]
                    potential_donors = valid_corr[valid_corr >= min_corr]
                    potential_donors = potential_donors[potential_donors.index != target_station]

                    if len(potential_donors) > 0:
                        # Sort donors by correlation (highest first)
                        potential_donors_sorted = potential_donors.sort_values(ascending=False)

                        # Try each donor station
                        for donor_station in potential_donors_sorted.index:
                            # Use cached or compute donor data
                            if donor_station not in target_data_cache:
                                continue
                                
                            data_window_donor = target_data_cache.get(donor_station, {}).get(doy, pd.Series(dtype=float))

                            # We can continue if there are enough donor data to build cdf
                            if len(data_window_donor) >= min_obs_cdf:
                                # Get the closest donor value
                                donor_data_in_window = data_window[donor_station].dropna()
                                if len(donor_data_in_window) > 0:
                                    # Find closest value by days_to_date
                                    valid_indices = donor_data_in_window.index
                                    closest_idx = data_window.loc[valid_indices, 'days_to_date'].idxmin()
                                    value_donor = data_window.loc[closest_idx, donor_station]

                                    # Perform the gap filling using quantile mapping
                                    value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                    if value_target is not None:
                                        gapfilled_data.loc[d, target_station] = value_target
                                        data_type_flags.loc[d, target_station] = 1
                                        donor_stationIDs.loc[d, target_station] = donor_station
                                        total_filled += 1
                                        break

    print(f"Gap filling completed. Filled {total_filled} missing values.")
    return gapfilled_data, data_type_flags, donor_stationIDs

def quantile_mapping(data_donor, data_target, value_donor, min_obs_cdf, flag):

    """Calculating target station's gap filling value from donor station's value using quantile mapping.

    Keyword arguments:
    ------------------
    - data_donor: Pandas DataFrame of donor station observations used to build empirical cdf
    - data_target: Pandas DataFrame of target station observations used to build empirical cdf
    - value_donor: Integer of donor station value used in the quantile mapping
    - min_obs_cdf: Positive integer for the minimum number of unique observations required to calculate a station's cdf
    - flag: Integer to plot (1) or not (0) the donor and target stations' cdfs

    Returns:
    --------
    - value_target: Integer of target station value calculated using quantile mapping
    - plot of the donor and target stations' cdfs (optional)

    """

    # build the donor station's empirical cdf
    sorted_data_donor = data_donor.drop_duplicates().sort_values().reset_index(drop=True)

    # build the target station's empiral cdf
    sorted_data_target = data_target.drop_duplicates().sort_values().reset_index(drop=True)

    # Calculate the donor & target stations' cdfs if they both have at least X unique observations
    if (len(sorted_data_donor) >= min_obs_cdf) & (len(sorted_data_target) >= min_obs_cdf):

        # Calculate the cumulative probability corresponding to the donor value
        # Find the position (rank) of the donor value in the sorted array
        matching_indices = np.where(sorted_data_donor == value_donor)[0]
        if len(matching_indices) > 0:
            rank_donor_obs = matching_indices[0]  # This is now a positional index (integer)
        else:
            # If exact match not found, find the closest value
            closest_idx = np.argmin(np.abs(sorted_data_donor - value_donor))
            rank_donor_obs = closest_idx
        
        total_obs_donor = len(sorted_data_donor)
        cumul_prob_donor_obs = (rank_donor_obs + 1) / total_obs_donor

        # Calculate the cumulative probability corresponding to the target value
        cumul_prob_target = np.arange(1, len(sorted_data_target) + 1) / len(sorted_data_target)

        # inter-/extrapolate linearly to get the target value corresponding to the donor station's cumulative probability
        inverted_edf = interp1d(cumul_prob_target, sorted_data_target.values, fill_value="extrapolate")
        value_target = round(float(inverted_edf(cumul_prob_donor_obs)), 2)

        # set any potential negative values from interpolation/extrapolation to zero
        if value_target < 0:
            value_target = 0

        # if requested, plot the target & donor stations' cdfs
        if flag == 1:
            plt.figure()
            plt.plot(sorted_data_donor.values, np.arange(1, len(sorted_data_donor) + 1) / len(sorted_data_donor), label='donor')
            plt.plot(sorted_data_target.values, cumul_prob_target, label='target')
            plt.scatter(value_donor, cumul_prob_donor_obs)
            plt.legend()

        return value_target

    # If either/both the target & donor stations have < X observations do nothing
    else:
        return None
