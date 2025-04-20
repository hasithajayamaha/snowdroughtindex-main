# Gap Filling Methodology

This document provides a detailed explanation of the gap filling methodology implemented in the Snow Drought Index package.

## Overview

Gap filling is a critical preprocessing step in snow water equivalent (SWE) data analysis, as SWE observations often contain missing values due to various reasons such as instrument failures, maintenance issues, or data transmission problems. The Snow Drought Index package implements a robust gap filling methodology based on quantile mapping, which leverages spatial correlations between stations to fill in missing values.

## Methodology

The gap filling process consists of several steps:

1. **Linear Interpolation**: Small gaps in the data are filled using linear interpolation.
2. **Quantile Mapping**: Larger gaps are filled using quantile mapping, which leverages donor stations with high correlations to the target station.
3. **Evaluation**: The gap filling performance is evaluated using artificial gap filling.

### Linear Interpolation

Linear interpolation is used to fill small gaps in the data (typically less than 15 days). This is implemented using the `interpolate_na` method from xarray:

```python
SWE_obs_basin_interp_da = SWE_stations_ds.snw.interpolate_na(method='linear', dim='time', max_gap=datetime.timedelta(days=max_gap_days))
```

### Quantile Mapping

For larger gaps, the package uses quantile mapping, which leverages donor stations with high correlations to the target station. The quantile mapping process consists of the following steps:

1. **Calculate Correlations**: Calculate correlations between stations for each day of the year (doy) using a window of days centered around the doy.
2. **Identify Donor Stations**: For each missing value, identify potential donor stations with correlations above a minimum threshold.
3. **Build Empirical CDFs**: Build empirical cumulative distribution functions (CDFs) for the target and donor stations.
4. **Map Quantiles**: Map the donor station's value to the target station's CDF to estimate the missing value.

#### Calculate Correlations

```python
def calculate_stations_doy_corr(stations_obs, window_days, min_obs_corr):
    """Calculating stations' correlations for each day of the year (doy; with a X-day window centered around the doy)."""
    # Set up the dictionary to save all correlations
    stations_doy_corr = {}

    # Duplicate the stations observations Pandas DataFrame and add doy column
    stations_obs_doy = stations_obs.copy()
    stations_obs_doy['doy'] = stations_obs_doy.index.dayofyear

    # Loop over days of the year
    for doy in range(1,366+1):
        # calculate the start & end of the data selection window
        window_start = (doy-window_days)%366
        window_start = 366 if window_start == 0 else window_start
        window_end = (doy+window_days)%366
        window_end = 366 if window_end == 0 else window_end

        # select data for the window of interest
        if window_start > window_end:
            data_window = stations_obs_doy[(stations_obs_doy['doy']>=window_start) | (stations_obs_doy['doy'] <= window_end)]
        else:
            data_window = stations_obs_doy[(stations_obs_doy['doy']>=window_start) & (stations_obs_doy['doy'] <= window_end)]

        # calculate the Pearson product-moment correlations between stations
        data_window = data_window.drop(columns=['doy'])
        corr = data_window.corr(method='spearman', min_periods=min_obs_corr)

        # save correlation for the doy to the dictionary
        stations_doy_corr[doy] = corr

    return stations_doy_corr
```

#### Quantile Mapping

```python
def quantile_mapping(data_donor, data_target, value_donor, min_obs_cdf, flag):
    """Calculating target station's gap filling value from donor station's value using quantile mapping."""
    # build the donor station's empirical cdf
    sorted_data_donor = data_donor.drop_duplicates().sort_values(ignore_index=True)

    # build the target station's empiral cdf
    sorted_data_target = data_target.drop_duplicates().sort_values(ignore_index=True)

    # Calculate the donor & target stations' cdfs if they both have at least X unique observations
    if (len(sorted_data_donor) >= min_obs_cdf) & (len(sorted_data_target) >= min_obs_cdf):
        # Calculate the cumulative probability corresponding to the donor value
        rank_donor_obs = sorted_data_donor[sorted_data_donor == value_donor].index[0]
        total_obs_donor = len(sorted_data_donor)
        cumul_prob_donor_obs = (rank_donor_obs + 1) / total_obs_donor

        # Calculate the cumulative probability corresponding to the target value
        cumul_prob_target = np.arange(1,len(sorted_data_target)+1) / (len(sorted_data_target))

        # inter-/extrapolate linearly to get the target value corresponding to the donor station's cumulative probability
        inverted_edf = interp1d(cumul_prob_target, sorted_data_target, fill_value="extrapolate")
        value_target = round(float(inverted_edf(cumul_prob_donor_obs)),2)

        # set any potential negative values from interpolation/extrapolation to zero
        if(value_target) < 0:
            value_target = 0

        return value_target
    # If either/both the target & donor stations have < X observations do nothing
    else:
        return None
```

#### Gap Filling Process

```python
def qm_gap_filling(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr):
    """Performing the gap filling for all missing observations (when possible) using quantile mapping."""
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
        doy = original_data.loc[d,'doy']

        # Calculate the start and end dates of the time window for the gap filling steps
        window_startdate = d - pd.Timedelta(days=window_days)
        window_enddate = d + pd.Timedelta(days=window_days)

        # Get IDs of all stations with data for this date (and within time window)
        data_window = original_data[window_startdate:window_enddate].dropna(axis=1, how='all')
        non_missing_stations = [c for c in data_window.columns if 'doy' not in c]
        data_window['days_to_date'] = abs((d - data_window.index).days)

        # Calculate the start & end doys of the time window for quantile mapping
        window_startdoy = (data_window['doy'].iloc[0])%366
        window_startdoy = 366 if window_startdoy == 0 else window_startdoy
        window_enddoy = (data_window['doy'].iloc[-1])%366
        window_enddoy = 366 if window_enddoy == 0 else window_enddoy

        # Loop over stations to gap fill
        for target_station in cols:
            # If station has no data, proceed with the gap filling
            if np.isnan(original_data.loc[d,target_station]):
                # Select target data within time window for this doy from all years
                if window_startdoy > window_enddoy:
                    data_window_target = original_data[target_station].dropna()[(original_data['doy']>=window_startdoy) | (original_data['doy'] <= window_enddoy)]
                else:
                    data_window_target = original_data[target_station].dropna()[(original_data['doy']>=window_startdoy) & (original_data['doy'] <= window_enddoy)]

                # We can continue if there are enough target data to build cdf
                if len(data_window_target.index) >= min_obs_cdf:
                    # Get ids of all stations with correlations >= a minimum correlation for this doy
                    non_missing_corr = corr[doy][target_station].dropna()
                    non_missing_corr = non_missing_corr[non_missing_corr.index.isin(non_missing_stations)]
                    potential_donor_stations = non_missing_corr[non_missing_corr >= min_corr].index.values
                    potential_donor_stations = [c for c in potential_donor_stations if target_station not in c]

                    # If there is at least one potential donor station, proceed
                    if len(potential_donor_stations) > 0:
                        # Sort the donor stations from highest to lowest value
                        potential_donor_stations_sorted = corr[doy].loc[potential_donor_stations,target_station].dropna().sort_values(ascending=False).index.values

                        # Loop over sorted donor stations until I find one with enough data to build a cdf
                        for donor_station in potential_donor_stations_sorted:
                            # Select data within time window for this doy from all years
                            if window_startdoy > window_enddoy:
                                data_window_donor = original_data[donor_station].dropna()[(original_data['doy'] >= window_startdoy) | (original_data['doy'] <= window_enddoy)]
                            else:
                                data_window_donor = original_data[donor_station].dropna()[(original_data['doy'] >= window_startdoy) & (original_data['doy'] <= window_enddoy)]

                            # We can continue if there are enough donor data to build cdf
                            if len(data_window_donor.index) >= min_obs_cdf:
                                # If the donor station has multiple values within the window, we keep the closest donor station value to the date we are gap filling
                                sorted_data_window = data_window.sort_values(by=['days_to_date'])
                                value_donor = sorted_data_window[donor_station].dropna()[0]

                                # Perform the gap filling using quantile mapping
                                value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                if value_target != None:
                                    gapfilled_data.loc[d,target_station] = value_target
                                    data_type_flags.loc[d,target_station] = 1
                                    donor_stationIDs.loc[d,target_station] = donor_station

                                break

                            else:
                                continue

    return gapfilled_data, data_type_flags, donor_stationIDs
```

### Evaluation

The gap filling performance is evaluated using artificial gap filling, which involves creating artificial gaps in the data and then evaluating how well the gap filling methodology can fill these gaps. This is implemented in the `artificial_gap_filling` function:

```python
def artificial_gap_filling(original_data, iterations, artificial_gap_perc, window_days, min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag):
    """Creating random artificial gaps in the original dataset for each month & station, and running the gap filling function to assess its performance."""
    # suppresses the "SettingWithCopyWarning"
    pd.set_option("mode.chained_assignment", None)

    # Identify stations for gap filling (without P & external SWE stations (buffer) as we don't do any gap filling for these)
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Create an empty dictionary to store the metric values for each month, station & iteration
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan

    # Calculate correlations between stations that have overlapping observations
    corr = calculate_stations_doy_corr(original_data, window_days, min_obs_corr)

    # loop over months
    for mo in range(1,12+1):
        # loop over iterations
        for i in range(iterations):
            # initialize counter to assign results to the right station
            elem = -1

            # looping over stations
            for s in cols:
                # update counter to assign results to the right station
                elem += 1

                # duplicate original data to create artificial gaps from this
                artificial_gaps_data = original_data.copy()

                # remove all missing values for a given station for which to perform gap filling
                station_nomissing_values = pd.DataFrame(artificial_gaps_data[s].dropna())

                # add DOY to select data to gap fill within a time window around first day of month
                station_nomissing_values['doy'] = station_nomissing_values.index.dayofyear

                # calculate the doy corresponding to the date - using 2010 as common year (not leap year)
                doy = int(datetime.datetime(2010,mo,1).strftime('%j'))

                # calculate the start & end doys of the time window for quantile mapping
                window_startdoy = (doy-window_days)%365
                window_startdoy = 365 if window_startdoy == 0 else window_startdoy
                window_enddoy = (doy+window_days)%365
                window_enddoy = 366 if window_enddoy == 0 or window_enddoy == 365 else window_enddoy

                # select data within time window
                if window_startdoy > window_enddoy:
                    data_window = station_nomissing_values[(station_nomissing_values['doy']>=window_startdoy) | (station_nomissing_values['doy'] <= window_enddoy)]
                else:
                    data_window = station_nomissing_values[(station_nomissing_values['doy']>=window_startdoy) & (station_nomissing_values['doy'] <= window_enddoy)]

                # Select target data within this time window
                data_window_target = data_window[s]

                # calculate the number of observations to remove for this station & month's first day
                n = int(len(data_window.index) * artificial_gap_perc / 100)

                # if the number of observations is above zero we can proceed with the gap filling
                if n > 0:
                    # randomly select n dates from the station's data (no duplicates) and remove them from the original dataset
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
                        doy = data_window.dropna().loc[d,'doy']

                        # Get IDs of all stations with data for this date (and within time window)
                        data_window_allstations = artificial_gaps_data.dropna(axis=1, how='all')
                        non_missing_stations = [c for c in data_window_allstations.columns]
                        data_window_allstations['days_to_date'] = abs((d - data_window_allstations.index).days)

                        # We can continue if there are enough target data to build cdf
                        if len(data_window_target.index) >= min_obs_cdf:
                            # Get ids of all stations with correlations >= a minimum correlation for this doy
                            non_missing_corr = corr[doy][s].dropna()
                            non_missing_corr = non_missing_corr[non_missing_corr.index.isin(non_missing_stations)]
                            potential_donor_stations = non_missing_corr[non_missing_corr >= min_corr].index.values
                            potential_donor_stations = [c for c in potential_donor_stations if s not in c]

                            # If there is at least one potential donor station, proceed
                            if len(potential_donor_stations) > 0:
                                # Sort the donor stations from highest to lowest value
                                potential_donor_stations_sorted = corr[doy].loc[potential_donor_stations,s].dropna().sort_values(ascending=False).index.values

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

                                        if value_target != None:
                                            gapfilled_data.loc[d] = value_target

                                        break

                                    else:
                                        continue

                    # combine observed & predicted data into a single Pandas dataframe
                    results = gapfilled_data.to_frame(name='pre')
                    results['obs'] = original_data[s].loc[dates_to_remove]
                    results = results.dropna()

                    # if there are no predicted values set the metrics to nan
                    if results.empty == True:
                        for m in metrics:
                            evaluation[m][mo-1,elem,i] = np.nan

                    # otherwise proceed with evaluating the gap filling performance
                    else:
                        rmse = mean_squared_error(results['obs'], results['pre'], squared=False)
                        kge_prime_prime = KGE_Tang2021(results['obs'].values, results['pre'].values, min_obs_KGE)
                        evaluation['RMSE'][mo-1,elem,i] = rmse
                        evaluation["KGE''"][mo-1,elem,i] = kge_prime_prime['KGE']
                        evaluation["KGE''_corr"][mo-1,elem,i] = kge_prime_prime['r']
                        evaluation["KGE''_bias"][mo-1,elem,i] = kge_prime_prime['beta']
                        evaluation["KGE''_var"][mo-1,elem,i] = kge_prime_prime['alpha']

                # else if the number of observations is zero we go to the next station
                else:
                    continue

    return evaluation
```

## Parameters

The gap filling process can be customized using several parameters:

- **window_days**: The number of days to select data for around a certain day of year (doy), to calculate correlations.
- **min_obs_corr**: The minimum number of overlapping observations required to calculate the correlation between 2 stations.
- **min_obs_cdf**: The minimum number of stations required to calculate a station's cumulative distribution function (CDF).
- **min_corr**: The minimum correlation value required to keep a donor station.
- **min_obs_KGE**: The minimum number of stations required to calculate the Kling-Gupta Efficiency (KGE).
- **max_gap_days**: The maximum number of days for gaps allowed in the daily SWE data for the linear interpolation.

## Visualization

The package provides several functions to visualize the gap filling results:

- **data_availability_monthly_plots_1**: Plots the percentage of SWE stations available on the first day of each month of each year, before and after gap filling.
- **data_availability_monthly_plots_2**: Creates bar chart subplots of the days with SWE observations around the 1st day of each month.
- **plots_artificial_gap_evaluation**: Plots the evaluation results for the artificial gap filling.

## References

- Tang, G., Clark, M. P., & Papalexiou, S. M. (2021). SC-earth: A station-based serially complete earth dataset from 1950 to 2019. Journal of Climate, 34(16), 6493-6511.
