# Functions Documentation

This document provides detailed information about the functions available in the `functions.py` module of the Snow Drought Index package.

## Table of Contents

1. [artificial_gap_filling](#artificial_gap_filling)
2. [basins_maps](#basins_maps)
3. [calculate_stations_doy_corr](#calculate_stations_doy_corr)
4. [circular_stats](#circular_stats)
5. [continuous_rank_prob_score](#continuous_rank_prob_score)
6. [corr_coeff_squared](#corr_coeff_squared)
7. [data_availability_monthly_plots_1](#data_availability_monthly_plots_1)
8. [data_availability_monthly_plots_2](#data_availability_monthly_plots_2)
9. [deterministic_forecasting](#deterministic_forecasting)
10. [deterministic_forecast_plots](#deterministic_forecast_plots)
11. [det_metrics_calculation](#det_metrics_calculation)
12. [ensemble_dressing](#ensemble_dressing)
13. [ensemble_forecasting](#ensemble_forecasting)
14. [ensemble_forecast_plots](#ensemble_forecast_plots)
15. [extract_monthly_data](#extract_monthly_data)
16. [extract_stations_in_basin](#extract_stations_in_basin)
17. [hydrographs](#hydrographs)
18. [KGE_Tang2021](#kge_tang2021)
19. [leave_out](#leave_out)
20. [maps_loadings](#maps_loadings)
21. [metrics_bootstrap_plots](#metrics_bootstrap_plots)
22. [OLS_model_fitting](#ols_model_fitting)
23. [perc_difference](#perc_difference)
24. [plots_artificial_gap_evaluation](#plots_artificial_gap_evaluation)
25. [polar_plot](#polar_plot)
26. [predictor_predictand_corr_plot](#predictor_predictand_corr_plot)
27. [principal_component_analysis](#principal_component_analysis)
28. [prob_metrics_calculation](#prob_metrics_calculation)
29. [qm_gap_filling](#qm_gap_filling)
30. [quantile_mapping](#quantile_mapping)
31. [regime_classification](#regime_classification)
32. [reli_index](#reli_index)
33. [ROC](#roc)
34. [ROC_plots](#roc_plots)
35. [split_sample](#split_sample)
36. [stations_basin_map](#stations_basin_map)
37. [streamflow_peaks_statistics](#streamflow_peaks_statistics)

---

## artificial_gap_filling

```python
def artificial_gap_filling(original_data, iterations, artificial_gap_perc, window_days, min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag):
```

Creates random artificial gaps in the original dataset for each month & station, and runs the gap filling function to assess its performance.

### Parameters

- **original_data**: Pandas DataFrame of original stations' observations dataset, to which data will be removed for artificial gap filling
- **iterations**: Positive integer denoting the number of times we want to repeat the artificial gap filling (we remove data at random each time in the original dataset)
- **artificial_gap_perc**: Percentage between 1 and 100 for the amount of data to remove for each station & month's first day
- **window_days**: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
- **min_obs_corr**: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations
- **min_obs_cdf**: Positive integer for the minimum number of stations required to calculate a station's cdf
- **min_corr**: Value between 0 and 1 for the minimum correlation value required to keep a donor station
- **min_obs_KGE**: Positive integer for the minimum number of stations required to calculate a station's cdf
- **flag**: Integer to plot the gap filled values vs the observed values (1) or not (0)

### Returns

- **evaluation**: Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration
- **fig** (optional): A figure of the gap filled vs. the actual SWE observations for each first day of the month

---

## basins_maps

```python
def basins_maps(basins, method, variable, nival_start_doy, nival_end_doy, domain):
```

Plots two maps of basins provided, one that shows the basins' shapes & one that shows the basins' outlets.

### Parameters

- **basins**: Pandas GeoDataFrame of all basin shapefiles available to subset from
- **method**: String of the metric used to identify streamflow peaks (shown on maps' title)
- **variable**: String of the column label to be used for colouring the maps
- **nival_start_doy**: Integer day of year (doy) of the start of the nival period
- **nival_end_doy**: Integer day of year (doy) of the end of the nival period
- **domain**: String of the geographical domain to plot

### Returns

- Two maps of basins.

---

## calculate_stations_doy_corr

```python
def calculate_stations_doy_corr(stations_obs, window_days, min_obs_corr):
```

Calculating stations' correlations for each day of the year (doy; with a X-day window centered around the doy).

### Parameters

- **stations_obs**: Pandas DataFrame of all SWE & P stations observations
- **window_days**: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
- **min_obs_corr**: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations

### Returns

- **stations_doy_corr**: Dictionary containing a Pandas DataFrame of stations correlations for each day of year

---

## circular_stats

```python
def circular_stats(doy, year_length):
```

Calculates circular statistics. See: https://onlinelibrary.wiley.com/doi/abs/10.1002/hyp.7625

### Parameters

- **doy**: Numpy array of day of year (doy) values for identified streamflow peaks
- **year_length**: Numpy array of year length values for identified streamflow peaks

### Returns

- **theta_rad**: Numpy array of the angular values (in radians) that correspond to the doy
- **regularity**: Dimensionless value indicating the spread of the data (ranges from 0: high spread to 1: low spread)

---

## continuous_rank_prob_score

```python
def continuous_rank_prob_score(Qobs, Qfc_ens, min_obs):
```

Calculates the Continuous Rank Probability Score (CRPS) and the Continuous Rank Probability Skill Score (CRPSS).
The CRPS is a measure of the difference between the predicted (from an ensemble or probabilistic forecast) and the observed cumulative distribution functions (cdf).
The CRPSS measures the performance (CRPS) of the forecast against a baseline (e.g., the observation climatology).

### Parameters

- **Qobs**: xarray DataArray containing a timeseries of observed flow values.
- **Qfc_ens**: xarray DataArray containing a timeseries of ensemble flow forecasts.
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

### Returns

- **CRPS**: Float of the CRPS value between the ensemble forecasts & observations.
- **CRPSS**: Float of the CRPSS value between the ensemble forecasts & observations.

---

## corr_coeff_squared

```python
def corr_coeff_squared(Qobs, Qfc_det, min_obs):
```

Calculates the squared Pearson correlation coefficient between deterministic forecasts & observations.

### Parameters

- **Qobs**: Numpy Array containing a timeseries of observed flow values.
- **Qfc_det**: Numpy Array containing a timeseries of deterministic flow forecasts (e.g., medians or means of the ensemble forecasts).
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics.

### Returns

- **r_squared**: Float of the squared correlation coefficient between the deterministic forecasts & observations.

---

## data_availability_monthly_plots_1

```python
def data_availability_monthly_plots_1(SWE_stations, original_SWE_data, gapfilled_SWE_data, flag):
```

Calculating and plotting the % of SWE stations available on the first day of each month of each year.

### Parameters

- **SWE_stations**: Pandas GeoDataFrame of all SWE stations
- **original_SWE_data**: xarray DataArray of the original SWE observations
- **gapfilled_SWE_data**: xarray DataArray of the SWE observations after gap filling
- **flag**: Flag to indicate if gap filled data was provided (1) or not (0). In the case that it is provided, a comparison plot will be made to compare data availability in the original data vs the gap filled data

### Returns

- Bar chart timeseries of SWE stations available on the first day of each month of each year

---

## data_availability_monthly_plots_2

```python
def data_availability_monthly_plots_2(SWE_data):
```

Creating bar chart subplots of the days with SWE observations around the 1st day of each month.

### Parameters

- **SWE_data**: Pandas DataFrame containing the SWE stations observations

### Returns

- Bar chart subplots of the days with SWE observations around the 1st day of each month

---

## deterministic_forecasting

```python
def deterministic_forecasting(model, test_timeseries):
```

Out-of-sample forecasting based on the test data predictor(s) and the model developed on the training data.

### Parameters

- **model**: Regression model from statsmodels developed on the training data
- **test_timeseries**: Pandas DataFrame containing the predictor & predictand data to use for testing the forecast model

### Returns

- **flow_fc_mean**: Pandas Dataframe containing the flow forecast means

---

## deterministic_forecast_plots

```python
def deterministic_forecast_plots(obs_timeseries, det_fc_timeseries, predictor_month, predictand_start_month, predictand_end_month, units):
```

Plot deterministic forecasts.

### Parameters

- **obs_timeseries**: Pandas DataFrame of the observed flow accumulation data
- **det_fc_timeseries**: Pandas DataFrame of the deterministic forecasts of flow accumulation
- **predictor_month**: Integer of the month of predictor data to use
- **predictand_start_month**: Integer of the starting month of predictand data to use
- **predictand_end_month**: Integer of the end month of predictand data to use
- **units**: string of the flow units to use on the plot

### Returns

- Timeseries and scatter plots of the flow accumulation deterministic forecasts and observations

---

## det_metrics_calculation

```python
def det_metrics_calculation(Qobs, Qfc_det, flag, niterations, min_obs):
```

Calculates deterministic metrics for whole hindcast timeseries (1 value per hindcast start date & target period).

### Parameters

- **Qobs**: xarray Dataset containing timeseries of observed flow values for various target periods of flow accumulation.
- **Qfc_det**: xarray Dataset containing timeseries of deterministic flow forecasts for various target periods and forecast start dates.
- **flag**: Integer to indicate whether the metrics should be calculated without (0) or with (1) bootstrapping
- **niterations**: Integer > 0 of the number of bootstrapping iterations to perform
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

### Returns

- **rsquared_da**: xarray DataArray of squared correlation coefficient values for various forecast start dates & target periods
- **kge_da**: xarray DataArray of KGE" values for various forecast start dates & target periods
- **perc_diff_da**: xarray DataArray of percentage difference values for various forecast start dates & target periods

---

## ensemble_dressing

```python
def ensemble_dressing(det_fc, SD, ens_size):
```

Generate ensembles around deterministic forecasts. They are generated by drawing random samples from a normal (Gaussian) distribution.

### Parameters

- **det_fc**: Pandas DataFrame of the deterministic forecasts
- **SD**: Positive value of the standard deviation of errors between the forecast means and observed values during training
- **ens_size**: Integer > 0 of the number of ensemble members to generate

### Returns

- **ens_fc**: Pandas DataFrame of the ensemble forecasts produced

---

## ensemble_forecasting

```python
def ensemble_forecasting(predictor_data, predictand_data, PC_ids, ens_size, min_overlap_years, method_traintest, nyears_leaveout):
```

Generate ensemble forecasts of flow accumulations (predictand) from SWE PC(s) (predictors).

### Parameters

- **predictor_data**: Pandas DataFrame of the predictor data
- **predictand_data**: Pandas DataFrame of the predictand data
- **PC_ids**: String (if only 1) or list (if > 1) of the PC(s) to use as predictor data
- **ens_size**: Integer > 0 of the number of ensemble members to generate
- **min_overlap_years**: Positive integer indicating the minimum number of years required of PC-volume to be able to generate a forecast
- **method_traintest**: String to define the method used to split the data into training and testing samples
- **nyears_leaveout**: Integer above zero for the number of years to leave out at a time

### Returns

- **fc_ens_df**: Pandas dataframe containing all generated ensemble hindcasts

---

## ensemble_forecast_plots

```python
def ensemble_forecast_plots(obs_timeseries, ens_fc_timeseries, predictor_month, predictand_target_period, units):
```

Plot ensemble forecasts timeseries as boxplots.

### Parameters

- **obs_timeseries**: Pandas DataFrame of the observed flow accumulation data
- **ens_fc_timeseries**: Pandas DataFrame of the ensemble forecasts of flow accumulation
- **predictor_month**: String of the month of predictor data to use
- **predictand_target_period**: String of the predictand target period to use
- **units**: string of the flow units to use on the plot

### Returns

- Timeseries plot of the flow accumulation ensemble forecasts and observations

---

## extract_monthly_data

```python
def extract_monthly_data(stations_data, month, flag):
```

For the PCA & forecasting, we need a full dataset (no missing data) for specific dates.
For our use, we extract data (with no missing values) for the first day of a given month.
We find the optimal number of stations and years of data we keep.

### Parameters

- **stations_data**: Pandas DataFrame containing the (gapfilled) SWE stations observations
- **month**: Integer between 1 and 12 to specify the month for which we want to extract data (1st day of the month extracted)
- **flag**: Integer to plot the evolution of the selection criteria (1) or not (0)

### Returns

- **month_stations_final**: Pandas DataFrame containing the SWE stations observations to keep
- optional plot of the evolution of the # of stations & years we can keep

---

## extract_stations_in_basin

```python
def extract_stations_in_basin(stations, basins, basin_id, buffer_km=0):
```

Extracts stations within a specified basin (with or without a buffer) and returns the extracted stations.

### Parameters

- **stations**: Pandas GeoDataFrame of all stations available to subset from
- **basins**: Pandas GeoDataFrame of all basin shapefiles available to subset from
- **basin_id**: String of basin station ID
- **buffer_km**: Positive value (in km) of buffer to add around basin shapefile (default=0; i.e., no buffer)

### Returns

- **stations_in_basin**: Pandas GeoDataFrame of all stations within the specified basin
- **basin_buffer**: Zero if the default buffer is selected, otherwise buffer geometry for plotting

---

## hydrographs

```python
def hydrographs(basins, streamflow_obs, month_start_water_year, day_start_water_year):
```

Plots normalized climatological streamflows for the provided basins, differenciating between nival and glacial basins.

### Parameters

- **basins**: Pandas GeoDataFrame of all basins to plot
- **streamflow_obs**: xarray Dataset of streamflow observations
- **month_start_water_year**: Integer of the water year starting month
- **day_start_water_year**: Integer of the water year starting day of the month

### Returns

- A plot of all basins' climatological hydrographs.

---

## KGE_Tang2021

```python
def KGE_Tang2021(obs, pre, min_obs_KGE):
```

Calculates the modified Kling-Gupta Efficiency (KGE") and its 3 components.
The KGE measures the correlation, bias and variability of the simulated values against the observed values.
KGE" was proposed by Tang et al. (2021) to solve issues arising with 0 values in the KGE or KGE'.

### Parameters

- **obs**: Numpy Array of observations to evaluate
- **pre**: Numpy Array of predictions/simulations to evaluate
- **min_obs_KGE**: Positive integer for the minimum number of stations required to calculate a station's cdf

### Returns

- **KGEgroup**: Dictionary containing the final KGE'' value as well as all elements of the KGE''

---

## leave_out

```python
def leave_out(original_timeseries, nyears_leaveout):
```

Splits predictor & predictand timeseries data using leave years out method (training & testing the forecast model).
E.g., if nyears_leaveout = 1, we leave one year out each time for which we want to test (i.e., validate) the model. All other years will be used for training the model.
If nyears_leaveout = 3, we leave 3 successive years out at a time.

### Parameters

- **original_timeseries**: Pandas DataFrame of the combined predictor & predictand timeseries
- **nyears_leaveout**: Integer above zero for the number of years to leave out at a time

### Returns

- **train_timeseries**: Pandas DataFrame containing the 1st half of the timeseries used for training the forecast model
- **test_timeseries**: Pandas DataFrame containing the 2nd half of the timeseries used for testing the forecast model

---

## maps_loadings

```python
def maps_loadings(dem_dir, basin_id, basin, SWE_stations, loadings, PC):
```

Creating maps of loadings between basin SWE stations and a given PC for each 1st of the month to see correlations in time & space.

### Parameters

- **dem_dir**: String of the path to DEMs
- **basin_id**: String of the basin id to plot
- **basin**: Pandas GeoDataFrame of basin to plot
- **SWE_stations**: Pandas GeoDataFrame of SWE stations to plot
- **loadings**: Dictionary of the PCA loadings (correlation between PCs & stations data)
- **PC**: String of the PC to create the maps for (e.g., 'PC1')

### Returns

- Maps of loadings between basin SWE stations and a given PC for each 1st of the month

---

## metrics_bootstrap_plots

```python
def metrics_bootstrap_plots(metric_values, min_value, max_value, flag_skill, flag_events):
```

Plots metrics median values with confidence intervals from bootstrapping.

### Parameters

- **metric_values**: xarray DataArray containing verification metric values for various target periods and forecast start dates.
- **min_value**: Minimum value of that verification metric
- **max_value**: Maximum value of that verification metric
- **flag_skill**: Integer to indicate whether to plot a 0 value threshold line (1) or not (0)
- **flag_events**: Integer to indicate whether to plot 1 score (0) or 2 scores/events to compare (1)

### Returns

- Sub-plots of the evolution of the verification metric values per start date for each target period.

---

## OLS_model_fitting

```python
def OLS_model_fitting(PC_ids, train_timeseries):
```

Fits the OLS model using the specified predictor(s) and training data.

### Parameters

- **PC_ids**: String (if only 1) or list (if more than 1) of the PC(s) to use as predictor data
- **train_timeseries**: Pandas DataFrame containing the predictor & predictand data to use for training the forecast model

### Returns

- **model_fit**: OLS model

---

## perc_difference

```python
def perc_difference(Qobs, Qfc_det, min_obs):
```

Calculates the percentage difference between deterministic forecasts & observations.

### Parameters

- **Qobs**: Numpy Array containing a timeseries of observed flow values
- **Qfc_det**: Numpy Array containing a timeseries of deterministic flow forecasts (e.g., medians or means of the ensemble forecasts)
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

### Returns

- **perc_diff**: Float of the percent difference between the deterministic forecasts & observations.

---

## plots_artificial_gap_evaluation

```python
def plots_artificial_gap_evaluation(evaluation_scores):
```

Plotting evaluation results for the artificial gap filling.

### Parameters

- **evaluation_metrics**: Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration

### Returns

- plots of the various evaluation metrics for all stations, iterations & each month's first day

---

## polar_plot

```python
def polar_plot(theta_rad, regularity, flag, nival_start_doy, nival_end_doy, nival_regularity_threshold):
```

Plots circular statistics on a polar plot for a single or multiple basins.

### Parameters

- **theta_rad**: Numpy array of the angular values (in radians) that correspond to the doy
- **regularity**: Dimensionless value indicating the spread of the data (ranges from 0: high spread to 1: low spread)
- **flag**: Flag to indicate if a single (0) or multiple (1) basins should be plotted
- **nival_start_doy**: Integer day of year (doy) of the start of the nival period
- **nival_end_doy**: Integer day of year (doy) of the end of the nival period
- **nival_regularity_threshold**: Float of the minimum regularity threshold allowed for basins to be categorized as being nival

### Returns

- A polar plot of circular statistics.

---

## predictor_predictand_corr_plot

```python
def predictor_predictand_corr_plot(predictor_data, predictand_data, PC_id, start_months, end_month, min_obs_corr):
```

Calculates and plots correlations between a SWE PC (predictor) & flow volumes (predictand) for different lead times and for different volume accumulation periods.

### Parameters

- **predictor_data**: Pandas DataFrame of the predictor (SWE PC) data
- **predictand_data**: Pandas DataFrame of the predictand (flow columes) data
- **PC_id**: String of the PC to use for the predictor data
- **start_months**: List of integers of the starting months of volume accumulation periods (predictand)
- **end_month**: Integer of the end month of volume accumulation periods (predictand)
- **min_obs_corr**: Positive integer defining the minimum number of observations required to calculate the correlation between predictand-predictor

### Returns

- **correlations**: Pandas DataFrame of the correlations between predictors and predictands
- A matrix plot of the correlations between predictors and predictands

---

## principal_component_analysis

```python
def principal_component_analysis(stations_data, flag):
```

Transforming stations observations into principal components.

### Parameters

- **stations_data**: Pandas DataFrame containing the (gapfilled) SWE stations observations with no missing values
- **flag**: Integer to plot the PCA explained variance per PC (1) or not (0)

### Returns

- **PCs_df**: Pandas DataFrame containing the principal components data
- **loadings_df**: Dictionary of the PCA loadings (correlation between PCs & stations data)

---

## prob_metrics_calculation

```python
def prob_metrics_calculation(Qobs, Qfc_ens, flag, niterations, perc_event_low, perc_event_high, min_obs, bins_thresholds):
```

Calculates deterministic metrics for whole hindcast timeseries (1 value per hindcast start date & target period).

### Parameters

- **Qobs**: xarray Dataset containing timeseries of observed flow values for various target periods of flow accumulation.
- **Qfc_ens**: xarray Dataset containing timeseries of ensemble flow forecasts for various target periods and forecast start dates.
- **flag**: Integer to indicate whether the metrics should be calculated without (0) or with (1) bootstrapping
- **niterations**: Integer > 0 of the number of bootstrapping iterations to perform
- **perc_event_low**: Float between 0 and 1 to indicate the percentile of the low flow event for which calculations are made.
- **perc_event_high**: Float between 0 and 1 to indicate the percentile of the high flow event for which calculations are made.
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics
- **bins_thresholds**: Numpy array of increasing probability thresholds to make the yes/no decision

### Returns

- **crps_da**: xarray DataArray containing the CRPS for each hindcast start date & target period.
- **crpss_da**: xarray DataArray containing the CRPSS for each hindcast start date & target period.
- **reli_da**: xarray DataArray containing the reliability index for each hindcast start date & target period.
- **roc_auc_da**: xarray DataArray containing the ROC area under the curve for each hindcast start date & target period.
- **roc_da**: xarray DataArray containing the ROC curves for each hindcast start date & target period.

---

## qm_gap_filling

```python
def qm_gap_filling(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr):
```

Performing the gap filling for all missing observations (when possible) using quantile mapping.
For each target station and each date for which date is missing, we identify a donor stations as the station with:
- data for this date,
- a cdf for this doy,
- and the best correlation to the target station (correlation >= min_corr for this doy).

### Parameters

- **original_data**: Pandas DataFrame of original stations' observations dataset, which will be gap filled
- **window_days**: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
- **min_obs_corr**: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations
- **min_obs_cdf**: Positive integer for the minimum number of stations required to calculate a station's cdf
- **min_corr**: Value between 0 and 1 for the minimum correlation value required to keep a donor station

### Returns

- **gapfilled_data**: Pandas DataFrame of gap filled stations' observations
- **data_type_flags**: Pandas DataFrame with information about the type of data (estimates or observations) in the gap filled dataset
- **donor_stationIDs**: Pandas DataFrame with information about the donor station used to fill each of the gaps

---

## quantile_mapping

```python
def quantile_mapping(data_donor, data_target, value_donor, min_obs_cdf, flag):
```

Calculating target station's gap filling value from donor station's value using quantile mapping.

### Parameters

- **data_donor**: Pandas DataFrame of donor station observations used to build empirical cdf
- **data_target**: Pandas DataFrame of target station observations used to build empirical cdf
- **value_donor**: Integer of donor station value used in the quantile mapping
- **min_obs_cdf**: Positive integer for the minimum number of unique observations required to calculate a station's cdf
- **flag**: Integer to plot (1) or not (0) the donor and target stations' cdfs

### Returns

- **value_target**: Integer of target station value calculated using quantile mapping
- plot of the donor and target stations' cdfs (optional)

---

## regime_classification

```python
def regime_classification(streamflow_obs, start_water_year, max_gap_days, flag):
```

Performs the regime classification for a given method (i.e., flag), using circular statistics from Burn et al. (2010): https://doi.org/10.1002/hyp.7625

### Parameters

- **streamflow_obs**: xarray Dataset of streamflow observations
- **start_water_year**: Tuple with (month, day) of the water year starting date
- **max_gap_days**: Positive integer of the max. number of days for gaps allowed in the daily streamflow data for the linear interpolation
- **flag**: An integer of 1, 2 or 3 defining the method to be used for identifying streamflow peaks
  -> flag=1: streamflow annual maxima
  -> flag=2: peak over threshold (POT) where the threshold = minimum value of all annual maxima
  -> flag=3: annual centres of mass (i.e., doy where 1/2 of the total water year streamflow has passed through the river - see: https://journals.ametsoc.org/view/journals/clim/18/2/jcli-3272.1.xml)

### Returns

- **basins_regimes_gdf**: Pandas GeoDataFrame containing station information (e.g., outlet lat, lon and ID) & circular statistics for all stations

---

## reli_index

```python
def reli_index(Qobs, Qfc_ens, min_obs):
```

Calculates the reliability index.
A measure of the average agreement between the predictive distribution (from the ensemble forecasts) & the observations.
Quantifies the closeness between the empirical CDF of the forecast with the CDF of a uniform distribution (i.e., flat rank histogram).

### Parameters

- **Qobs**: xarray DataArray containing a timeseries of observed flow values.
- **Qfc_ens**: xarray DataArray containing a timeseries of ensemble flow forecasts.
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

### Returns

- **alpha**: Reliability index value between the ensemble forecast and a uniform distribution.

---

## ROC

```python
def ROC(Qobs, Qfc_ens, percentile, sign, min_obs, bins_thresholds):
```

Function to calculate the Relative Operating Characteristic (ROC) for a given percentile.
It measures the ability of the forecast to discriminate between events (given percentile) and non-events and says something about its resolution.

### Parameters

- **Qobs**: xarray DataArray containing a timeseries of observed flow values.
- **Qfc_ens**: xarray DataArray containing a timeseries of ensemble flow forecasts.
- **percentile**: Float between 0 and 1 to indicate the percentile of the event for which calculations are made (e.g., 0.5 will mean that we look at flows either below or above the median of all observations).
- **sign**: String indicating the side of the percentile to use as a threshold for calculations. It can be 'supeq' for >= given percentile or 'infeq' for <= percentile.
- **min_obs**: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics
- **bins_thresholds**: Numpy array of increasing probability thresholds to make the yes/no decision

### Returns

- **roc_curve**: Pandas DataFrame containing the ROC curve information containing the false alarm rate & hit rate for bins of the data.
- **roc_auc**: ROC area under the curve value.

---

## ROC_plots

```python
def ROC_plots(metric_values, percentile):
```

Plots ROC curves for a given event (i.e., percentile).

### Parameters

- **metric_values**: xarray DataArray containing verification metric values for various target periods and forecast start dates.
- **percentile**: Float between 0 and 1 to indicate the percentile of the event for which calculations are made.

### Returns

- Sub-plots of the ROC curves for each target period (start dates are represented on the same sub-plot for the same target period).

---

## split_sample

```python
def split_sample(original_timeseries):
```

Splits predictor & predictand timeseries data in half for split sample testing (training & testing the forecast model).

### Parameters

- **original_timeseries**: Pandas DataFrame of the combined predictor & predictand timeseries

### Returns

- **train_timeseries**: Pandas DataFrame containing the 1st half of the timeseries used for training the forecast model
- **test_timeseries**: Pandas DataFrame containing the 2nd half of the timeseries used for testing the forecast model

---

## stations_basin_map

```python
def stations_basin_map(basins, basin_id, SWE_stations, P_stations, flag, buffer_km=0):
```

Plots map of SWE and P stations in and around the basin.

### Parameters

- **basins**: Pandas GeoDataFrame of all basins available
- **basin_id**: String of the basin id to plot
- **SWE_stations**: Pandas GeoDataFrame of SWE stations to plot
- **P_stations**: Pandas GeoDataFrame of P stations to plot
- **dem_dir**: String of the path to DEMs
- **flag**: Flag to indicate if no buffer (0) or a buffer (1) should be plotted around the basin
- **buffer_km**: Positive value (in km) of buffer to add around basin shapefile (default=0; i.e., no buffer)

### Returns

- A map of SWE stations and basin shape.

---

## streamflow_peaks_statistics

```python
def streamflow_peaks_statistics(streamflow_data, flag):
```

Identifies the streamflow peaks for a given method (i.e., flag).

### Parameters

- **streamflow_data**: Pandas DataFrame of the daily streamflow observations for a basin
- **flag**: An integer of 1, 2 or 3 defining the method to be used for identifying streamflow peaks
  -> flag=1: streamflow annual maxima
  -> flag=2: peak over threshold (POT) where the threshold = minimum value of all annual maxima
  -> flag=3: annual centres of mass (i.e., doy where 1/2 of the total water year streamflow has passed through the river - see: https://journals.ametsoc.org/view/journals/clim/18/2/jcli-3272.1.xml)

### Returns

- **streamflow_stats**: Pandas DataFrame of the streamflow peaks statistics
