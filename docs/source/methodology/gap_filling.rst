Gap Filling Methodology
=======================

This document provides a detailed explanation of the gap filling methodology implemented in the Snow Drought Index package.

Overview
--------

Gap filling is a critical preprocessing step in snow water equivalent (SWE) data analysis, as SWE observations often contain missing values due to various reasons such as instrument failures, maintenance issues, or data transmission problems. The Snow Drought Index package implements a robust gap filling methodology based on quantile mapping, which leverages spatial correlations between stations to fill in missing values.

Methodology
-----------

The gap filling process consists of several steps:

1. **Linear Interpolation**: Small gaps in the data are filled using linear interpolation.
2. **Quantile Mapping**: Larger gaps are filled using quantile mapping, which leverages donor stations with high correlations to the target station.
3. **Evaluation**: The gap filling performance is evaluated using artificial gap filling.

Linear Interpolation
~~~~~~~~~~~~~~~~~~~

Linear interpolation is used to fill small gaps in the data (typically less than 15 days). This is implemented using the ``interpolate_na`` method from xarray:

.. code-block:: python

   SWE_obs_basin_interp_da = SWE_stations_ds.snw.interpolate_na(
       method='linear', 
       dim='time', 
       max_gap=datetime.timedelta(days=max_gap_days)
   )

Quantile Mapping
~~~~~~~~~~~~~~~

For larger gaps, the package uses quantile mapping, which leverages donor stations with high correlations to the target station. The quantile mapping process consists of the following steps:

1. **Calculate Correlations**: Calculate correlations between stations for each day of the year (doy) using a window of days centered around the doy.
2. **Identify Donor Stations**: For each missing value, identify potential donor stations with correlations above a minimum threshold.
3. **Build Empirical CDFs**: Build empirical cumulative distribution functions (CDFs) for the target and donor stations.
4. **Map Quantiles**: Map the donor station's value to the target station's CDF to estimate the missing value.

Calculate Correlations
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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

Quantile Mapping Function
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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

Gap Filling Process
^^^^^^^^^^^^^^^^^^

The main gap filling function processes all missing observations using the quantile mapping approach. The process involves:

1. **Data Preparation**: Create copies of the dataset and set up tracking dataframes
2. **Correlation Calculation**: Calculate day-of-year correlations between stations
3. **Gap Identification**: Loop through dates and identify missing values
4. **Donor Selection**: For each missing value, find suitable donor stations
5. **Quantile Mapping**: Apply quantile mapping to estimate missing values

.. code-block:: python

   def qm_gap_filling(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr):
       """Performing the gap filling for all missing observations (when possible) using quantile mapping."""
       # Create a duplicate of the dataset to gap fill
       gapfilled_data = original_data.copy()
       
       # Process each missing observation using quantile mapping
       # [Implementation details as shown in the full methodology]
       
       return gapfilled_data, data_type_flags, donor_stationIDs

Evaluation
----------

The gap filling performance is evaluated using artificial gap filling, which involves creating artificial gaps in the data and then evaluating how well the gap filling methodology can fill these gaps. This provides metrics such as:

- **RMSE**: Root Mean Square Error
- **KGE''**: Modified Kling-Gupta Efficiency
- **Correlation**: Pearson correlation coefficient
- **Bias**: Systematic bias in predictions
- **Variability**: Variance ratio

Parameters
----------

The gap filling process can be customized using several parameters:

.. list-table:: Gap Filling Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``window_days``
     - 7
     - Number of days around a DOY for correlation calculation
   * - ``min_obs_corr``
     - 3
     - Minimum overlapping observations for correlation calculation
   * - ``min_obs_cdf``
     - 10
     - Minimum observations required to build a CDF
   * - ``min_corr``
     - 0.6
     - Minimum correlation threshold for donor stations
   * - ``min_obs_KGE``
     - 10
     - Minimum observations for KGE calculation
   * - ``max_gap_days``
     - 15
     - Maximum gap size for linear interpolation

Visualization
------------

The package provides several functions to visualize the gap filling results:

- **data_availability_monthly_plots_1**: Plots the percentage of SWE stations available on the first day of each month of each year, before and after gap filling.
- **data_availability_monthly_plots_2**: Creates bar chart subplots of the days with SWE observations around the 1st day of each month.
- **plots_artificial_gap_evaluation**: Plots the evaluation results for the artificial gap filling.

Performance Optimization
-----------------------

For large datasets, the package includes optimized implementations:

- **Chunked Processing**: Process stations in chunks to manage memory usage
- **Pre-computed Correlations**: Cache correlation calculations for reuse
- **Vectorized Operations**: Use NumPy vectorization where possible
- **Progress Monitoring**: Track progress during long operations

Usage Example
------------

Here's a complete example of using the gap filling functionality:

.. code-block:: python

   import pandas as pd
   import xarray as xr
   from snowdroughtindex.core.gap_filling import qm_gap_filling
   
   # Load SWE data
   swe_data = pd.read_csv('swe_data.csv', index_col='date', parse_dates=True)
   
   # Set gap filling parameters
   window_days = 7
   min_obs_corr = 3
   min_obs_cdf = 10
   min_corr = 0.6
   
   # Perform gap filling
   gapfilled_data, flags, donors = qm_gap_filling(
       swe_data, 
       window_days, 
       min_obs_corr, 
       min_obs_cdf, 
       min_corr
   )
   
   # Evaluate gap filling performance
   from snowdroughtindex.core.gap_filling import artificial_gap_filling
   
   evaluation = artificial_gap_filling(
       swe_data,
       iterations=10,
       artificial_gap_perc=20,
       window_days=window_days,
       min_obs_corr=min_obs_corr,
       min_obs_cdf=min_obs_cdf,
       min_corr=min_corr,
       min_obs_KGE=10,
       flag=1
   )

References
----------

- Tang, G., Clark, M. P., & Papalexiou, S. M. (2021). SC-earth: A station-based serially complete earth dataset from 1950 to 2019. *Journal of Climate*, 34(16), 6493-6511.

.. seealso::
   
   - :doc:`sswei` for SSWEI methodology
   - :doc:`../user_guide/workflows/gap_filling` for practical gap filling workflows
   - :doc:`../user_guide/performance_optimization` for optimization techniques
