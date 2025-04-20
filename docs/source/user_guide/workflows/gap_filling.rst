Gap Filling Workflow
===================

This guide explains the gap filling workflow for the Snow Drought Index package.

Purpose
-------

The gap filling workflow is a critical step in preparing SWE data for analysis. It involves:

1. Loading SWE data with missing values
2. Exploring the extent of missing data
3. Filling gaps using quantile mapping
4. Evaluating the performance of the gap filling method
5. Saving the gap-filled data for use in subsequent analyses

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Completed the :doc:`data preparation workflow <data_preparation>`
- Prepared SWE data in NetCDF format with identified gaps

Workflow Steps
-------------

Step 1: Data Loading
^^^^^^^^^^^^^^^^^^^

First, load the SWE data that needs gap filling:

.. code-block:: python

    from snowdroughtindex.core import data_preparation, gap_filling
    
    # Define data path
    swe_path = 'path/to/SWE_data.nc'
    
    # Load data
    swe_data = data_preparation.load_swe_data(swe_path)
    
    # Convert to DataFrame for gap filling
    swe_df = data_preparation.preprocess_swe(swe_data)
    
    # Set the index to time for time-series operations
    if 'time' in swe_df.columns:
        swe_df = swe_df.set_index('time')

Step 2: Data Exploration
^^^^^^^^^^^^^^^^^^^^^^^

Explore the data to understand the extent of missing values:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Count missing values per station
    missing_values = swe_df.isna().sum()
    
    # Calculate percentage of missing values per station
    missing_percentage = (missing_values / len(swe_df)) * 100
    
    # Display stations with missing data
    print("Stations with missing data:")
    print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))
    
    # Plot missing data percentage
    plt.figure(figsize=(12, 6))
    missing_percentage[missing_percentage > 0].sort_values(ascending=False).plot(kind='bar')
    plt.title('Percentage of Missing Values by Station')
    plt.ylabel('Missing Values (%)')
    plt.xlabel('Station ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

Step 3: Gap Filling Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define parameters for the gap filling process:

.. code-block:: python

    # Parameters for gap filling
    window_days = 15  # Number of days to select data for around a certain day of year
    min_obs_corr = 10  # Minimum number of overlapping observations required to calculate correlation
    min_obs_cdf = 5  # Minimum number of stations required to calculate a station's CDF
    min_corr = 0.7  # Minimum correlation value required to keep a donor station

Step 4: Perform Gap Filling
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the quantile mapping method to fill gaps in the SWE data:

.. code-block:: python

    # Perform gap filling
    gapfilled_data, data_type_flags, donor_stationIDs = gap_filling.qm_gap_filling(
        swe_df, window_days, min_obs_corr, min_obs_cdf, min_corr
    )
    
    # Display summary of gap filling results
    filled_gaps = (data_type_flags == 1).sum().sum()
    total_gaps = swe_df.isna().sum().sum()
    print(f"Total gaps in original data: {total_gaps}")
    print(f"Gaps filled: {filled_gaps}")
    print(f"Percentage of gaps filled: {filled_gaps / total_gaps * 100:.2f}%")

Step 5: Visualize Gap Filling Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the results of the gap filling process:

.. code-block:: python

    # Select stations with filled gaps for visualization
    stations_with_filled_gaps = data_type_flags.sum()[data_type_flags.sum() > 0].sort_values(ascending=False).index[:3]
    
    # Plot original and gap-filled data for selected stations
    for station in stations_with_filled_gaps:
        plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.plot(swe_df.index, swe_df[station], 'b-', label='Original Data')
        
        # Plot gap-filled data
        filled_mask = data_type_flags[station] == 1
        plt.scatter(gapfilled_data.loc[filled_mask].index, 
                    gapfilled_data.loc[filled_mask, station], 
                    color='r', marker='o', label='Gap-Filled Data')
        
        plt.title(f'Gap Filling Results for Station {station}')
        plt.xlabel('Date')
        plt.ylabel('SWE (mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

Step 6: Evaluate Gap Filling Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate the performance of the gap filling method using artificial gaps:

.. code-block:: python

    # Parameters for artificial gap filling evaluation
    iterations = 3  # Number of iterations for artificial gap filling
    artificial_gap_perc = 20  # Percentage of data to remove for artificial gap filling
    min_obs_KGE = 5  # Minimum number of observations for KGE calculation
    
    # Perform artificial gap filling evaluation
    evaluation = gap_filling.artificial_gap_filling(
        swe_df, iterations, artificial_gap_perc, window_days, 
        min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag=0
    )
    
    # Plot evaluation results
    evaluation_plot = gap_filling.plots_artificial_gap_evaluation(evaluation)
    plt.show()

Step 7: Save Gap-Filled Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the gap-filled data for use in subsequent analyses:

.. code-block:: python

    import xarray as xr
    
    # Convert gap-filled data back to xarray Dataset
    gapfilled_dataset = xr.Dataset.from_dataframe(gapfilled_data)
    
    # Save gap-filled data
    gapfilled_dataset.to_netcdf('path/to/swe_gapfilled.nc')
    
    # Save data type flags and donor station IDs for reference
    data_type_flags.to_csv('path/to/data_type_flags.csv')
    donor_stationIDs.to_csv('path/to/donor_stationIDs.csv')
    
    print("Gap-filled data and metadata saved successfully.")

Key Functions
------------

The gap filling workflow uses the following key functions from the ``gap_filling`` module:

- ``qm_gap_filling()`` for filling gaps in the data using quantile mapping
- ``artificial_gap_filling()`` for evaluating the performance of the gap filling method
- ``plots_artificial_gap_evaluation()`` for visualizing the evaluation results

Parameters Explanation
---------------------

- **window_days**: Number of days to select data for around a certain day of year. A larger window provides more data for correlation calculation but may reduce seasonal specificity.
- **min_obs_corr**: Minimum number of overlapping observations required to calculate correlation between stations. Higher values provide more reliable correlation estimates but may reduce the number of potential donor stations.
- **min_obs_cdf**: Minimum number of stations required to calculate a station's cumulative distribution function (CDF). Higher values provide more reliable CDF estimates but may reduce the number of stations that can be gap-filled.
- **min_corr**: Minimum correlation value required to keep a donor station. Higher values ensure that only highly correlated stations are used as donors but may reduce the number of gaps that can be filled.
- **iterations**: Number of iterations for artificial gap filling evaluation. Higher values provide more robust evaluation results but increase computation time.
- **artificial_gap_perc**: Percentage of data to remove for artificial gap filling evaluation. Higher values provide a more challenging test but may reduce the reliability of the evaluation.
- **min_obs_KGE**: Minimum number of observations required to calculate the Kling-Gupta Efficiency (KGE) metric. Higher values provide more reliable KGE estimates but may reduce the number of stations that can be evaluated.

Example Notebook
---------------

For a complete example of the gap filling workflow, refer to the 
`gap_filling_workflow.ipynb <https://github.com/yourusername/snowdroughtindex/blob/main/notebooks/workflows/gap_filling_workflow.ipynb>`_ 
notebook in the package repository.

Next Steps
---------

After completing the gap filling workflow, you can proceed to:

- :doc:`SSWEI calculation workflow <sswei_calculation>` to calculate the Standardized Snow Water Equivalent Index
- :doc:`Drought classification workflow <drought_classification>` to classify drought conditions based on SSWEI values
