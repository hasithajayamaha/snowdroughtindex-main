SSWEI Calculation Workflow
========================

This guide explains the SSWEI (Standardized Snow Water Equivalent Index) calculation workflow for the Snow Drought Index package.

Purpose
-------

The SSWEI calculation workflow is a key step in analyzing snow drought conditions. It involves:

1. Loading gap-filled SWE data
2. Calculating basin-wide mean SWE values
3. Preparing seasonal data
4. Calculating seasonal integration of SWE
5. Computing SSWEI values
6. Classifying drought conditions
7. Analyzing and visualizing results

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Completed the :doc:`data preparation workflow <data_preparation>`
- Completed the :doc:`gap filling workflow <gap_filling>` to obtain gap-filled SWE data

Workflow Steps
-------------

Step 1: Data Loading
^^^^^^^^^^^^^^^^^^^

First, load the gap-filled SWE data:

.. code-block:: python

    from snowdroughtindex.core import data_preparation, sswei
    import matplotlib.pyplot as plt
    
    # Define data path
    swe_path = 'path/to/swe_gapfilled.nc'
    
    # Load data
    swe_data = data_preparation.load_swe_data(swe_path)
    
    # Convert to DataFrame for processing
    swe_df = data_preparation.preprocess_swe(swe_data)
    
    # Ensure the DataFrame has a time index
    if 'time' in swe_df.columns:
        swe_df = swe_df.set_index('time')

Step 2: Calculate Basin-Wide Mean SWE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the daily mean SWE across all stations in the basin:

.. code-block:: python

    # Identify SWE columns (excluding metadata columns)
    swe_columns = [col for col in swe_df.columns if col not in ['station_id', 'lat', 'lon', 'elevation']]
    
    # Calculate daily mean SWE across all stations
    daily_mean = pd.DataFrame({
        'date': swe_df.index,
        'mean_SWE': swe_df[swe_columns].mean(axis=1)
    })

Step 3: Visualize Daily Mean SWE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the daily mean SWE values to understand the seasonal patterns:

.. code-block:: python

    # Plot the daily mean SWE values
    plt.figure(figsize=(12, 6))
    plt.plot(daily_mean['date'], daily_mean['mean_SWE'])
    plt.xlabel('Date')
    plt.ylabel('Mean SWE (mm)')
    plt.title('Daily Mean SWE Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Step 4: Prepare Seasonal Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prepare the seasonal data by filtering for complete snow seasons:

.. code-block:: python

    # Define season parameters
    start_month = 11  # November
    start_day = 1
    end_month = 4    # April
    end_day = 30
    min_swe = 15     # Minimum SWE to consider as the start of the snow season
    
    # Prepare seasonal data
    season_data = sswei.prepare_season_data(
        daily_mean, 
        start_month=start_month, 
        start_day=start_day, 
        end_month=end_month, 
        end_day=end_day, 
        min_swe=min_swe
    )

Step 5: Calculate Seasonal Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the seasonal integration of SWE values:

.. code-block:: python

    # Calculate seasonal integration
    integrated_data_season = sswei.calculate_seasonal_integration(
        season_data, 
        start_month=start_month
    )

Step 6: Calculate SSWEI
^^^^^^^^^^^^^^^^^^^^^

Calculate the SSWEI values and classify drought conditions:

.. code-block:: python

    # Calculate SSWEI directly from daily mean data
    sswei_results = sswei.calculate_sswei(
        daily_mean, 
        start_month=start_month, 
        start_day=start_day, 
        end_month=end_month, 
        end_day=end_day, 
        min_swe=min_swe
    )
    
    # Display the SSWEI results
    sswei_results[['season_year', 'total_SWE_integration', 'Gringorten_probabilities', 'SWEI', 'Drought_Classification']]

Step 7: Visualize SSWEI Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the SSWEI values and drought classifications:

.. code-block:: python

    # Plot SSWEI values with drought classification thresholds
    sswei_plot = sswei.plot_sswei(sswei_results)
    plt.show()

Step 8: Analyze Drought Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the drought conditions over the years:

.. code-block:: python

    # Count occurrences of each drought classification
    drought_counts = sswei_results['Drought_Classification'].value_counts()
    
    # Plot the counts
    plt.figure(figsize=(10, 6))
    drought_counts.plot(kind='bar', color='skyblue')
    plt.title('Frequency of Drought Classifications')
    plt.xlabel('Drought Classification')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate percentage of each classification
    drought_percentage = (drought_counts / len(sswei_results) * 100).round(1)
    print("Percentage of each drought classification:")
    for classification, percentage in drought_percentage.items():
        print(f"{classification}: {percentage}%")

Step 9: Identify Drought Years
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identify the years with drought conditions:

.. code-block:: python

    # Filter for drought years (SWEI < -0.5)
    drought_years = sswei_results[sswei_results['SWEI'] < -0.5]
    
    # Sort by SWEI to see the most severe droughts first
    drought_years = drought_years.sort_values('SWEI')
    
    # Display drought years
    print("Years with drought conditions (SWEI < -0.5):")
    drought_years[['season_year', 'SWEI', 'Drought_Classification']]

Step 10: Save SSWEI Results
^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the SSWEI results for future analysis:

.. code-block:: python

    # Save SSWEI results to CSV
    sswei_results.to_csv('path/to/sswei_results.csv', index=False)

Key Functions
------------

The SSWEI calculation workflow uses the following key functions from the ``sswei`` module:

- ``prepare_season_data()`` for filtering complete snow seasons
- ``calculate_seasonal_integration()`` for integrating SWE values over the season
- ``calculate_sswei()`` for calculating SSWEI values and classifying drought conditions
- ``plot_sswei()`` for visualizing SSWEI values with drought classification thresholds

Parameters Explanation
---------------------

- **start_month**, **start_day**: The month and day to start the snow season (e.g., November 1)
- **end_month**, **end_day**: The month and day to end the snow season (e.g., April 30)
- **min_swe**: Minimum SWE value to consider as the start of the snow season. This helps filter out noise and ensures that only significant snow accumulation is considered.

SSWEI Interpretation
------------------

The SSWEI values are interpreted as follows:

- **SSWEI > 1.5**: Extremely wet
- **1.0 < SSWEI ≤ 1.5**: Moderately wet
- **0.5 < SSWEI ≤ 1.0**: Slightly wet
- **-0.5 ≤ SSWEI ≤ 0.5**: Near normal
- **-1.0 ≤ SSWEI < -0.5**: Slight drought
- **-1.5 ≤ SSWEI < -1.0**: Moderate drought
- **SSWEI < -1.5**: Extreme drought

These classifications help identify the severity of snow drought conditions and can be used to compare different years or regions.

Example Notebook
---------------

For a complete example of the SSWEI calculation workflow, refer to the 
`sswei_calculation_workflow.ipynb <https://github.com/yourusername/snowdroughtindex/blob/main/notebooks/workflows/sswei_calculation_workflow.ipynb>`_ 
notebook in the package repository.

Next Steps
---------

After completing the SSWEI calculation workflow, you can proceed to:

- :doc:`Drought classification workflow <drought_classification>` for more detailed drought classification
- :doc:`SCS analysis workflow <scs_analysis>` for analyzing snow-to-precipitation ratios
- :doc:`Case study workflow <case_study>` for applying the SSWEI to specific case studies
