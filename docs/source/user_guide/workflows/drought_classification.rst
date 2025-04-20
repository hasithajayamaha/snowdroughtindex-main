Drought Classification Workflow
============================

This guide explains the drought classification workflow for the Snow Drought Index package.

Purpose
-------

The drought classification workflow provides detailed analysis of drought conditions based on SSWEI values. It involves:

1. Loading SSWEI data
2. Classifying drought conditions with configurable thresholds
3. Visualizing drought classifications
4. Calculating drought characteristics (duration, severity, intensity)
5. Analyzing drought trends over time
6. Comparing drought classifications across different time periods
7. Analyzing the distribution of drought severity

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Completed the :doc:`SSWEI calculation workflow <sswei_calculation>` to obtain SSWEI values

Workflow Steps
-------------

Step 1: Load SSWEI Data
^^^^^^^^^^^^^^^^^^^^^

First, load the SSWEI data calculated in the previous workflow:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from snowdroughtindex.core import drought_classification
    
    # Load SSWEI data from CSV file
    sswei_data = pd.read_csv('path/to/sswei_results.csv')

Step 2: Classify Drought Conditions with Configurable Thresholds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The drought classification module allows for configurable thresholds:

.. code-block:: python

    # Define custom thresholds
    custom_thresholds = {
        "exceptional": -1.8,  # More severe threshold for exceptional drought
        "extreme": -1.3,      # More severe threshold for extreme drought
        "severe": -0.8,       # More severe threshold for severe drought
        "moderate": -0.4,     # Less severe threshold for moderate drought
        "normal_lower": -0.4, # Lower bound for normal conditions
        "normal_upper": 0.4,  # Upper bound for normal conditions
        "abnormally_wet": 0.8, # Threshold for abnormally wet conditions
        "moderately_wet": 1.3, # Threshold for moderately wet conditions
        "very_wet": 1.8       # Threshold for very wet conditions
    }
    
    # Apply custom classification
    sswei_data['Custom_Classification'] = sswei_data['SWEI'].apply(
        lambda x: drought_classification.classify_drought(x, custom_thresholds)
    )
    
    # Compare default and custom classifications
    comparison = sswei_data[['season_year', 'SWEI', 'Drought_Classification', 'Custom_Classification']]

Step 3: Visualize Drought Classifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the drought classifications:

.. code-block:: python

    # Plot default drought classification
    fig1 = drought_classification.plot_drought_classification(
        sswei_data,
        year_column='season_year',
        swei_column='SWEI',
        classification_column='Drought_Classification'
    )
    plt.title('Default Drought Classification')
    plt.show()
    
    # Plot custom drought classification
    fig2 = drought_classification.plot_drought_classification(
        sswei_data,
        year_column='season_year',
        swei_column='SWEI',
        classification_column='Custom_Classification'
    )
    plt.title('Custom Drought Classification')
    plt.show()

Step 4: Calculate Drought Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate drought characteristics such as duration, severity, and intensity:

.. code-block:: python

    # Calculate drought characteristics using default threshold (-0.5)
    drought_chars = drought_classification.calculate_drought_characteristics(
        sswei_data,
        year_column='season_year',
        swei_column='SWEI'
    )
    
    # Display drought characteristics
    if not drought_chars.empty:
        print("Drought Characteristics:")
        print(drought_chars)

Step 5: Visualize Drought Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the drought characteristics:

.. code-block:: python

    # Plot drought characteristics
    if not drought_chars.empty:
        fig3 = drought_classification.plot_drought_characteristics(drought_chars)
        plt.show()

Step 6: Analyze Drought Trends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze drought trends over time using a moving window approach:

.. code-block:: python

    # Define window size for trend analysis
    window_size = 10  # 10-year moving window
    
    # Analyze drought trends
    trend_data = drought_classification.analyze_drought_trends(
        sswei_data,
        year_column='season_year',
        swei_column='SWEI',
        window_size=window_size
    )
    
    # Display trend data
    if not trend_data.empty:
        print(f"Drought Trends (using {window_size}-year moving window):")
        print(trend_data.head())

Step 7: Visualize Drought Trends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the drought trends:

.. code-block:: python

    # Plot drought trends
    if not trend_data.empty:
        fig4 = drought_classification.plot_drought_trends(trend_data)
        plt.show()

Step 8: Compare Drought Classifications by Decade
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze how drought classifications have changed by decade:

.. code-block:: python

    # Add decade column
    sswei_data['decade'] = (sswei_data['season_year'] // 10) * 10
    
    # Count classifications by decade
    decade_counts = pd.crosstab(sswei_data['decade'], sswei_data['Drought_Classification'])
    
    # Display counts
    print("Drought Classifications by Decade:")
    print(decade_counts)
    
    # Plot heatmap of classifications by decade
    plt.figure(figsize=(12, 8))
    sns.heatmap(decade_counts, cmap='YlOrRd', annot=True, fmt='d', cbar_kws={'label': 'Count'})
    plt.title('Drought Classifications by Decade')
    plt.ylabel('Decade')
    plt.xlabel('Drought Classification')
    plt.tight_layout()
    plt.show()

Step 9: Analyze Drought Severity Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the distribution of drought severity values:

.. code-block:: python

    # Calculate drought severity for all years
    sswei_data['drought_severity'] = sswei_data['SWEI'].apply(drought_classification.get_drought_severity)
    
    # Plot histogram of drought severity
    plt.figure(figsize=(10, 6))
    sns.histplot(sswei_data[sswei_data['drought_severity'] > 0]['drought_severity'], bins=10, kde=True)
    plt.title('Distribution of Drought Severity')
    plt.xlabel('Drought Severity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate summary statistics for drought severity
    severity_stats = sswei_data[sswei_data['drought_severity'] > 0]['drought_severity'].describe()
    print("Drought Severity Statistics:")
    print(severity_stats)

Step 10: Save Drought Analysis Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the drought analysis results for future reference:

.. code-block:: python

    # Save drought characteristics
    if not drought_chars.empty:
        drought_chars.to_csv('path/to/drought_characteristics.csv', index=False)
    
    # Save drought trends
    if not trend_data.empty:
        trend_data.to_csv('path/to/drought_trends.csv', index=False)
    
    # Save SSWEI data with custom classification and severity
    sswei_data.to_csv('path/to/sswei_with_drought_analysis.csv', index=False)

Key Functions
------------

The drought classification workflow uses the following key functions from the ``drought_classification`` module:

- ``classify_drought()`` for classifying drought conditions based on SSWEI values
- ``calculate_drought_characteristics()`` for analyzing drought events
- ``analyze_drought_trends()`` for examining changes in drought patterns over time
- ``plot_drought_classification()`` for visualizing drought classifications
- ``plot_drought_characteristics()`` for visualizing drought characteristics
- ``plot_drought_trends()`` for visualizing drought trends
- ``get_drought_severity()`` for calculating drought severity

Drought Characteristics Explanation
---------------------------------

The drought characteristics calculated by the ``calculate_drought_characteristics()`` function include:

- **Duration**: The number of consecutive years with SSWEI values below the drought threshold
- **Severity**: The sum of the absolute SSWEI values during a drought event
- **Intensity**: The average severity per year (severity divided by duration)
- **Start Year**: The first year of the drought event
- **End Year**: The last year of the drought event

These characteristics help quantify drought events and enable comparison between different drought periods.

Drought Trends Analysis
---------------------

The drought trends analysis uses a moving window approach to examine changes in drought patterns over time. For each window, the following metrics are calculated:

- **Frequency**: The percentage of years with drought conditions
- **Average Severity**: The average severity of drought events
- **Average Duration**: The average duration of drought events
- **Maximum Severity**: The maximum severity observed in the window
- **Maximum Duration**: The maximum duration observed in the window

These metrics help identify long-term changes in drought patterns, which may be related to climate change or other factors.

Example Notebook
---------------

For a complete example of the drought classification workflow, refer to the 
`drought_classification_workflow.ipynb <https://github.com/yourusername/snowdroughtindex/blob/main/notebooks/workflows/drought_classification_workflow.ipynb>`_ 
notebook in the package repository.

Next Steps
---------

After completing the drought classification workflow, you can proceed to:

- :doc:`SCS analysis workflow <scs_analysis>` for analyzing snow-to-precipitation ratios
- :doc:`Case study workflow <case_study>` for applying the drought classification to specific case studies
