Heldmyer 2024 Classification
==========================

This guide explains how to implement the Heldmyer et al. (2024) snow drought classification methodology using the Snow Drought Index package.

Overview
--------

The Heldmyer et al. (2024) classification approach provides a more nuanced way to categorize snow drought types based on both Snow Water Equivalent (SWE) and precipitation data. This classification distinguishes between different types of snow drought:

1. **Dry snow drought**: Low precipitation leads to low SWE
2. **Warm snow drought**: Normal/high precipitation but low SWE due to warm temperatures
3. **Warm & dry snow drought**: Combination of precipitation deficit and warm temperatures

This workflow demonstrates how to:

- Implement the Heldmyer et al. classification methodology
- Use K-means clustering to identify different snow drought types
- Visualize the classification results
- Compare results with traditional drought indices

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Snow Water Equivalent (SWE) data for your study area
- Precipitation data for the same area and time period
- Python packages: numpy, pandas, matplotlib, scikit-learn

Workflow Steps
-------------

1. Data Preparation
^^^^^^^^^^^^^^^^^

First, load and prepare your SWE and precipitation data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    # Import snowdroughtindex package
    from snowdroughtindex.core import data_preparation, drought_classification
    
    # Load SWE and precipitation data
    swe_data = pd.read_csv('path/to/swe_data.csv')
    precip_data = pd.read_csv('path/to/precipitation_data.csv')
    
    # Ensure data is in the correct format with date columns as datetime
    swe_data['date'] = pd.to_datetime(swe_data['date'])
    precip_data['date'] = pd.to_datetime(precip_data['date'])

2. Calculate Seasonal Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the seasonal metrics needed for the Heldmyer classification:

.. code-block:: python

    # Define water years
    def assign_water_year(date):
        if date.month >= 10:
            return date.year + 1
        else:
            return date.year
    
    swe_data['water_year'] = swe_data['date'].apply(assign_water_year)
    precip_data['water_year'] = precip_data['date'].apply(assign_water_year)
    
    # Calculate seasonal metrics for each water year
    seasonal_metrics = []
    
    for year in sorted(swe_data['water_year'].unique()):
        # Filter data for the current water year
        year_swe = swe_data[swe_data['water_year'] == year]
        year_precip = precip_data[precip_data['water_year'] == year]
        
        # Calculate metrics
        max_swe = year_swe['swe'].max()
        mean_swe = year_swe['swe'].mean()
        total_precip = year_precip['precip'].sum()
        mean_precip = year_precip['precip'].mean()
        
        # Calculate SWE to precipitation ratio
        swe_p_ratio = mean_swe / total_precip if total_precip > 0 else np.nan
        
        # Store metrics
        seasonal_metrics.append({
            'water_year': year,
            'max_swe': max_swe,
            'mean_swe': mean_swe,
            'total_precip': total_precip,
            'mean_precip': mean_precip,
            'swe_p_ratio': swe_p_ratio
        })
    
    # Convert to DataFrame
    seasonal_df = pd.DataFrame(seasonal_metrics)

3. Identify Snow Drought Years
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identify snow drought years based on below-average maximum SWE:

.. code-block:: python

    # Calculate average maximum SWE for the reference period
    reference_period = seasonal_df[(seasonal_df['water_year'] >= 1981) & 
                                  (seasonal_df['water_year'] <= 2010)]
    avg_max_swe = reference_period['max_swe'].mean()
    
    # Identify snow drought years (years with max SWE below average)
    snow_drought_years = seasonal_df[seasonal_df['max_swe'] < avg_max_swe].copy()
    normal_years = seasonal_df[seasonal_df['max_swe'] >= avg_max_swe].copy()
    
    print(f"Number of snow drought years identified: {len(snow_drought_years)}")
    print(f"Number of normal years: {len(normal_years)}")

4. Calculate Precipitation Anomalies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate precipitation anomalies for each year:

.. code-block:: python

    # Calculate precipitation anomalies
    avg_precip = seasonal_df['total_precip'].mean()
    
    snow_drought_years['precip_anomaly'] = snow_drought_years['total_precip'] - avg_precip
    normal_years['precip_anomaly'] = normal_years['total_precip'] - avg_precip
    
    # Standardize features for clustering
    snow_drought_years['precip_anomaly_z'] = (snow_drought_years['precip_anomaly'] - 
                                             snow_drought_years['precip_anomaly'].mean()) / 
                                             snow_drought_years['precip_anomaly'].std()
    
    snow_drought_years['swe_p_ratio_z'] = (snow_drought_years['swe_p_ratio'] - 
                                          snow_drought_years['swe_p_ratio'].mean()) / 
                                          snow_drought_years['swe_p_ratio'].std()

5. Apply K-means Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^

Apply K-means clustering to classify snow drought types:

.. code-block:: python

    # Prepare features for clustering
    features = snow_drought_years[['precip_anomaly_z', 'swe_p_ratio_z']].dropna()
    
    # Apply K-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # Store the water years for the rows used in clustering
    years_for_clustering = snow_drought_years.loc[features.index, 'water_year'].values
    
    # Fit the model and predict cluster labels
    clusters = kmeans.fit_predict(features)
    
    # Create a mapping from water years to cluster labels
    year_to_cluster = dict(zip(years_for_clustering, clusters))
    
    # Add cluster labels back to the original DataFrame
    snow_drought_years['cluster'] = snow_drought_years['water_year'].map(year_to_cluster)
    
    # Print cluster centers
    print("Cluster centers:")
    print(kmeans.cluster_centers_)

6. Assign Drought Type Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assign meaningful labels to the clusters based on their characteristics:

.. code-block:: python

    # Analyze cluster centers to determine drought types
    centers = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=['precip_anomaly_z', 'swe_p_ratio_z']
    )
    
    # Determine cluster types based on centers
    # This is a simplified approach - you may need to adjust based on your results
    cluster_types = {}
    
    for i, (precip_anom, swe_p_ratio) in centers.iterrows():
        if precip_anom < -0.3 and swe_p_ratio < -0.3:
            cluster_types[i] = 'Warm & Dry'
        elif precip_anom < -0.3:
            cluster_types[i] = 'Dry'
        elif swe_p_ratio < -0.3:
            cluster_types[i] = 'Warm'
        else:
            cluster_types[i] = 'Mild'
    
    # Map cluster numbers to types
    snow_drought_years['drought_type'] = snow_drought_years['cluster'].map(cluster_types)
    
    # Display results
    print("Snow drought classification results:")
    print(snow_drought_years[['water_year', 'max_swe', 'total_precip', 
                             'swe_p_ratio', 'precip_anomaly', 'drought_type']])

7. Visualize Classification Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a scatter plot to visualize the different drought types:

.. code-block:: python

    # Define colors for each drought type
    colors = {
        'Warm': 'red',
        'Dry': 'blue',
        'Warm & Dry': 'purple',
        'Mild': 'orange'
    }
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot normal years
    plt.scatter(
        normal_years['precip_anomaly'], 
        normal_years['swe_p_ratio'],
        c='gray', alpha=0.5, label='Normal'
    )
    
    # Plot drought years by type
    for drought_type, color in colors.items():
        mask = snow_drought_years['drought_type'] == drought_type
        if mask.any():
            plt.scatter(
                snow_drought_years.loc[mask, 'precip_anomaly'],
                snow_drought_years.loc[mask, 'swe_p_ratio'],
                c=color, label=drought_type
            )
    
    # Add year labels to drought points
    for i, row in snow_drought_years.iterrows():
        if pd.notna(row['drought_type']):
            plt.annotate(
                str(int(row['water_year'])),
                (row['precip_anomaly'], row['swe_p_ratio']),
                fontsize=8
            )
    
    # Add reference lines
    plt.axhline(y=snow_drought_years['swe_p_ratio'].mean(), color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Precipitation Anomaly (mm)')
    plt.ylabel('SWE/P Ratio')
    plt.title('Heldmyer et al. Snow Drought Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()

8. Compare with Traditional Drought Indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare the Heldmyer classification with traditional drought indices like SSWEI:

.. code-block:: python

    # Load SSWEI data if available
    try:
        sswei_data = pd.read_csv('path/to/sswei_results.csv')
        
        # Merge with snow drought classification
        comparison = pd.merge(
            snow_drought_years[['water_year', 'drought_type']],
            sswei_data[['season_year', 'SWEI', 'Drought_Classification']],
            left_on='water_year',
            right_on='season_year',
            how='inner'
        )
        
        # Display comparison
        print("Comparison of Heldmyer classification with SSWEI:")
        print(comparison[['water_year', 'drought_type', 'SWEI', 'Drought_Classification']])
        
        # Create a confusion matrix-like table
        confusion = pd.crosstab(
            comparison['drought_type'], 
            comparison['Drought_Classification'],
            margins=True,
            margins_name='Total'
        )
        
        print("\nCross-tabulation of drought classifications:")
        print(confusion)
        
    except FileNotFoundError:
        print("SSWEI data not found. Skipping comparison.")

9. Save Results
^^^^^^^^^^^^^

Save the classification results for future reference:

.. code-block:: python

    # Save classification results
    snow_drought_years.to_csv('path/to/heldmyer_classification_results.csv', index=False)
    print("Classification results saved to 'path/to/heldmyer_classification_results.csv'")
    
    # Save the figure if needed
    plt.savefig('path/to/heldmyer_classification_plot.png', dpi=300, bbox_inches='tight')
    print("Classification plot saved to 'path/to/heldmyer_classification_plot.png'")

Interpretation of Results
------------------------

The Heldmyer et al. classification provides insights into the mechanisms behind snow drought:

1. **Dry Snow Drought**: Characterized by negative precipitation anomalies and relatively normal SWE/P ratios. These droughts are primarily caused by a lack of precipitation.

2. **Warm Snow Drought**: Characterized by normal or positive precipitation anomalies but low SWE/P ratios. These droughts occur when precipitation falls as rain instead of snow due to warm temperatures.

3. **Warm & Dry Snow Drought**: Characterized by negative precipitation anomalies and low SWE/P ratios. These droughts are caused by a combination of precipitation deficit and warm temperatures.

This classification approach helps water resource managers understand the underlying causes of snow drought, which can inform more targeted adaptation strategies.

Complete Example
---------------

A complete example notebook is available in the package repository:

``notebooks/workflows/2_Heldmyer_2024_classification.ipynb``

This notebook demonstrates the full workflow with example data.

References
---------

Heldmyer, A. J., Livneh, B., Molotch, N. P., & Harpold, A. A. (2024). Classifying snow drought types: A new approach to understanding snow drought mechanisms. *Journal of Hydrometeorology*.

Next Steps
---------

After completing this workflow, you can:

- Compare the Heldmyer classification with other drought indices
- Analyze the frequency of different snow drought types in your region
- Investigate the relationship between drought types and climate variables
- Develop region-specific drought monitoring approaches
- Apply the classification to future climate scenarios
