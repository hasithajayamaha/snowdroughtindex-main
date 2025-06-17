# Heldmyer Classification Methodology

## Overview

The Heldmyer et al. (2024) classification methodology provides a nuanced approach to categorizing snow drought types based on both Snow Water Equivalent (SWE) and precipitation data. This methodology uses K-means clustering to identify different types of snow drought based on precipitation anomalies and SWE-to-precipitation ratios.

## Snow Drought Types

The Heldmyer classification distinguishes between three primary types of snow drought:

1. **Dry Snow Drought**: Characterized by negative precipitation anomalies and relatively normal SWE/P ratios. These droughts are primarily caused by a lack of precipitation.

2. **Warm Snow Drought**: Characterized by normal or positive precipitation anomalies but low SWE/P ratios. These droughts occur when precipitation falls as rain instead of snow due to warm temperatures.

3. **Warm & Dry Snow Drought**: Characterized by negative precipitation anomalies and low SWE/P ratios. These droughts are caused by a combination of precipitation deficit and warm temperatures.

## Methodology Steps

### 1. Data Preparation

- Collect daily SWE and precipitation data for the study area
- Calculate water years (typically October 1 to September 30)
- Calculate seasonal metrics for each water year:
  - Maximum SWE
  - Mean SWE
  - Total precipitation
  - Mean precipitation
  - SWE-to-precipitation ratio (SWE/P)

### 2. Identify Snow Drought Years

- Calculate the average maximum SWE for a reference period (typically 1981-2010)
- Identify snow drought years as those with maximum SWE below the reference period average

### 3. Calculate Precipitation Anomalies

- Calculate precipitation anomalies for each year (difference from the mean)
- Standardize precipitation anomalies and SWE/P ratios for clustering

### 4. Apply K-means Clustering

- Apply K-means clustering with 3 clusters to the standardized precipitation anomalies and SWE/P ratios
- Analyze cluster centers to determine drought types:
  - Cluster with negative precipitation anomaly and normal SWE/P ratio: Dry snow drought
  - Cluster with normal/positive precipitation anomaly and low SWE/P ratio: Warm snow drought
  - Cluster with negative precipitation anomaly and low SWE/P ratio: Warm & dry snow drought

### 5. Assign Drought Type Labels

- Assign meaningful labels to each cluster based on their characteristics
- Map these labels back to the original data

## Visualization

The Heldmyer classification is typically visualized using a scatter plot with:
- X-axis: Precipitation anomaly
- Y-axis: SWE/P ratio
- Points colored by drought type
- Reference lines at the mean SWE/P ratio and zero precipitation anomaly

## Advantages

The Heldmyer classification offers several advantages over traditional drought indices:

1. **Mechanistic Understanding**: Provides insight into the underlying causes of snow drought (lack of precipitation, warm temperatures, or both)
2. **Targeted Management**: Enables more targeted water resource management strategies based on drought type
3. **Climate Change Context**: Helps understand how climate change may affect different types of snow drought
4. **Regional Applicability**: Can be applied to different regions with varying climate characteristics

## Implementation

The Snow Drought Index package provides tools for implementing the Heldmyer classification methodology. For a detailed implementation guide, refer to the [Heldmyer Classification Workflow](../source/user_guide/workflows/heldmyer_classification.rst).

## References

Heldmyer, A. J., Livneh, B., Molotch, N. P., & Harpold, A. A. (2024). Classifying snow drought types: A new approach to understanding snow drought mechanisms. *Journal of Hydrometeorology*.
