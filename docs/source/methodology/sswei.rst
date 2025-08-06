Standardized Snow Water Equivalent Index (SSWEI) Methodology
===========================================================

This document provides a detailed explanation of the Standardized Snow Water Equivalent Index (SSWEI) methodology implemented in the Snow Drought Index package.

Overview
--------

The Standardized Snow Water Equivalent Index (SSWEI) is a drought index specifically designed for snow drought analysis, based on the methodology developed by Huning & AghaKouchak (2020). It quantifies the severity of snow drought conditions by standardizing the seasonal snow water equivalent (SWE) accumulation.

Theoretical Background
---------------------

The SSWEI follows a similar approach to other standardized indices like the Standardized Precipitation Index (SPI) and the Standardized Precipitation Evapotranspiration Index (SPEI). The key steps in the SSWEI calculation are:

1. **Seasonal SWE Integration**: Calculate the total SWE accumulation over the snow season (typically November to May in the Northern Hemisphere).
2. **Probability Transformation**: Transform the integrated SWE values to probabilities using the Gringorten plotting position formula.
3. **Standardization**: Convert the probabilities to standardized values using the inverse normal distribution.
4. **Classification**: Classify drought conditions based on the standardized values.

Implementation Details
---------------------

1. Seasonal SWE Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The seasonal SWE integration is calculated using the trapezoidal rule to integrate the SWE values over the snow season:

.. code-block:: python

   def integrate_season(group):
       """Integrates SWE values from November 1st to May 1st."""
       # Ensure dates are sorted
       group = group.sort_values(by='date')
       # Convert dates to numerical days since start of the season
       days_since_start = (group['date'] - group['date'].min()).dt.days
       # Integrate SWE over the period
       total_swe_integration = trapz(group['mean_SWE'], days_since_start)
       return pd.Series({'total_SWE_integration': total_swe_integration})

2. Probability Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gringorten plotting position formula is used to transform the integrated SWE values to probabilities:

.. code-block:: python

   def gringorten_probabilities(values):
       """Compute Gringorten plotting position probabilities."""
       sorted_values = np.sort(values)
       ranks = np.argsort(np.argsort(values)) + 1  # Rank from smallest to largest
       n = len(values)
       probabilities = (ranks - 0.44) / (n + 0.12)
       return probabilities

3. Standardization
~~~~~~~~~~~~~~~~~

The probabilities are converted to standardized values using the inverse normal distribution:

.. code-block:: python

   def compute_swei(probabilities):
       """Transform probabilities to SWEI using the inverse normal distribution."""
       return norm.ppf(probabilities)

4. Classification
~~~~~~~~~~~~~~~~

The standardized values are classified into drought categories based on the following thresholds:

.. code-block:: python

   def classify_drought(swei):
       """Classify drought conditions based on SWEI values."""
       if swei <= -2.0:
           return "Exceptional Drought"
       elif -2.0 < swei <= -1.5:
           return "Extreme Drought"
       elif -1.5 < swei <= -1.0:
           return "Severe Drought"
       elif -1.0 < swei <= -0.5:
           return "Moderate Drought"
       elif -0.5 < swei <= 0.5:
           return "Near Normal"
       elif 0.5 < swei <= 1.0:
           return "Abnormally Wet"
       elif 1.0 < swei <= 1.5:
           return "Moderately Wet"
       elif 1.5 < swei <= 2.0:
           return "Very Wet"
       else:
           return "Extremely Wet"

Drought Classification Thresholds
---------------------------------

The SSWEI values are classified into the following drought categories:

.. list-table:: SSWEI Drought Classification
   :header-rows: 1
   :widths: 30 20 50

   * - SSWEI Range
     - Classification
     - Description
   * - ≤ -2.0
     - Exceptional Drought
     - Extremely dry conditions, occurring ~2% of the time
   * - -2.0 to -1.5
     - Extreme Drought
     - Very dry conditions, occurring ~7% of the time
   * - -1.5 to -1.0
     - Severe Drought
     - Severely dry conditions, occurring ~16% of the time
   * - -1.0 to -0.5
     - Moderate Drought
     - Moderately dry conditions, occurring ~34% of the time
   * - -0.5 to 0.5
     - Near Normal
     - Normal conditions, occurring ~68% of the time
   * - 0.5 to 1.0
     - Abnormally Wet
     - Moderately wet conditions, occurring ~34% of the time
   * - 1.0 to 1.5
     - Moderately Wet
     - Very wet conditions, occurring ~16% of the time
   * - 1.5 to 2.0
     - Very Wet
     - Extremely wet conditions, occurring ~7% of the time
   * - > 2.0
     - Extremely Wet
     - Exceptionally wet conditions, occurring ~2% of the time

Handling Zero Values
-------------------

Zero SWE values can cause issues in the calculation of the SSWEI. To address this, the package implements a perturbation approach to replace zero values with small positive values:

.. code-block:: python

   def perturb_zeros(swe_column):
       """Perturbs zero values with small positive values."""
       swe_array = swe_column.to_numpy()
       nonzero_min = swe_array[swe_array > 0].min()  # Find the smallest nonzero value
       
       # Generate perturbations for zero values
       perturbation = np.random.uniform(0, nonzero_min / 2, size=swe_column[swe_column == 0].shape)
       
       # Replace zeros with perturbation
       swe_column[swe_column == 0] = perturbation
       
       return swe_column

Elevation-Based Analysis
-----------------------

The package also supports elevation-based SSWEI analysis, which allows for the comparison of snow drought conditions at different elevation bands. This is particularly useful for understanding the spatial variability of snow drought conditions in mountainous regions.

The elevation-based analysis involves:

1. **Elevation Binning**: Grouping stations or grid points into elevation bands
2. **Band-wise SSWEI Calculation**: Computing SSWEI for each elevation band separately
3. **Comparative Analysis**: Comparing drought conditions across different elevations

Usage Example
------------

Here's a complete example of calculating SSWEI:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from scipy.stats import norm
   from scipy.integrate import trapz
   
   # Load your SWE data
   swe_data = pd.read_csv('swe_data.csv')
   swe_data['date'] = pd.to_datetime(swe_data['date'])
   
   # Group by season and integrate SWE
   season_data = swe_data.groupby('season_year').apply(integrate_season).reset_index()
   
   # Calculate Gringorten probabilities
   season_data['Gringorten_probabilities'] = gringorten_probabilities(
       season_data['total_SWE_integration']
   )
   
   # Compute SSWEI
   season_data['SSWEI'] = compute_swei(season_data['Gringorten_probabilities'])
   
   # Classify drought conditions
   season_data['Drought_Classification'] = season_data['SSWEI'].apply(classify_drought)

References
----------

- Huning, L. S., & AghaKouchak, A. (2020). Global snow drought hot spots and characteristics. *Proceedings of the National Academy of Sciences*, 117(33), 19753-19759.
- Heldmyer, A., Livneh, B., Molotch, N. P., & Harpold, A. A. (2022). Sensitivity of snowpack storage efficiency to precipitation and temperature using a regional snow drought index. *Hydrology and Earth System Sciences*, 26(22), 5721-5735.

.. seealso::
   
   - :doc:`gap_filling` for gap filling methodology
   - :doc:`heldmyer_classification` for alternative classification approaches
   - :doc:`../user_guide/workflows/sswei_calculation` for practical implementation workflows
