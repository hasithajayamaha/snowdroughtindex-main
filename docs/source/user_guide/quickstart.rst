Quickstart
==========

This guide provides a quick introduction to using the Snow Drought Index package for analyzing snow drought conditions.

Basic Usage
----------

The Snow Drought Index package provides several modules for analyzing snow drought conditions. Here's a basic workflow:

1. Load and prepare SWE data
2. Perform gap filling if needed
3. Calculate SSWEI (Standardized Snow Water Equivalent Index)
4. Classify drought conditions
5. Visualize results

Loading Data
-----------

.. code-block:: python

   from snowdroughtindex.core.data_preparation import load_swe_data
   
   # Load SWE data from a NetCDF file
   swe_data = load_swe_data('path/to/swe_data.nc')
   
   # Filter data for a specific basin or region
   filtered_data = filter_by_basin(swe_data, 'path/to/basin_shapefile.shp', basin_name='Basin Name')

Gap Filling
----------

.. code-block:: python

   from snowdroughtindex.core.gap_filling import qm_gap_filling
   
   # Fill gaps in SWE data using quantile mapping
   filled_data = qm_gap_filling(swe_data, reference_data, method='linear')

Calculating SSWEI
----------------

.. code-block:: python

   from snowdroughtindex.core.sswei import integrate_season, gringorten_probabilities, compute_swei
   
   # Integrate SWE over the season
   integrated_swe = integrate_season(swe_data)
   
   # Calculate Gringorten probabilities
   probabilities = gringorten_probabilities(integrated_swe)
   
   # Compute SSWEI
   sswei = compute_swei(probabilities)

Classifying Drought
------------------

.. code-block:: python

   from snowdroughtindex.core.drought_classification import classify_drought
   
   # Classify drought conditions based on SSWEI
   drought_classes = classify_drought(sswei)

Visualization
------------

.. code-block:: python

   from snowdroughtindex.utils.visualization import plot_sswei_timeseries, plot_drought_classification
   
   # Plot SSWEI time series
   plot_sswei_timeseries(sswei, years)
   
   # Plot drought classification
   plot_drought_classification(drought_classes, years)

Using Classes
------------

The package also provides class-based implementations for more complex analyses:

.. code-block:: python

   from snowdroughtindex.core.dataset import SWEDataset
   from snowdroughtindex.core.sswei_class import SSWEI
   from snowdroughtindex.core.drought_analysis import DroughtAnalysis
   
   # Create a SWEDataset object
   dataset = SWEDataset('path/to/swe_data.nc')
   
   # Fill gaps
   dataset.fill_gaps(method='linear')
   
   # Create an SSWEI object
   sswei = SSWEI(dataset)
   
   # Calculate SSWEI
   sswei.calculate()
   
   # Classify drought
   sswei.classify_drought()
   
   # Create a DroughtAnalysis object
   analysis = DroughtAnalysis(sswei)
   
   # Analyze drought conditions
   analysis.analyze_elevation_bands()
   
   # Visualize results
   analysis.plot_drought_trends()
