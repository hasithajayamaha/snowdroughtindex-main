Data Preparation Workflow
========================

This guide explains the data preparation workflow for the Snow Drought Index package.

Purpose
-------

The data preparation workflow is the first step in the Snow Drought Index analysis process. It involves:

1. Loading SWE (Snow Water Equivalent) data and other required datasets
2. Preprocessing the data to prepare it for analysis
3. Extracting stations within a basin of interest
4. Assessing data availability
5. Saving processed data for use in subsequent analyses

Prerequisites
------------

Before starting this workflow, ensure you have:

- Installed the Snow Drought Index package
- Prepared input data files:
  - SWE data in NetCDF format
  - Precipitation data in NetCDF format (if needed)
  - Basin shapefile for spatial filtering

Workflow Steps
-------------

Step 1: Data Loading
^^^^^^^^^^^^^^^^^^^

First, load the SWE data and other required datasets using the functions from the ``data_preparation`` module:

.. code-block:: python

    from snowdroughtindex.core import data_preparation
    
    # Define data paths
    swe_path = 'path/to/SWE_data.nc'
    precip_path = 'path/to/precip_data.nc'
    basin_path = 'path/to/basin_shapefile.shp'
    
    # Load data
    swe_data = data_preparation.load_swe_data(swe_path)
    precip_data = data_preparation.load_precip_data(precip_path)
    basin_data = data_preparation.load_basin_data(basin_path)

Step 2: Data Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^

Next, preprocess the data to prepare it for analysis:

.. code-block:: python

    # Preprocess SWE data
    swe_processed = data_preparation.preprocess_swe(swe_data)
    
    # Preprocess precipitation data
    precip_processed = data_preparation.preprocess_precip(precip_data)
    
    # Convert to GeoDataFrame for spatial operations
    swe_gdf = data_preparation.convert_to_geodataframe(swe_processed)

Step 3: Station Extraction and Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract stations within the basin of interest:

.. code-block:: python

    # Define basin ID
    basin_id = 'example_basin'  # Replace with actual basin ID
    
    # Extract stations within the basin
    stations_in_basin, basin_buffer = data_preparation.extract_stations_in_basin(
        swe_gdf, basin_data, basin_id
    )
    
    # Filter data for stations in the basin
    station_ids = stations_in_basin['station_id'].tolist()
    swe_basin = data_preparation.filter_stations(swe_data, station_ids)

Step 4: Data Availability Assessment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assess the availability of data for the stations in the basin:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Assess data availability
    availability = data_preparation.assess_data_availability(swe_basin)
    
    # Visualize data availability
    plt.figure(figsize=(10, 6))
    availability.plot(cmap='viridis')
    plt.colorbar(label='Data Availability (%)')
    plt.title('SWE Data Availability by Station')
    plt.xlabel('Station ID')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.show()

Step 5: Save Processed Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the processed data for use in subsequent analyses:

.. code-block:: python

    # Save processed data using xarray's built-in methods
    swe_basin.to_netcdf('path/to/swe_basin_processed.nc')

Key Functions
------------

The data preparation workflow uses the following key functions from the ``data_preparation`` module:

- ``load_swe_data()``, ``load_precip_data()``, ``load_basin_data()`` for data loading
- ``preprocess_swe()``, ``preprocess_precip()`` for data preprocessing
- ``convert_to_geodataframe()`` for converting data to GeoDataFrame
- ``extract_stations_in_basin()`` for extracting stations within a basin
- ``filter_stations()`` for filtering data by station
- ``assess_data_availability()`` for assessing data availability

Example Notebook
---------------

For a complete example of the data preparation workflow, refer to the 
`data_preparation_workflow.ipynb <https://github.com/yourusername/snowdroughtindex/blob/main/notebooks/workflows/data_preparation_workflow.ipynb>`_ 
notebook in the package repository.

Next Steps
---------

After completing the data preparation workflow, you can proceed to:

- :doc:`Gap filling workflow <gap_filling>` to fill gaps in the SWE data
- :doc:`SSWEI calculation workflow <sswei_calculation>` to calculate the Standardized Snow Water Equivalent Index
