Elevation Data Extraction Tool
=============================

The Elevation Data Extraction tool extracts data from combined CaSR NetCDF files at elevation point locations defined in shapefiles. It can handle both temporal combined and full combined data types.

Overview
--------

The ``extract_elevation_data.py`` script provides a comprehensive solution for extracting CaSR (Canadian Surface Reanalysis) data at specific elevation points. It reads elevation data from shapefiles and extracts corresponding meteorological data from combined NetCDF files.

Features
--------

- **Flexible Input**: Works with elevation shapefiles and combined CaSR NetCDF files
- **Multiple Data Types**: Handles both temporal combined and full combined data
- **Coordinate Handling**: Automatically detects and handles different coordinate systems (lat/lon, rlon/rlat)
- **Time Series Support**: Extracts time series data when available
- **Multiple Output Formats**: Saves results in CSV and/or Parquet formats
- **Comprehensive Logging**: Detailed logging for monitoring progress and debugging
- **Summary Reports**: Generates JSON summary reports with extraction statistics

Requirements
-----------

Install the required dependencies:

.. code-block:: bash

   pip install -r requirements_extraction.txt

Required packages:

- ``numpy>=1.21.0``
- ``pandas>=1.3.0``
- ``xarray>=0.19.0``
- ``geopandas>=0.10.0``
- ``shapely>=1.8.0``
- ``netcdf4>=1.5.0``
- ``pyproj>=3.2.0``

Directory Structure
------------------

The script expects the following directory structure:

.. code-block:: text

   data/
   ├── input_data/
   │   └── Elevation/
   │       ├── Bow_elevation_combined.shp
   │       ├── Bow_elevation_combined.dbf
   │       ├── Bow_elevation_combined.shx
   │       └── Bow_elevation_combined.cpg
   └── output_data/
       ├── combined_casr/
       │   ├── CaSR_v3.1_A_PR24_SFC_combined_full.nc
       │   ├── CaSR_v3.1_A_PR24_SFC_*_temporal_combined.nc
       │   ├── CaSR_v3.1_P_SWE_LAND_combined_full.nc
       │   └── CaSR_v3.1_P_SWE_LAND_*_temporal_combined.nc
       └── elevation/  # Output directory (created automatically)

Usage
-----

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~

Basic usage:

.. code-block:: bash

   python extract_elevation_data.py

With custom parameters:

.. code-block:: bash

   python extract_elevation_data.py \
       --elevation-dir data/input_data/Elevation \
       --casr-dir data/output_data/combined_casr \
       --output-dir data/output_data/elevation \
       --file-types temporal full \
       --format csv \
       --verbose

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~

- ``--elevation-dir``: Directory containing elevation shapefiles (default: ``data/input_data/Elevation``)
- ``--casr-dir``: Directory containing combined CaSR NetCDF files (default: ``data/output_data/combined_casr``)
- ``--output-dir``: Output directory for extracted data (default: ``data/output_data/elevation``)
- ``--file-types``: Types of files to process (``temporal``, ``full``, or both; default: both)
- ``--format``: Output format (``csv``, ``parquet``, or ``both``; default: ``csv``)
- ``--verbose``: Enable verbose logging

Python API
~~~~~~~~~~

.. code-block:: python

   from extract_elevation_data import ElevationDataExtractor

   # Initialize extractor
   extractor = ElevationDataExtractor(
       elevation_dir="data/input_data/Elevation",
       combined_casr_dir="data/output_data/combined_casr",
       output_dir="data/output_data/elevation"
   )

   # Process all files
   results = extractor.process_all_files(file_types=['temporal', 'full'])

   # Save results
   extractor.save_results(results, format='csv')

   # Generate summary report
   extractor.generate_summary_report(results)

Examples
--------

Example 1: Process Only Temporal Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python extract_elevation_data.py --file-types temporal --format csv

Example 2: Process Only Full Combined Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python extract_elevation_data.py --file-types full --format parquet

Example 3: Process All Files with Multiple Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python extract_elevation_data.py --file-types temporal full --format both --verbose

Example 4: Using the Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the example script:

.. code-block:: bash

   python example_extraction.py

Output Files
-----------

The script generates the following output files:

Data Files
~~~~~~~~~

- ``elevation_extracted_temporal_[filename].csv/parquet``: Extracted data from temporal files
- ``elevation_extracted_full_[filename].csv/parquet``: Extracted data from full combined files

Summary Report
~~~~~~~~~~~~~

- ``extraction_summary.json``: JSON file containing extraction statistics and metadata

Output Data Structure
~~~~~~~~~~~~~~~~~~~~

The extracted data includes the following columns:

- ``point_id``: Unique identifier for each elevation point
- ``original_lon``, ``original_lat``: Original coordinates from the shapefile
- ``grid_lon``, ``grid_lat``: Nearest grid coordinates from the NetCDF file
- ``elevation``: Elevation value (if available in shapefile)
- ``time``: Time stamp (for time series data)
- ``[variable_names]``: All variables from the NetCDF file (e.g., temperature, precipitation, SWE)

Data Types Handled
-----------------

Temporal Combined Files
~~~~~~~~~~~~~~~~~~~~~~

Files with ``temporal_combined`` in the name contain time series data for specific spatial regions:

- ``CaSR_v3.1_A_PR24_SFC_*_temporal_combined.nc``
- ``CaSR_v3.1_P_SWE_LAND_*_temporal_combined.nc``

Full Combined Files
~~~~~~~~~~~~~~~~~~

Files with ``combined_full`` in the name contain spatially and temporally combined data:

- ``CaSR_v3.1_A_PR24_SFC_combined_full.nc``
- ``CaSR_v3.1_P_SWE_LAND_combined_full.nc``

Coordinate Systems
-----------------

The script automatically handles different coordinate systems:

- **Geographic coordinates**: ``lon``/``lat`` or ``longitude``/``latitude``
- **Rotated coordinates**: ``rlon``/``rlat`` (common in regional climate models)

Error Handling
-------------

The script includes comprehensive error handling:

- Missing files or directories
- Coordinate system mismatches
- NetCDF file reading errors
- Point extraction failures

All errors are logged with detailed information for debugging.

Logging
-------

The script provides detailed logging at different levels:

- **INFO**: General progress information
- **WARNING**: Non-critical issues (e.g., points that couldn't be extracted)
- **ERROR**: Critical errors that prevent processing

Use ``--verbose`` flag for more detailed logging.

Performance Considerations
-------------------------

- **Memory Usage**: Large NetCDF files may require significant memory
- **Processing Time**: Time depends on the number of elevation points and size of NetCDF files
- **Output Size**: Time series data can generate large output files

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Missing Dependencies**: Install all required packages from ``requirements_extraction.txt``
2. **File Not Found**: Check that input directories contain the expected files
3. **Coordinate Mismatch**: The script attempts to handle different coordinate systems automatically
4. **Memory Issues**: For large datasets, consider processing files individually

Debug Mode
~~~~~~~~~

Run with verbose logging to see detailed processing information:

.. code-block:: bash

   python extract_elevation_data.py --verbose

.. seealso::
   
   - :doc:`casr_data_combiner` for information on combining CaSR data files
   - :doc:`../workflows/data_preparation` for data preparation workflows
