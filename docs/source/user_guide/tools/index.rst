Data Processing Tools
====================

The Snow Drought Index package includes several command-line tools for data processing and preparation. These tools are designed to work with CaSR (Canadian Arctic Snow Reanalysis) data and elevation datasets.

Available Tools
--------------

.. toctree::
   :maxdepth: 2

   casr_data_combiner
   elevation_data_extraction

Tool Overview
------------

CaSR Data Combiner
~~~~~~~~~~~~~~~~~

The :doc:`casr_data_combiner` tool combines NetCDF files from the CaSR dataset across temporal and/or spatial dimensions. This is essential for creating manageable datasets from the original fragmented files.

**Key Features:**
- Temporal and spatial combination strategies
- Memory-efficient processing
- Comprehensive logging and error handling
- Multiple output formats

Elevation Data Extraction
~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`elevation_data_extraction` tool extracts meteorological data from combined CaSR files at specific elevation points defined in shapefiles.

**Key Features:**
- Shapefile-based point extraction
- Multiple coordinate system support
- Time series data handling
- CSV and Parquet output formats

Installation Requirements
------------------------

Each tool has specific requirements. Install the extraction tools dependencies:

.. code-block:: bash

   pip install -r requirements_extraction.txt

General Workflow
---------------

1. **Data Combination**: Use the CaSR Data Combiner to create manageable combined files from the original CaSR dataset
2. **Point Extraction**: Use the Elevation Data Extraction tool to extract data at specific locations
3. **Analysis**: Use the extracted data with the main Snow Drought Index analysis workflows

.. seealso::
   
   - :doc:`../workflows/data_preparation` for complete data preparation workflows
   - :doc:`../installation` for general package installation instructions
