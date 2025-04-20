Command-Line Interface
====================

The Snow Drought Index package provides a command-line interface (CLI) for common operations, allowing you to perform analyses without writing Python code.

Installation
-----------

The CLI is automatically installed with the package. You can access it using:

.. code-block:: bash

    python -m snowdroughtindex.cli [command] [options]

Available Commands
----------------

The CLI provides the following commands:

- ``fill-gaps``: Fill gaps in SWE data
- ``calculate-sswei``: Calculate SSWEI from SWE data
- ``classify-drought``: Classify drought conditions based on SSWEI values
- ``plot-sswei``: Plot SSWEI time series
- ``run-workflow``: Run a complete workflow

To get help for a specific command, use:

.. code-block:: bash

    python -m snowdroughtindex.cli [command] --help

Fill Gaps Command
---------------

The ``fill-gaps`` command fills gaps in SWE data using the quantile mapping method.

.. code-block:: bash

    python -m snowdroughtindex.cli fill-gaps \
        --input-file path/to/swe_data.nc \
        --output-file path/to/gap_filled_data.nc \
        --window-days 15 \
        --min-obs-corr 10 \
        --min-obs-cdf 5 \
        --min-corr 0.7 \
        --parallel \
        --n-jobs 4 \
        --memory-efficient

Options:

- ``--input-file``: Path to input SWE data file (required)
- ``--output-file``: Path to output file for gap-filled data (required)
- ``--window-days``: Number of days to select data for around a certain day of year (default: 15)
- ``--min-obs-corr``: Minimum number of overlapping observations required to calculate correlation (default: 10)
- ``--min-obs-cdf``: Minimum number of stations required to calculate a station's CDF (default: 5)
- ``--min-corr``: Minimum correlation value required to keep a donor station (default: 0.7)
- ``--parallel``: Enable parallel processing
- ``--n-jobs``: Number of jobs for parallel processing (default: -1, all available cores)
- ``--memory-efficient``: Enable memory-efficient algorithms

Calculate SSWEI Command
---------------------

The ``calculate-sswei`` command calculates the Standardized Snow Water Equivalent Index (SSWEI) from SWE data.

.. code-block:: bash

    python -m snowdroughtindex.cli calculate-sswei \
        --input-file path/to/swe_data.nc \
        --output-file path/to/sswei_results.csv \
        --start-month 12 \
        --end-month 3 \
        --min-years 10 \
        --distribution gamma \
        --reference-period 1980 2010 \
        --parallel \
        --n-jobs 4 \
        --memory-efficient

Options:

- ``--input-file``: Path to input SWE data file (required)
- ``--output-file``: Path to output file for SSWEI results (required)
- ``--start-month``: Starting month of the season (1-12) (default: 12)
- ``--end-month``: Ending month of the season (1-12) (default: 3)
- ``--min-years``: Minimum number of years required for calculation (default: 10)
- ``--distribution``: Probability distribution to use (gamma or normal) (default: gamma)
- ``--reference-period``: Reference period (start_year end_year) for standardization
- ``--parallel``: Enable parallel processing
- ``--n-jobs``: Number of jobs for parallel processing (default: -1, all available cores)
- ``--memory-efficient``: Enable memory-efficient algorithms

Classify Drought Command
----------------------

The ``classify-drought`` command classifies drought conditions based on SSWEI values.

.. code-block:: bash

    python -m snowdroughtindex.cli classify-drought \
        --input-file path/to/sswei_results.csv \
        --output-file path/to/drought_classes.csv \
        --thresholds exceptional=-2.0 extreme=-1.5 severe=-1.0 moderate=-0.5

Options:

- ``--input-file``: Path to input SSWEI data file (required)
- ``--output-file``: Path to output file for drought classification results (required)
- ``--thresholds``: Custom thresholds for drought classification (e.g., exceptional=-2.0 extreme=-1.5)

Plot SSWEI Command
----------------

The ``plot-sswei`` command creates a plot of SSWEI time series.

.. code-block:: bash

    python -m snowdroughtindex.cli plot-sswei \
        --input-file path/to/sswei_results.csv \
        --output-file path/to/sswei_plot.png \
        --title "SSWEI Time Series" \
        --figsize 10 6 \
        --dpi 100

Options:

- ``--input-file``: Path to input SSWEI data file (required)
- ``--output-file``: Path to output image file (required)
- ``--title``: Plot title (default: "SSWEI Time Series")
- ``--figsize``: Figure size (width height) (default: 10 6)
- ``--dpi``: Figure DPI (default: 100)

Run Workflow Command
------------------

The ``run-workflow`` command runs a complete workflow, from data loading to visualization.

.. code-block:: bash

    python -m snowdroughtindex.cli run-workflow \
        --input-file path/to/swe_data.nc \
        --output-dir path/to/output_directory \
        --config-file path/to/config.yaml \
        --workflow sswei

Options:

- ``--input-file``: Path to input SWE data file (required)
- ``--output-dir``: Directory to save results (required)
- ``--config-file``: Path to configuration file (YAML or JSON)
- ``--workflow``: Workflow to run (sswei, drought-analysis, or elevation-analysis) (default: sswei)

Available Workflows:

- ``sswei``: Basic SSWEI calculation and drought classification
- ``drought-analysis``: SSWEI calculation, drought classification, and drought characteristics analysis
- ``elevation-analysis``: SSWEI calculation, drought classification, and elevation band analysis

Example Usage
-----------

Here are some example workflows:

Basic SSWEI Calculation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Fill gaps in SWE data
    python -m snowdroughtindex.cli fill-gaps \
        --input-file raw_swe_data.nc \
        --output-file gap_filled_data.nc
    
    # Calculate SSWEI
    python -m snowdroughtindex.cli calculate-sswei \
        --input-file gap_filled_data.nc \
        --output-file sswei_results.csv
    
    # Classify drought conditions
    python -m snowdroughtindex.cli classify-drought \
        --input-file sswei_results.csv \
        --output-file drought_classes.csv
    
    # Plot SSWEI time series
    python -m snowdroughtindex.cli plot-sswei \
        --input-file drought_classes.csv \
        --output-file sswei_plot.png

Complete Workflow
^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run a complete workflow
    python -m snowdroughtindex.cli run-workflow \
        --input-file raw_swe_data.nc \
        --output-dir results \
        --workflow drought-analysis

Using a Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^

You can create a configuration file (YAML or JSON) to specify parameters for the workflow:

.. code-block:: yaml

    # config.yaml
    gap_filling:
      window_days: 15
      min_obs_corr: 10
      min_obs_cdf: 5
      min_corr: 0.7
    
    sswei:
      start_month: 12
      end_month: 3
      min_years: 10
      distribution: gamma
    
    drought_classification:
      exceptional: -2.0
      extreme: -1.5
      severe: -1.0
      moderate: -0.5
    
    performance:
      parallel: true
      n_jobs: 4
      memory_efficient: true

Then use it with the ``run-workflow`` command:

.. code-block:: bash

    python -m snowdroughtindex.cli run-workflow \
        --input-file raw_swe_data.nc \
        --output-dir results \
        --config-file config.yaml \
        --workflow elevation-analysis
