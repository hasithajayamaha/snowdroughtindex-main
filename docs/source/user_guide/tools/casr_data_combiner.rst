CaSR Data Combiner Tool
======================

The CaSR Data Combiner is a utility for combining NetCDF files from the CaSR (Canadian Arctic Snow Reanalysis) SWE dataset. The tool can handle both temporal and spatial combinations of the data files.

Dataset Structure
----------------

The CaSR SWE dataset contains files organized by:

**Variable types:**
  - ``A_PR24_SFC`` (precipitation data)
  - ``P_SWE_LAND`` (snow water equivalent data)

**Spatial regions:** 4 regions in a 2×2 grid
  - ``rlon211-245_rlat386-420``
  - ``rlon211-245_rlat421-455``
  - ``rlon246-280_rlat386-420``
  - ``rlon246-280_rlat421-455``

**Time periods:** 4-year chunks from 1980-2023 (11 time periods total)

Usage
-----

Basic Commands
~~~~~~~~~~~~~~

.. code-block:: bash

   # Get information about the dataset
   python combine_casr_swe_files.py --info-only

   # Combine files across time only (keep spatial regions separate)
   python combine_casr_swe_files.py --temporal-only

   # Combine files across space only (keep time periods separate)
   python combine_casr_swe_files.py --spatial-only

   # Combine files across both time and space (creates 2 large files)
   python combine_casr_swe_files.py --combine-all

   # Show help and all options
   python combine_casr_swe_files.py --help

Advanced Options
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Specify custom input directory
   python combine_casr_swe_files.py --input /path/to/casr/files --temporal-only

   # Specify custom output directory
   python combine_casr_swe_files.py --output /path/to/output --temporal-only

   # Enable verbose logging
   python combine_casr_swe_files.py --temporal-only --verbose

Output Files
-----------

Temporal-only combination (``--temporal-only``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates 8 files (4 spatial regions × 2 variables):

- ``CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat386-420_temporal_combined.nc``
- ``CaSR_v3.1_A_PR24_SFC_rlon211-245_rlat421-455_temporal_combined.nc``
- ``CaSR_v3.1_A_PR24_SFC_rlon246-280_rlat386-420_temporal_combined.nc``
- ``CaSR_v3.1_A_PR24_SFC_rlon246-280_rlat421-455_temporal_combined.nc``
- ``CaSR_v3.1_P_SWE_LAND_rlon211-245_rlat386-420_temporal_combined.nc``
- ``CaSR_v3.1_P_SWE_LAND_rlon211-245_rlat421-455_temporal_combined.nc``
- ``CaSR_v3.1_P_SWE_LAND_rlon246-280_rlat386-420_temporal_combined.nc``
- ``CaSR_v3.1_P_SWE_LAND_rlon246-280_rlat421-455_temporal_combined.nc``

Spatial-only combination (``--spatial-only``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates 22 files (11 time periods × 2 variables):

- Files named like: ``CaSR_v3.1_A_PR24_SFC_1980-1983_spatial_combined.nc``

Full combination (``--combine-all``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates 2 files (1 per variable):

- ``CaSR_v3.1_A_PR24_SFC_combined_full.nc``
- ``CaSR_v3.1_P_SWE_LAND_combined_full.nc``

Requirements
-----------

The script requires the following Python packages:

- ``xarray`` (for NetCDF handling)
- ``numpy``
- ``pandas``
- ``netCDF4``
- ``pathlib`` (built-in)
- ``argparse`` (built-in)
- ``logging`` (built-in)

Features
--------

- **Automatic file parsing**: Intelligently parses CaSR filename conventions
- **Memory efficient**: Processes files in chunks and closes datasets after use
- **Metadata preservation**: Adds combination metadata to output files
- **Error handling**: Robust error handling with informative logging
- **Flexible output**: Multiple combination strategies available
- **Progress tracking**: Detailed logging of processing steps

File Size Considerations
-----------------------

- Original files: ~88 files, each ~100-200 MB
- Temporal combinations: 8 files, each ~1-2 GB
- Full combinations: 2 files, each ~8-16 GB

.. warning::
   Ensure you have sufficient disk space before running full combinations.

Example Workflow
---------------

1. **Explore the dataset**:

   .. code-block:: bash

      python combine_casr_swe_files.py --info-only

2. **Create temporal combinations** (recommended for most use cases):

   .. code-block:: bash

      python combine_casr_swe_files.py --temporal-only

3. **Verify output**:

   .. code-block:: bash

      ls -la data/output_data/combined_casr/

Troubleshooting
--------------

- **Memory issues**: Use ``--temporal-only`` or ``--spatial-only`` instead of ``--combine-all``
- **Disk space**: Check available space before running combinations
- **File permissions**: Ensure write permissions to output directory
- **Missing files**: Use ``--info-only`` to verify all expected files are present

.. note::
   The script should be run from the project root directory where the ``data/input_data/CaSR_SWE/`` folder is located.
