Migration Overview
================

This guide provides an overview of migrating from the original Snow Drought Index notebooks to the refactored package structure.

Why Migrate?
-----------

The refactored Snow Drought Index package offers several advantages over the original notebooks:

1. **Improved Code Organization**: Functions are logically grouped into modules, making the code easier to navigate and understand.
2. **Enhanced Reusability**: Functions have standardized signatures, comprehensive docstrings, and type hints.
3. **Better Error Handling**: Input validation and meaningful error messages help identify and fix issues.
4. **Class-Based Implementation**: Classes like ``SWEDataset``, ``SSWEI``, and ``DroughtAnalysis`` provide better state management and encapsulation.
5. **Comprehensive Testing**: Unit tests, integration tests, and performance tests ensure code reliability.
6. **Detailed Documentation**: API references, methodology guides, and usage examples make the package easier to use.

Migration Strategy
----------------

We recommend the following strategy for migrating to the new package:

1. **Understand the New Structure**: Familiarize yourself with the new package structure and API.
2. **Identify Equivalent Functionality**: Map your existing code to the new API.
3. **Incremental Migration**: Migrate one component at a time, testing as you go.
4. **Use the Migration Guides**: Follow the specific migration guides for each notebook.

Package Structure
---------------

The refactored package has the following structure:

.. code-block:: text

    snowdroughtindex/
    ├── snowdroughtindex/
    │   ├── __init__.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── data_preparation.py
    │   │   ├── gap_filling.py
    │   │   ├── sswei.py
    │   │   └── drought_classification.py
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── visualization.py
    │   │   ├── statistics.py
    │   │   └── io.py
    │   └── analysis/
    │       ├── __init__.py
    │       ├── scs_analysis.py
    │       └── case_studies.py
    ├── notebooks/
    │   ├── workflows/
    │   │   ├── data_preparation_workflow.ipynb
    │   │   ├── gap_filling_workflow.ipynb
    │   │   ├── sswei_calculation_workflow.ipynb
    │   │   ├── drought_classification_workflow.ipynb
    │   │   ├── scs_analysis_workflow.ipynb
    │   │   └── case_study_workflow.ipynb

Key Module Mapping
----------------

Here's how the original notebooks map to the new modules:

.. list-table::
   :header-rows: 1

   * - Original Notebook
     - New Modules
   * - ``SSWEI.ipynb``
     - | ``snowdroughtindex.core.data_preparation``
       | ``snowdroughtindex.core.sswei``
       | ``snowdroughtindex.core.drought_classification``
       | ``snowdroughtindex.utils.visualization``
   * - ``SCS_analysis.ipynb``
     - | ``snowdroughtindex.core.data_preparation``
       | ``snowdroughtindex.analysis.scs_analysis``
       | ``snowdroughtindex.utils.statistics``
       | ``snowdroughtindex.utils.visualization``
   * - ``Data_preparation.ipynb``
     - | ``snowdroughtindex.core.data_preparation``
       | ``snowdroughtindex.core.gap_filling``
       | ``snowdroughtindex.utils.visualization``
   * - Case study notebooks
     - | ``snowdroughtindex.analysis.case_studies``
       | ``snowdroughtindex.core.sswei``
       | ``snowdroughtindex.core.drought_classification``
       | ``snowdroughtindex.utils.visualization``

Class-Based Implementation
------------------------

The refactored package includes several classes that encapsulate related functionality:

1. **SWEDataset Class**: Handles loading, preprocessing, and gap filling of SWE data.
2. **SSWEI Class**: Calculates SSWEI and classifies drought conditions.
3. **DroughtAnalysis Class**: Analyzes drought conditions across elevation bands and time periods.
4. **Configuration Class**: Manages parameters for gap filling, SSWEI calculation, and visualization.

Example Migration
--------------

Here's a simple example of migrating from the original notebooks to the new package:

Original code:

.. code-block:: python

    # From SSWEI.ipynb
    import xarray as xr
    import pandas as pd
    import numpy as np
    
    # Load data
    ds = xr.open_dataset('data/swe_data.nc')
    
    # Calculate seasonal mean
    seasonal_mean = ds.groupby('time.dayofyear').mean()
    
    # Integrate SWE over the season
    integrated_swe = integrate_season(ds, start_date='10-01', end_date='06-30')
    
    # Calculate SSWEI
    sswei = calculate_sswei(integrated_swe)
    
    # Classify drought
    drought_class = classify_drought(sswei)

Migrated code:

.. code-block:: python

    from snowdroughtindex.core.dataset import SWEDataset
    from snowdroughtindex.core.sswei_class import SSWEI
    
    # Load data using SWEDataset class
    swe_dataset = SWEDataset('data/swe_data.nc')
    
    # Calculate SSWEI using SSWEI class
    sswei_obj = SSWEI(swe_dataset)
    sswei_obj.calculate(start_date='10-01', end_date='06-30')
    
    # Classify drought
    drought_class = sswei_obj.classify_drought()

Next Steps
---------

For detailed migration instructions for specific notebooks, see the following guides:

- :doc:`SSWEI Notebook Migration <sswei_notebook>`
- :doc:`SCS Analysis Notebook Migration <scs_analysis_notebook>`
- :doc:`Case Study Notebooks Migration <case_study_notebooks>`
