Installation
===========

This guide will help you install the Snow Drought Index package and its dependencies.

Requirements
-----------

The Snow Drought Index package requires the following dependencies:

* Python 3.8 or higher
* numpy
* pandas
* xarray
* scipy
* matplotlib
* seaborn
* scikit-learn
* netCDF4
* geopandas
* shapely

Installation from Source
-----------------------

To install the Snow Drought Index package from source, clone the repository and install using pip:

.. code-block:: bash

   git clone https://github.com/username/snowdroughtindex.git
   cd snowdroughtindex
   pip install -e .

This will install the package in development mode, allowing you to make changes to the code and have them immediately reflected in your environment.

Using the Package
----------------

After installation, you can import the package in your Python code:

.. code-block:: python

   import snowdroughtindex as sdi

   # Example usage
   data = sdi.core.data_preparation.load_swe_data('path/to/data.nc')
