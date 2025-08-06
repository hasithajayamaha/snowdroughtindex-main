Installation
===========

This guide will help you install the Snow Drought Index package and its dependencies.

Requirements
-----------

The Snow Drought Index package requires Python 3.8 or higher and uses three separate requirements files to manage dependencies for different use cases.

System Requirements
~~~~~~~~~~~~~~~~~~

* **Python**: 3.8 or higher
* **Operating System**: Windows, macOS, or Linux
* **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
* **Storage**: At least 1GB free space for installation and data

Quick Setup (Recommended)
-------------------------

Clone the repository and run the automated setup:

.. code-block:: bash

   git clone https://github.com/username/snowdroughtindex.git
   cd snowdroughtindex
   python setup.py

This will automatically:

1. Create a virtual environment in ``./venv/``
2. Activate the virtual environment
3. Upgrade pip to the latest version
4. Install all required dependencies from multiple requirements files
5. Install the package in development mode

Setup Options
~~~~~~~~~~~~

The enhanced ``setup.py`` provides several installation modes:

.. code-block:: bash

   # Full automated setup with virtual environment (default)
   python setup.py

   # Full automated setup with virtual environment
   python setup.py install

   # Standard setuptools development mode
   python setup.py develop

   # Show detailed usage instructions
   python setup.py --help

Manual Installation
------------------

If you prefer manual installation:

.. code-block:: bash

   git clone https://github.com/username/snowdroughtindex.git
   cd snowdroughtindex

   # Create and activate virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate.bat
   # On Unix/Mac:
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements_extraction.txt
   pip install -r requirements_notebook.txt

   # Install package in development mode
   pip install -e .

Virtual Environment Activation
-----------------------------

After setup, activate your virtual environment:

**Windows Command Prompt:**

.. code-block:: batch

   venv\Scripts\activate.bat

**Windows PowerShell:**

.. code-block:: powershell

   venv\Scripts\Activate.ps1

**Unix/Mac:**

.. code-block:: bash

   source venv/bin/activate

Dependencies
-----------

The package uses three separate requirements files to manage dependencies:

Core Dependencies (requirements.txt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Essential packages for the main functionality:

* ``numpy>=1.20.0``
* ``pandas>=1.3.0``
* ``xarray>=0.19.0``
* ``matplotlib>=3.4.0``
* ``scipy>=1.7.0``
* ``geopandas>=0.10.0``
* ``seaborn>=0.11.0``
* ``scikit-learn>=1.0.0``
* ``netCDF4>=1.5.0``
* ``h5py>=3.7.0``
* ``statsmodels>=0.13.0``
* ``properscoring>=0.1``
* ``rasterio>=1.2.0``
* ``shapely>=1.8.0``

Extraction Tools (requirements_extraction.txt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized packages for data extraction tools:

* ``numpy>=1.21.0``
* ``pandas>=1.3.0``
* ``xarray>=0.19.0``
* ``geopandas>=0.10.0``
* ``shapely>=1.8.0``
* ``netcdf4>=1.5.0``
* ``pyproj>=3.2.0``

Notebook Environment (requirements_notebook.txt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Packages optimized for Jupyter notebook environments:

* ``numpy<2.0.0`` (compatibility constraint)
* ``pandas>=1.3.0``
* ``xarray>=0.19.0``
* ``matplotlib>=3.3.0``
* ``netcdf4>=1.5.0``
* ``h5netcdf>=0.8.0``
* ``pathlib2>=2.3.0`` (Python <3.4)
* ``logging>=0.4.9.6``
* ``dask>=2021.6.0``
* ``scipy>=1.7.0``

.. note::
   The notebook requirements include NumPy version constraints (<2.0.0) to ensure compatibility with Jupyter notebooks and avoid NumPy 2.x compatibility issues.

Installation Verification
-------------------------

Test that the package is installed correctly:

.. code-block:: python

   python -c "import snowdroughtindex; print('Package imported successfully!')"

You can also verify the installation by running a simple test:

.. code-block:: python

   import snowdroughtindex as sdi
   
   # Check version
   print(f"Snow Drought Index version: {sdi.__version__}")
   
   # Test basic functionality
   from snowdroughtindex.core import sswei
   print("Core modules loaded successfully!")

Using the Package
----------------

After installation, you can import and use the package in your Python code:

.. code-block:: python

   import snowdroughtindex as sdi
   
   # Import specific modules
   from snowdroughtindex.core import sswei, gap_filling, drought_analysis
   from snowdroughtindex.utils import visualization, statistics
   
   # Example: Load and process SWE data
   # data = sdi.core.data_preparation.load_swe_data('path/to/data.nc')

Troubleshooting
--------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Permission Errors**

If you encounter permission errors during installation:

.. code-block:: bash

   # Use --user flag for user-level installation
   pip install --user -e .

**2. Virtual Environment Issues**

If virtual environment creation fails:

.. code-block:: bash

   # Try using python3 explicitly
   python3 -m venv venv
   
   # Or use virtualenv
   pip install virtualenv
   virtualenv venv

**3. Dependency Conflicts**

If you encounter dependency conflicts:

.. code-block:: bash

   # Create a fresh virtual environment
   rm -rf venv  # or rmdir /s venv on Windows
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install -r requirements.txt

**4. NumPy 2.x Compatibility Issues**

If you encounter NumPy 2.x compatibility issues in notebooks:

.. code-block:: bash

   # Install notebook-specific requirements
   pip install -r requirements_notebook.txt

**5. GeoPandas Installation Issues**

GeoPandas can be challenging to install. If you encounter issues:

.. code-block:: bash

   # On Windows, consider using conda
   conda install geopandas
   
   # Or install GDAL first
   pip install GDAL
   pip install geopandas

Memory and Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets:

* **Memory**: Ensure you have sufficient RAM (8GB+ recommended)
* **Storage**: Large NetCDF files may require significant disk space
* **Processing**: Consider using the chunked processing options for very large datasets

Development Installation
-----------------------

For developers who want to contribute to the package:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/username/snowdroughtindex.git
   cd snowdroughtindex
   
   # Create development environment
   python -m venv dev-env
   source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows
   
   # Install in development mode with all dependencies
   pip install -e .
   pip install -r requirements.txt
   pip install -r requirements_extraction.txt
   pip install -r requirements_notebook.txt
   
   # Install development tools (optional)
   pip install pytest black flake8 sphinx

Docker Installation (Advanced)
-----------------------------

For containerized deployment:

.. code-block:: dockerfile

   FROM python:3.9-slim
   
   WORKDIR /app
   COPY . .
   
   RUN pip install -r requirements.txt
   RUN pip install -e .
   
   CMD ["python", "-c", "import snowdroughtindex; print('Ready!')"]

.. code-block:: bash

   # Build and run
   docker build -t snowdroughtindex .
   docker run snowdroughtindex

Next Steps
---------

After successful installation:

1. **Read the Quickstart Guide**: :doc:`quickstart` for basic usage examples
2. **Explore Workflows**: :doc:`workflows/index` for detailed analysis workflows  
3. **Check Examples**: :doc:`examples` for practical implementation examples
4. **Review Tools**: :doc:`tools/index` for data processing utilities

.. seealso::
   
   - :doc:`quickstart` for getting started with the package
   - :doc:`workflows/index` for detailed analysis workflows
   - :doc:`tools/index` for data processing tools
   - :doc:`performance_optimization` for optimization tips
