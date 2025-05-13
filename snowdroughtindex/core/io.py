"""
Input/Output module for handling various file formats including NetCDF and HDF5.
"""

import xarray as xr
import h5py
import numpy as np
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def read_netcdf(filepath: str, **kwargs) -> xr.Dataset:
    """
    Read data from a NetCDF file.
    
    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    **kwargs : dict
        Additional arguments passed to xarray.open_dataset
        
    Returns
    -------
    xarray.Dataset
        The loaded dataset
    """
    try:
        logger.info(f"Reading NetCDF file: {filepath}")
        return xr.open_dataset(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Error reading NetCDF file {filepath}: {str(e)}")
        raise

def write_netcdf(dataset: xr.Dataset, filepath: str, **kwargs) -> None:
    """
    Write data to a NetCDF file.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to write
    filepath : str
        Path where to save the NetCDF file
    **kwargs : dict
        Additional arguments passed to xarray.Dataset.to_netcdf
    """
    try:
        logger.info(f"Writing NetCDF file: {filepath}")
        dataset.to_netcdf(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Error writing NetCDF file {filepath}: {str(e)}")
        raise

def read_hdf5(filepath: str, group: str = None) -> Dict[str, Any]:
    """
    Read data from an HDF5 file.
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file
    group : str, optional
        Specific group to read from the HDF5 file
        
    Returns
    -------
    dict
        Dictionary containing the data arrays
    """
    try:
        logger.info(f"Reading HDF5 file: {filepath}")
        data = {}
        with h5py.File(filepath, 'r') as f:
            if group:
                f = f[group]
            
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = np.array(obj)
            
            f.visititems(visitor)
        return data
    except Exception as e:
        logger.error(f"Error reading HDF5 file {filepath}: {str(e)}")
        raise

def write_hdf5(data: Dict[str, np.ndarray], filepath: str, group: str = None) -> None:
    """
    Write data to an HDF5 file.
    
    Parameters
    ----------
    data : dict
        Dictionary containing numpy arrays to write
    filepath : str
        Path where to save the HDF5 file
    group : str, optional
        Group name where to store the data
    """
    try:
        logger.info(f"Writing HDF5 file: {filepath}")
        with h5py.File(filepath, 'w') as f:
            if group:
                f = f.create_group(group)
            
            for name, array in data.items():
                f.create_dataset(name, data=array)
    except Exception as e:
        logger.error(f"Error writing HDF5 file {filepath}: {str(e)}")
        raise 