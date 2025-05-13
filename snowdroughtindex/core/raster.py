"""
Raster data handling module for GeoTIFF and other raster formats.
"""

import rasterio
import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

def read_raster(filepath: Union[str, Path], 
                band: Optional[int] = None,
                masked: bool = True) -> xr.DataArray:
    """
    Read a raster file (GeoTIFF or other supported format).
    
    Parameters
    ----------
    filepath : str or Path
        Path to the raster file
    band : int, optional
        Specific band to read (default: all bands)
    masked : bool, optional
        Whether to mask nodata values (default: True)
        
    Returns
    -------
    xarray.DataArray
        The loaded raster data with geospatial metadata
    """
    try:
        logger.info(f"Reading raster file: {filepath}")
        with rasterio.open(filepath) as src:
            # Get metadata
            meta = src.meta
            transform = src.transform
            crs = src.crs
            
            # Read data
            if band is not None:
                data = src.read(band, masked=masked)
            else:
                data = src.read(masked=masked)
            
            # Create coordinates
            height, width = data.shape[-2:]
            x = np.arange(width) * transform.a + transform.c
            y = np.arange(height) * transform.e + transform.f
            
            # Create DataArray
            if len(data.shape) == 2:
                da = xr.DataArray(
                    data,
                    dims=('y', 'x'),
                    coords={'y': y, 'x': x},
                    attrs={
                        'transform': transform,
                        'crs': crs,
                        'nodata': meta.get('nodata'),
                        'driver': meta.get('driver'),
                        'dtype': meta.get('dtype')
                    }
                )
            else:
                da = xr.DataArray(
                    data,
                    dims=('band', 'y', 'x'),
                    coords={'band': np.arange(1, data.shape[0] + 1), 'y': y, 'x': x},
                    attrs={
                        'transform': transform,
                        'crs': crs,
                        'nodata': meta.get('nodata'),
                        'driver': meta.get('driver'),
                        'dtype': meta.get('dtype')
                    }
                )
            
            return da
            
    except Exception as e:
        logger.error(f"Error reading raster file {filepath}: {str(e)}")
        raise

def write_raster(data: Union[xr.DataArray, np.ndarray],
                 filepath: Union[str, Path],
                 crs: Optional[str] = None,
                 transform: Optional[rasterio.Affine] = None,
                 nodata: Optional[float] = None,
                 driver: str = 'GTiff',
                 **kwargs) -> None:
    """
    Write data to a raster file.
    
    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Data to write
    filepath : str or Path
        Path where to save the raster file
    crs : str, optional
        Coordinate reference system (default: from data if available)
    transform : rasterio.Affine, optional
        Affine transformation (default: from data if available)
    nodata : float, optional
        Nodata value (default: from data if available)
    driver : str, optional
        GDAL driver to use (default: 'GTiff')
    **kwargs : dict
        Additional arguments passed to rasterio.open
    """
    try:
        logger.info(f"Writing raster file: {filepath}")
        
        # Convert xarray to numpy if needed
        if isinstance(data, xr.DataArray):
            array = data.values
            if crs is None and 'crs' in data.attrs:
                crs = data.attrs['crs']
            if transform is None and 'transform' in data.attrs:
                transform = data.attrs['transform']
            if nodata is None and 'nodata' in data.attrs:
                nodata = data.attrs['nodata']
        else:
            array = data
        
        # Ensure array is 3D (bands, height, width)
        if len(array.shape) == 2:
            array = array[np.newaxis, ...]
        
        # Get metadata
        height, width = array.shape[-2:]
        count = array.shape[0]
        dtype = array.dtype
        
        # Create metadata
        meta = {
            'driver': driver,
            'height': height,
            'width': width,
            'count': count,
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'nodata': nodata
        }
        
        # Update with additional kwargs
        meta.update(kwargs)
        
        # Write file
        with rasterio.open(filepath, 'w', **meta) as dst:
            dst.write(array)
            
    except Exception as e:
        logger.error(f"Error writing raster file {filepath}: {str(e)}")
        raise

def raster_to_dataset(raster: xr.DataArray,
                      name: Optional[str] = None) -> xr.Dataset:
    """
    Convert a raster DataArray to a Dataset.
    
    Parameters
    ----------
    raster : xarray.DataArray
        Raster data to convert
    name : str, optional
        Name for the variable in the dataset
        
    Returns
    -------
    xarray.Dataset
        Dataset containing the raster data
    """
    if name is None:
        name = 'data'
    
    # Create dataset
    ds = xr.Dataset({name: raster})
    
    # Add metadata
    if 'crs' in raster.attrs:
        ds.attrs['crs'] = raster.attrs['crs']
    if 'transform' in raster.attrs:
        ds.attrs['transform'] = raster.attrs['transform']
    
    return ds

def dataset_to_raster(ds: xr.Dataset,
                      variable: str,
                      crs: Optional[str] = None,
                      transform: Optional[rasterio.Affine] = None) -> xr.DataArray:
    """
    Convert a dataset variable to a raster DataArray.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable
    variable : str
        Name of the variable to convert
    crs : str, optional
        Coordinate reference system (default: from dataset if available)
    transform : rasterio.Affine, optional
        Affine transformation (default: from dataset if available)
        
    Returns
    -------
    xarray.DataArray
        Raster data array
    """
    # Get variable
    da = ds[variable]
    
    # Add metadata
    if crs is not None:
        da.attrs['crs'] = crs
    elif 'crs' in ds.attrs:
        da.attrs['crs'] = ds.attrs['crs']
    
    if transform is not None:
        da.attrs['transform'] = transform
    elif 'transform' in ds.attrs:
        da.attrs['transform'] = ds.attrs['transform']
    
    return da 