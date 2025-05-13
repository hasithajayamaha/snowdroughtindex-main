"""
Metadata handling and provenance tracking module.
"""

import json
import datetime
import uuid
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import hashlib
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)

class ProvenanceTracker:
    """Class for tracking data provenance and metadata."""
    
    def __init__(self, 
                 creator: str,
                 institution: str,
                 email: str,
                 version: str = "1.0.0"):
        """
        Initialize the ProvenanceTracker.
        
        Parameters
        ----------
        creator : str
            Name of the data creator
        institution : str
            Institution name
        email : str
            Contact email
        version : str, optional
            Version of the processing (default: "1.0.0")
        """
        self.creator = creator
        self.institution = institution
        self.email = email
        self.version = version
        self.processing_history: List[Dict[str, Any]] = []
    
    def add_processing_step(self,
                          name: str,
                          description: str,
                          parameters: Dict[str, Any],
                          input_files: Optional[List[str]] = None,
                          output_files: Optional[List[str]] = None) -> str:
        """
        Add a processing step to the provenance history.
        
        Parameters
        ----------
        name : str
            Name of the processing step
        description : str
            Description of what was done
        parameters : dict
            Parameters used in the processing
        input_files : list of str, optional
            List of input file paths
        output_files : list of str, optional
            List of output file paths
            
        Returns
        -------
        str
            Unique identifier for the processing step
        """
        step_id = str(uuid.uuid4())
        
        step = {
            'id': step_id,
            'name': name,
            'description': description,
            'parameters': parameters,
            'timestamp': datetime.datetime.now().isoformat(),
            'creator': self.creator,
            'institution': self.institution,
            'email': self.email,
            'version': self.version,
            'input_files': input_files or [],
            'output_files': output_files or []
        }
        
        self.processing_history.append(step)
        logger.info(f"Added processing step: {name}")
        
        return step_id
    
    def get_provenance(self) -> Dict[str, Any]:
        """
        Get the complete provenance information.
        
        Returns
        -------
        dict
            Complete provenance information
        """
        return {
            'creator': self.creator,
            'institution': self.institution,
            'email': self.email,
            'version': self.version,
            'processing_history': self.processing_history
        }
    
    def save_provenance(self, filepath: Union[str, Path]) -> None:
        """
        Save provenance information to a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path where to save the provenance file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_provenance(), f, indent=2)
            logger.info(f"Saved provenance to {filepath}")
        except Exception as e:
            logger.error(f"Error saving provenance to {filepath}: {str(e)}")
            raise
    
    def load_provenance(self, filepath: Union[str, Path]) -> None:
        """
        Load provenance information from a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the provenance file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.creator = data['creator']
            self.institution = data['institution']
            self.email = data['email']
            self.version = data['version']
            self.processing_history = data['processing_history']
            
            logger.info(f"Loaded provenance from {filepath}")
        except Exception as e:
            logger.error(f"Error loading provenance from {filepath}: {str(e)}")
            raise

def add_metadata_to_dataset(ds: xr.Dataset,
                          metadata: Dict[str, Any]) -> xr.Dataset:
    """
    Add metadata to a dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add metadata to
    metadata : dict
        Metadata to add
        
    Returns
    -------
    xarray.Dataset
        Dataset with added metadata
    """
    ds = ds.copy()
    
    # Add metadata to dataset attributes
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            ds.attrs[key] = value
        elif isinstance(value, (list, dict)):
            ds.attrs[key] = json.dumps(value)
        elif isinstance(value, np.ndarray):
            ds.attrs[key] = value.tolist()
    
    return ds

def get_dataset_metadata(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Get metadata from a dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to get metadata from
        
    Returns
    -------
    dict
        Dataset metadata
    """
    metadata = {}
    
    # Get basic metadata
    metadata['dimensions'] = dict(ds.dims)
    metadata['variables'] = list(ds.data_vars)
    metadata['coordinates'] = list(ds.coords)
    
    # Get attributes
    for key, value in ds.attrs.items():
        try:
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                metadata[key] = json.loads(value)
            else:
                metadata[key] = value
        except:
            metadata[key] = str(value)
    
    return metadata

def calculate_data_hash(data: Union[xr.Dataset, xr.DataArray]) -> str:
    """
    Calculate a hash of the data for integrity checking.
    
    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data to hash
        
    Returns
    -------
    str
        SHA-256 hash of the data
    """
    if isinstance(data, xr.Dataset):
        # Convert dataset to bytes
        data_bytes = data.to_netcdf()
    else:
        # Convert dataarray to bytes
        data_bytes = data.to_netcdf()
    
    # Calculate hash
    return hashlib.sha256(data_bytes).hexdigest()

def verify_data_integrity(data: Union[xr.Dataset, xr.DataArray],
                         expected_hash: str) -> bool:
    """
    Verify data integrity using a hash.
    
    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data to verify
    expected_hash : str
        Expected SHA-256 hash
        
    Returns
    -------
    bool
        True if hash matches, False otherwise
    """
    actual_hash = calculate_data_hash(data)
    return actual_hash == expected_hash 