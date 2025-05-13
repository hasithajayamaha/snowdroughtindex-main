"""
Data validation and quality control module for climate data.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating and quality controlling climate data."""
    
    def __init__(self, 
                 missing_value_threshold: float = 0.2,
                 temporal_gap_threshold: int = 5,
                 spatial_coverage_threshold: float = 0.8,
                 value_range: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the DataValidator.
        
        Parameters
        ----------
        missing_value_threshold : float, optional
            Maximum allowed fraction of missing values (default: 0.2)
        temporal_gap_threshold : int, optional
            Maximum allowed gap in time series (default: 5)
        spatial_coverage_threshold : float, optional
            Minimum required spatial coverage (default: 0.8)
        value_range : dict, optional
            Dictionary mapping variable names to (min, max) value ranges
        """
        self.missing_value_threshold = missing_value_threshold
        self.temporal_gap_threshold = temporal_gap_threshold
        self.spatial_coverage_threshold = spatial_coverage_threshold
        self.value_range = value_range or {}
    
    def validate_dataset(self, dataset: xr.Dataset) -> Dict[str, List[str]]:
        """
        Validate a dataset for quality and consistency.
        
        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to validate
            
        Returns
        -------
        dict
            Dictionary containing validation results and issues found
        """
        issues = {}
        
        # Check for required variables
        required_vars = ['time', 'lat', 'lon']
        for var in required_vars:
            if var not in dataset.variables:
                issues.setdefault('missing_variables', []).append(var)
        
        # Check data types
        if 'time' in dataset.variables and not np.issubdtype(dataset.time.dtype, np.datetime64):
            issues.setdefault('data_types', []).append('time variable must be datetime64')
        
        # Check for missing values
        for var in dataset.data_vars:
            missing_frac = np.isnan(dataset[var]).mean().item()
            if missing_frac > self.missing_value_threshold:
                issues.setdefault('missing_values', []).append(
                    f"{var}: {missing_frac:.2%} missing values"
                )
        
        # Check temporal consistency
        if 'time' in dataset.variables:
            time_gaps = np.diff(dataset.time.values)
            if len(time_gaps) > 0:
                max_gap = np.max(time_gaps)
                if max_gap > np.timedelta64(self.temporal_gap_threshold, 'D'):
                    issues.setdefault('temporal_gaps', []).append(
                        f"Maximum time gap: {max_gap}"
                    )
        
        # Check spatial coverage
        if 'lat' in dataset.variables and 'lon' in dataset.variables:
            coverage = (~np.isnan(dataset[list(dataset.data_vars)[0]])).mean().item()
            if coverage < self.spatial_coverage_threshold:
                issues.setdefault('spatial_coverage', []).append(
                    f"Spatial coverage: {coverage:.2%}"
                )
        
        # Check value ranges
        for var, (min_val, max_val) in self.value_range.items():
            if var in dataset.variables:
                if np.any(dataset[var] < min_val) or np.any(dataset[var] > max_val):
                    issues.setdefault('value_ranges', []).append(
                        f"{var}: values outside range [{min_val}, {max_val}]"
                    )
        
        return issues
    
    def quality_control(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply quality control measures to the dataset.
        
        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to quality control
            
        Returns
        -------
        xarray.Dataset
            Quality-controlled dataset
        """
        qc_dataset = dataset.copy()
        
        # Apply value range constraints
        for var, (min_val, max_val) in self.value_range.items():
            if var in qc_dataset.variables:
                qc_dataset[var] = qc_dataset[var].where(
                    (qc_dataset[var] >= min_val) & (qc_dataset[var] <= max_val)
                )
        
        # Log quality control actions
        for var in qc_dataset.data_vars:
            original_missing = np.isnan(dataset[var]).sum().item()
            new_missing = np.isnan(qc_dataset[var]).sum().item()
            if new_missing > original_missing:
                logger.info(
                    f"Quality control: {var} - {new_missing - original_missing} "
                    f"additional values marked as missing"
                )
        
        return qc_dataset

def validate_time_series(data: Union[np.ndarray, xr.DataArray], 
                        time: np.ndarray) -> Dict[str, List[str]]:
    """
    Validate a time series for quality and consistency.
    
    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Time series data
    time : numpy.ndarray
        Time coordinates
        
    Returns
    -------
    dict
        Dictionary containing validation results and issues found
    """
    issues = {}
    
    # Check for monotonic time
    if not np.all(np.diff(time) > 0):
        issues['time_monotonicity'] = ['Time series is not strictly increasing']
    
    # Check for missing values
    missing_frac = np.isnan(data).mean()
    if missing_frac > 0.2:
        issues['missing_values'] = [f'Missing value fraction: {missing_frac:.2%}']
    
    # Check for outliers using IQR method
    if isinstance(data, np.ndarray):
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((data < lower_bound) | (data > upper_bound))
        if outliers > 0:
            issues['outliers'] = [f'Found {outliers} potential outliers']
    
    return issues 