"""
Statistics module for the Snow Drought Index package.

This module contains functions for statistical analysis of SWE data,
including circular statistics, principal component analysis, and correlation analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Union, List, Dict, Tuple, Optional, Any


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize data by calculating z-scores.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to be standardized.
        
    Returns
    -------
    np.ndarray
        Standardized data (z-scores).
    """
    return (data - np.mean(data)) / np.std(data)


def calculate_anomaly(data: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate anomalies from a reference value or the mean of the data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for which to calculate anomalies.
    reference : np.ndarray, optional
        Reference values to calculate anomalies from. If None, the mean of data is used.
        
    Returns
    -------
    np.ndarray
        Anomalies from the reference value or mean.
    """
    if reference is None:
        reference = np.mean(data)
    return data - reference


def fit_regression(x: np.ndarray, y: np.ndarray, degree: int = 1) -> Tuple[np.ndarray, np.poly1d]:
    """
    Fit a polynomial regression to data points.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    degree : int, optional
        Degree of the polynomial to fit. Default is 1 (linear regression).
        
    Returns
    -------
    Tuple[np.ndarray, np.poly1d]
        Tuple containing the polynomial coefficients and the polynomial function.
    """
    coeffs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coeffs)
    return coeffs, poly_func


def cluster_snow_drought(data: pd.DataFrame, features: List[str], n_clusters: int = 3) -> Tuple[pd.Series, np.ndarray]:
    """
    Perform K-means clustering for snow drought classification.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to cluster.
    features : List[str]
        List of column names to use as features for clustering.
    n_clusters : int, optional
        Number of clusters to form. Default is 3.
        
    Returns
    -------
    Tuple[pd.Series, np.ndarray]
        Tuple containing the cluster assignments and cluster centers.
    """
    # Extract features for clustering
    X = data[features].values
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X)
    
    return pd.Series(clusters), kmeans.cluster_centers_


def gringorten_probabilities(values: np.ndarray) -> np.ndarray:
    """
    Compute Gringorten plotting position probabilities.
    
    Parameters
    ----------
    values : np.ndarray
        Array of values to compute probabilities for.
        
    Returns
    -------
    np.ndarray
        Gringorten probabilities.
    """
    # Get the ranks (1-based) of the values from smallest to largest
    ranks = np.argsort(np.argsort(values)) + 1
    n = len(values)
    probabilities = (ranks - 0.44) / (n + 0.12)
    return probabilities


def compute_swei(probabilities: np.ndarray) -> np.ndarray:
    """
    Transform probabilities to SWEI using the inverse normal distribution.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities to transform.
        
    Returns
    -------
    np.ndarray
        SWEI values.
    """
    return stats.norm.ppf(probabilities)


def integrate_season(data: pd.DataFrame, time_col: str = 'date', value_col: str = 'mean_SWE') -> float:
    """
    Integrates values over time using the trapezoidal rule.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing time and value columns.
    time_col : str, optional
        Name of the column containing time values. Default is 'date'.
    value_col : str, optional
        Name of the column containing values to integrate. Default is 'mean_SWE'.
        
    Returns
    -------
    float
        Integrated value over the time period.
    """
    # Ensure data is sorted by time
    data = data.sort_values(by=time_col)
    
    # Convert dates to numerical days since start
    days_since_start = (data[time_col] - data[time_col].min()).dt.days
    
    # Integrate values over the period using numpy's trapz function
    total_integration = np.trapz(data[value_col], days_since_start)
    
    return total_integration


def circular_mean(angles: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate the circular mean of a set of angles.
    
    Parameters
    ----------
    angles : np.ndarray
        Array of angles.
    degrees : bool, optional
        If True, angles are in degrees. If False, angles are in radians. Default is True.
        
    Returns
    -------
    float
        Circular mean angle in the same units as input.
    """
    if degrees:
        angles_rad = np.radians(angles)
    else:
        angles_rad = angles.copy()
    
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    
    # Handle the case where the angles are evenly distributed around the circle
    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
        return 0.0 if degrees else 0.0
    
    mean_angle = np.arctan2(sin_sum, cos_sum)
    
    if degrees:
        mean_angle = np.degrees(mean_angle)
        # Ensure the result is in [0, 360)
        mean_angle = (mean_angle + 360) % 360
    
    return mean_angle


def circular_variance(angles: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate the circular variance of a set of angles.
    
    Parameters
    ----------
    angles : np.ndarray
        Array of angles.
    degrees : bool, optional
        If True, angles are in degrees. If False, angles are in radians. Default is True.
        
    Returns
    -------
    float
        Circular variance (between 0 and 1).
    """
    if degrees:
        angles = np.radians(angles)
    
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    
    return 1 - r


def perform_pca(data: np.ndarray, n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis on the data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix (samples x features).
    n_components : int, optional
        Number of components to keep. If None, all components are kept.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the transformed data, explained variance ratios, and components.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    
    return transformed_data, pca.explained_variance_ratio_, pca.components_
