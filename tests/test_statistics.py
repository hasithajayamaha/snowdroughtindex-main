"""
Unit tests for the statistics module.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from scipy import stats
from sklearn.datasets import make_blobs

from snowdroughtindex.utils.statistics import (
    standardize,
    calculate_anomaly,
    fit_regression,
    cluster_snow_drought,
    gringorten_probabilities,
    compute_swei,
    integrate_season,
    circular_mean,
    circular_variance,
    perform_pca
)


class TestStandardize:
    """Tests for the standardize function."""

    def test_standardize_basic(self):
        """Test basic standardization."""
        data = np.array([1, 2, 3, 4, 5])
        result = standardize(data)
        
        # Calculate expected result manually
        expected = (data - np.mean(data)) / np.std(data)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_standardize_constant(self):
        """Test standardization with constant data."""
        data = np.array([5, 5, 5, 5, 5])
        result = standardize(data)
        
        # For constant data, std is 0, so result should be NaN
        assert np.all(np.isnan(result))
    
    def test_standardize_2d(self):
        """Test standardization with 2D data."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = standardize(data)
        
        # Calculate expected result manually
        expected = (data - np.mean(data)) / np.std(data)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestCalculateAnomaly:
    """Tests for the calculate_anomaly function."""

    def test_calculate_anomaly_default(self):
        """Test anomaly calculation with default reference."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_anomaly(data)
        
        # Calculate expected result manually
        expected = data - np.mean(data)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_calculate_anomaly_custom_reference(self):
        """Test anomaly calculation with custom reference."""
        data = np.array([1, 2, 3, 4, 5])
        reference = 2
        result = calculate_anomaly(data, reference)
        
        # Calculate expected result manually
        expected = data - reference
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_calculate_anomaly_array_reference(self):
        """Test anomaly calculation with array reference."""
        data = np.array([1, 2, 3, 4, 5])
        reference = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        result = calculate_anomaly(data, reference)
        
        # Calculate expected result manually
        expected = data - reference
        
        np.testing.assert_array_almost_equal(result, expected)


class TestFitRegression:
    """Tests for the fit_regression function."""

    def test_fit_regression_linear(self):
        """Test linear regression fitting."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        coeffs, poly_func = fit_regression(x, y)
        
        # Expected coefficients for y = 2x are [2, 0]
        np.testing.assert_array_almost_equal(coeffs, [2, 0])
        
        # Test the polynomial function
        np.testing.assert_array_almost_equal(poly_func(x), y)
    
    def test_fit_regression_quadratic(self):
        """Test quadratic regression fitting."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])  # y = x^2
        
        coeffs, poly_func = fit_regression(x, y, degree=2)
        
        # Expected coefficients for y = x^2 are [1, 0, 0]
        np.testing.assert_array_almost_equal(coeffs, [1, 0, 0])
        
        # Test the polynomial function
        np.testing.assert_array_almost_equal(poly_func(x), y)


class TestClusterSnowDrought:
    """Tests for the cluster_snow_drought function."""

    def test_cluster_snow_drought(self):
        """Test snow drought clustering."""
        # Create synthetic data with 3 clusters
        X, y = make_blobs(n_samples=100, centers=3, random_state=0)
        
        # Create a DataFrame with the synthetic data
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        
        # Perform clustering
        clusters, centers = cluster_snow_drought(df, ['feature1', 'feature2'])
        
        # Check that we get the expected number of clusters
        assert len(np.unique(clusters)) == 3
        
        # Check that the cluster centers have the right shape
        assert centers.shape == (3, 2)


class TestGringortenProbabilities:
    """Tests for the gringorten_probabilities function."""

    def test_gringorten_probabilities(self):
        """Test Gringorten probability calculation."""
        values = np.array([10, 20, 30, 40, 50])
        result = gringorten_probabilities(values)
        
        # Calculate expected probabilities manually
        n = len(values)
        ranks = np.array([1, 2, 3, 4, 5])
        expected = (ranks - 0.44) / (n + 0.12)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_gringorten_probabilities_unsorted(self):
        """Test Gringorten probability calculation with unsorted data."""
        values = np.array([30, 10, 50, 20, 40])
        result = gringorten_probabilities(values)
        
        # Calculate expected probabilities manually
        n = len(values)
        # Ranks should be [3, 1, 5, 2, 4] for the unsorted values
        # 10 -> rank 1, 20 -> rank 2, 30 -> rank 3, 40 -> rank 4, 50 -> rank 5
        ranks = np.array([3, 1, 5, 2, 4])
        expected = (ranks - 0.44) / (n + 0.12)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeSWEI:
    """Tests for the compute_swei function."""

    def test_compute_swei(self):
        """Test SWEI computation."""
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = compute_swei(probabilities)
        
        # Calculate expected SWEI values using scipy.stats.norm.ppf
        expected = stats.norm.ppf(probabilities)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_compute_swei_extreme(self):
        """Test SWEI computation with extreme probabilities."""
        probabilities = np.array([0.01, 0.99])
        result = compute_swei(probabilities)
        
        # Calculate expected SWEI values using scipy.stats.norm.ppf
        expected = stats.norm.ppf(probabilities)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestIntegrateSeason:
    """Tests for the integrate_season function."""

    def test_integrate_season(self):
        """Test season integration."""
        # Create a sample DataFrame with dates and values
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = [i**2 for i in range(10)]  # Quadratic function
        
        df = pd.DataFrame({'date': dates, 'mean_SWE': values})
        
        result = integrate_season(df)
        
        # For a quadratic function y = x^2 from 0 to 9,
        # the integral is x^3/3 evaluated from 0 to 9,
        # which is 9^3/3 = 243
        # But since we're using the trapezoidal rule, there will be some numerical error
        expected = 243
        
        # Allow for some numerical error in the trapezoidal integration
        assert abs(result - expected) < 5
    
    def test_integrate_season_custom_columns(self):
        """Test season integration with custom column names."""
        # Create a sample DataFrame with dates and values
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = [i for i in range(10)]  # Linear function
        
        df = pd.DataFrame({'time': dates, 'value': values})
        
        result = integrate_season(df, time_col='time', value_col='value')
        
        # For a linear function y = x from 0 to 9,
        # the integral is x^2/2 evaluated from 0 to 9,
        # which is 9^2/2 = 40.5
        expected = 40.5
        
        # Allow for some numerical error in the trapezoidal integration
        assert abs(result - expected) < 1


class TestCircularStatistics:
    """Tests for circular statistics functions."""

    def test_circular_mean_degrees(self):
        """Test circular mean calculation with degrees."""
        angles = np.array([0, 90, 180, 270])
        result = circular_mean(angles)
        
        # For these angles, the circular mean should be close to 0 (or 360)
        assert abs(result) < 1e-10 or abs(result - 360) < 1e-10
    
    def test_circular_mean_radians(self):
        """Test circular mean calculation with radians."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        result = circular_mean(angles, degrees=False)
        
        # For these angles, the circular mean should be close to 0 (or 2π)
        assert abs(result) < 1e-10 or abs(result - 2*np.pi) < 1e-10
    
    def test_circular_variance_degrees(self):
        """Test circular variance calculation with degrees."""
        # All angles the same - variance should be 0
        angles1 = np.array([45, 45, 45, 45])
        result1 = circular_variance(angles1)
        assert result1 == pytest.approx(0)
        
        # Angles evenly distributed around the circle - variance should be 1
        angles2 = np.array([0, 90, 180, 270])
        result2 = circular_variance(angles2)
        assert result2 == pytest.approx(1)
    
    def test_circular_variance_radians(self):
        """Test circular variance calculation with radians."""
        # All angles the same - variance should be 0
        angles1 = np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4])
        result1 = circular_variance(angles1, degrees=False)
        assert result1 == pytest.approx(0)
        
        # Angles evenly distributed around the circle - variance should be 1
        angles2 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        result2 = circular_variance(angles2, degrees=False)
        assert result2 == pytest.approx(1)


class TestPerformPCA:
    """Tests for the perform_pca function."""

    def test_perform_pca(self):
        """Test PCA computation."""
        # Create synthetic data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        
        # Perform PCA
        transformed, explained_variance, components = perform_pca(X)
        
        # Check shapes
        assert transformed.shape[0] == X.shape[0]
        assert len(explained_variance) == min(X.shape)
        assert components.shape[1] == X.shape[1]
        
        # Check that explained variance sums to 1
        assert np.sum(explained_variance) == pytest.approx(1.0)
    
    def test_perform_pca_n_components(self):
        """Test PCA computation with specified number of components."""
        # Create synthetic data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        
        # Perform PCA with 2 components
        transformed, explained_variance, components = perform_pca(X, n_components=2)
        
        # Check shapes
        assert transformed.shape == (X.shape[0], 2)
        assert len(explained_variance) == 2
        assert components.shape == (2, X.shape[1])
