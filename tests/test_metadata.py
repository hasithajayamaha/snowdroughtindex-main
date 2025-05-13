"""
Tests for the metadata and provenance tracking module.
"""

import os
import tempfile
import json
import numpy as np
import xarray as xr
import pytest
from datetime import datetime
from snowdroughtindex.core.metadata import (
    ProvenanceTracker,
    add_metadata_to_dataset,
    get_dataset_metadata,
    calculate_data_hash,
    verify_data_integrity
)

def create_test_dataset():
    """Create a test dataset."""
    data = np.random.rand(10, 10)
    ds = xr.Dataset(
        data_vars=dict(temperature=(["x", "y"], data)),
        coords=dict(
            x=np.arange(10),
            y=np.arange(10)
        )
    )
    return ds

def test_provenance_tracker_initialization():
    """Test ProvenanceTracker initialization."""
    tracker = ProvenanceTracker(
        creator="Test User",
        institution="Test Institution",
        email="test@example.com",
        version="1.0.0"
    )
    
    assert tracker.creator == "Test User"
    assert tracker.institution == "Test Institution"
    assert tracker.email == "test@example.com"
    assert tracker.version == "1.0.0"
    assert len(tracker.processing_history) == 0

def test_add_processing_step():
    """Test adding a processing step."""
    tracker = ProvenanceTracker(
        creator="Test User",
        institution="Test Institution",
        email="test@example.com"
    )
    
    step_id = tracker.add_processing_step(
        name="Test Processing",
        description="Test description",
        parameters={"param1": 1, "param2": "value"},
        input_files=["input.nc"],
        output_files=["output.nc"]
    )
    
    assert len(tracker.processing_history) == 1
    step = tracker.processing_history[0]
    assert step['id'] == step_id
    assert step['name'] == "Test Processing"
    assert step['description'] == "Test description"
    assert step['parameters'] == {"param1": 1, "param2": "value"}
    assert step['input_files'] == ["input.nc"]
    assert step['output_files'] == ["output.nc"]

def test_save_load_provenance():
    """Test saving and loading provenance."""
    tracker = ProvenanceTracker(
        creator="Test User",
        institution="Test Institution",
        email="test@example.com"
    )
    
    tracker.add_processing_step(
        name="Test Processing",
        description="Test description",
        parameters={"param1": 1}
    )
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        # Save provenance
        tracker.save_provenance(tmp.name)
        
        # Create new tracker and load provenance
        new_tracker = ProvenanceTracker(
            creator="Different User",
            institution="Different Institution",
            email="different@example.com"
        )
        new_tracker.load_provenance(tmp.name)
        
        # Check that provenance was loaded correctly
        assert new_tracker.creator == "Test User"
        assert new_tracker.institution == "Test Institution"
        assert new_tracker.email == "test@example.com"
        assert len(new_tracker.processing_history) == 1
        
        # Clean up
        os.unlink(tmp.name)

def test_add_metadata_to_dataset():
    """Test adding metadata to a dataset."""
    ds = create_test_dataset()
    metadata = {
        "title": "Test Dataset",
        "author": "Test User",
        "description": "Test description",
        "parameters": {"param1": 1, "param2": "value"},
        "array": np.array([1, 2, 3])
    }
    
    ds_with_metadata = add_metadata_to_dataset(ds, metadata)
    
    assert ds_with_metadata.attrs["title"] == "Test Dataset"
    assert ds_with_metadata.attrs["author"] == "Test User"
    assert ds_with_metadata.attrs["description"] == "Test description"
    assert json.loads(ds_with_metadata.attrs["parameters"]) == {"param1": 1, "param2": "value"}
    assert ds_with_metadata.attrs["array"] == [1, 2, 3]

def test_get_dataset_metadata():
    """Test getting metadata from a dataset."""
    ds = create_test_dataset()
    ds.attrs["title"] = "Test Dataset"
    ds.attrs["parameters"] = json.dumps({"param1": 1})
    
    metadata = get_dataset_metadata(ds)
    
    assert metadata["dimensions"] == {"x": 10, "y": 10}
    assert metadata["variables"] == ["temperature"]
    assert metadata["coordinates"] == ["x", "y"]
    assert metadata["title"] == "Test Dataset"
    assert metadata["parameters"] == {"param1": 1}

def test_data_hash():
    """Test data hash calculation and verification."""
    ds = create_test_dataset()
    
    # Calculate hash
    hash_value = calculate_data_hash(ds)
    
    # Verify hash
    assert verify_data_integrity(ds, hash_value)
    
    # Modify dataset
    ds.temperature[0, 0] = 999
    
    # Verify hash fails
    assert not verify_data_integrity(ds, hash_value)

def test_provenance_timestamp():
    """Test that processing step timestamps are valid."""
    tracker = ProvenanceTracker(
        creator="Test User",
        institution="Test Institution",
        email="test@example.com"
    )
    
    tracker.add_processing_step(
        name="Test Processing",
        description="Test description",
        parameters={}
    )
    
    step = tracker.processing_history[0]
    timestamp = datetime.fromisoformat(step['timestamp'])
    
    # Check that timestamp is recent (within last minute)
    assert (datetime.now() - timestamp).total_seconds() < 60 