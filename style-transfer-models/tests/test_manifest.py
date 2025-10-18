"""
Unit tests for manifest generation and validation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'manifest'))

from generate_manifest import create_style_descriptor, validate_manifest


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_manifest(temp_dir):
    """Create a sample manifest for testing."""
    manifest = {
        "version": "1.0.0",
        "updated": "2025-10-18T00:00:00Z",
        "styles": [
            {
                "id": "test-style",
                "displayName": "Test Style",
                "description": "A test style",
                "category": "abstract",
                "modelPath": "../models/exported/test.mlmodel",
                "inputSize": {"width": 256, "height": 256},
                "outputSize": {"width": 256, "height": 256},
                "computeUnits": "ALL",
                "enabled": True
            }
        ]
    }
    
    manifest_path = temp_dir / 'test_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


def test_create_style_descriptor(temp_dir):
    """Test creating a StyleDescriptor."""
    descriptor = create_style_descriptor(
        style_id='sci-fi',
        display_name='Sci-Fi',
        category='sci-fi',
        model_path=str(temp_dir / 'test.mlmodel'),
        description='Test description',
        compute_units='ALL'
    )
    
    # Check required fields
    assert descriptor['id'] == 'sci-fi'
    assert descriptor['displayName'] == 'Sci-Fi'
    assert descriptor['category'] == 'sci-fi'
    assert 'modelPath' in descriptor
    assert 'inputSize' in descriptor
    assert 'computeUnits' in descriptor
    
    # Check input size defaults
    assert descriptor['inputSize']['width'] == 256
    assert descriptor['inputSize']['height'] == 256


def test_validate_manifest_structure(sample_manifest):
    """Test manifest structure validation."""
    # Should validate successfully
    is_valid = validate_manifest(sample_manifest)
    assert is_valid
    
    # Read and verify structure
    with open(sample_manifest, 'r') as f:
        manifest = json.load(f)
    
    assert 'version' in manifest
    assert 'updated' in manifest
    assert 'styles' in manifest
    assert isinstance(manifest['styles'], list)


def test_validate_manifest_required_fields(temp_dir):
    """Test validation of required fields."""
    # Missing required field
    invalid_manifest = {
        "version": "1.0.0",
        "updated": "2025-10-18T00:00:00Z",
        "styles": [
            {
                "id": "test",
                "displayName": "Test",
                # Missing category
                "modelPath": "test.mlmodel",
                "inputSize": {"width": 256, "height": 256},
                "computeUnits": "ALL"
            }
        ]
    }
    
    manifest_path = temp_dir / 'invalid_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(invalid_manifest, f)
    
    # Should fail validation
    is_valid = validate_manifest(manifest_path)
    assert not is_valid


def test_style_descriptor_categories():
    """Test valid style categories."""
    valid_categories = [
        'sci-fi', 'fantasy', 'modern', 'abstract',
        'impressionism', 'expressionism', 'other'
    ]
    
    for category in valid_categories:
        descriptor = create_style_descriptor(
            style_id='test',
            display_name='Test',
            category=category,
            model_path='test.mlmodel'
        )
        assert descriptor['category'] == category


def test_compute_units_options():
    """Test valid compute units."""
    valid_units = ['ALL', 'CPU_AND_GPU', 'CPU_ONLY', 'CPU_AND_NE']
    
    for units in valid_units:
        descriptor = create_style_descriptor(
            style_id='test',
            display_name='Test',
            category='abstract',
            model_path='test.mlmodel',
            compute_units=units
        )
        assert descriptor['computeUnits'] == units


def test_manifest_json_serialization(temp_dir):
    """Test manifest can be serialized and deserialized."""
    manifest = {
        "version": "1.0.0",
        "updated": "2025-10-18T00:00:00Z",
        "styles": [
            create_style_descriptor(
                style_id='test',
                display_name='Test',
                category='abstract',
                model_path='test.mlmodel'
            )
        ]
    }
    
    # Serialize
    manifest_path = temp_dir / 'test.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Deserialize
    with open(manifest_path, 'r') as f:
        loaded = json.load(f)
    
    # Verify
    assert loaded['version'] == manifest['version']
    assert len(loaded['styles']) == 1
    assert loaded['styles'][0]['id'] == 'test'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
