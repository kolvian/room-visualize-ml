"""
Unit tests for dataset preparation scripts
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'datasets' / 'scripts'))

from download_prepare import DatasetDownloader


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_images(temp_dir):
    """Create sample images for testing."""
    images_dir = temp_dir / 'sample_images'
    images_dir.mkdir(parents=True)
    
    # Create a few test images
    for i in range(5):
        img = Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(images_dir / f'test_{i:03d}.jpg')
    
    return images_dir


class TestDatasetDownloader:
    """Tests for DatasetDownloader class."""
    
    def test_init(self, temp_dir):
        """Test DatasetDownloader initialization."""
        downloader = DatasetDownloader(str(temp_dir))
        assert downloader.data_dir == temp_dir
        assert temp_dir.exists()
    
    def test_preprocess_images(self, temp_dir, sample_images):
        """Test image preprocessing and splitting."""
        downloader = DatasetDownloader(str(temp_dir))
        output_dir = temp_dir / 'processed'
        
        metadata = downloader.preprocess_images(
            sample_images,
            output_dir,
            target_size=(64, 64),
            split_ratios=(0.6, 0.2, 0.2)
        )
        
        # Check metadata
        assert metadata['total_images'] == 5
        assert metadata['target_size'] == (64, 64)
        assert sum(metadata['splits'].values()) == 5
        
        # Check output structure
        assert (output_dir / 'train').exists()
        assert (output_dir / 'val').exists()
        assert (output_dir / 'test').exists()
        assert (output_dir / 'metadata.json').exists()
        
        # Check image processing
        train_images = list((output_dir / 'train').glob('*.jpg'))
        assert len(train_images) == 3  # 60% of 5
        
        # Verify image size
        img = Image.open(train_images[0])
        assert img.size == (64, 64)
    
    def test_prepare_style_images(self, temp_dir, sample_images):
        """Test style image preparation."""
        # Create category structure
        styles_dir = temp_dir / 'styles'
        (styles_dir / 'category1').mkdir(parents=True)
        
        # Copy sample images
        for img_file in sample_images.glob('*.jpg'):
            shutil.copy(img_file, styles_dir / 'category1')
        
        downloader = DatasetDownloader(str(temp_dir))
        output_dir = temp_dir / 'processed_styles'
        
        metadata = downloader.prepare_style_images(styles_dir, output_dir)
        
        # Check metadata
        assert 'category1' in metadata
        assert metadata['category1']['num_images'] == 5
        assert (output_dir / 'style_metadata.json').exists()


def test_dataset_metadata_format(temp_dir):
    """Test dataset metadata JSON format."""
    metadata = {
        'source_dir': str(temp_dir),
        'target_size': (256, 256),
        'split_ratios': (0.8, 0.1, 0.1),
        'splits': {'train': 80, 'val': 10, 'test': 10},
        'total_images': 100
    }
    
    metadata_file = temp_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Verify JSON is valid
    with open(metadata_file, 'r') as f:
        loaded = json.load(f)
    
    assert loaded == metadata
    assert loaded['total_images'] == 100


def test_image_extensions(temp_dir):
    """Test handling of different image formats."""
    images_dir = temp_dir / 'mixed_formats'
    images_dir.mkdir(parents=True)
    
    # Create images in different formats
    formats = ['.jpg', '.jpeg', '.png', '.bmp']
    for i, ext in enumerate(formats):
        img = Image.new('RGB', (100, 100), color=(i * 60, i * 60, i * 60))
        img.save(images_dir / f'test_{i}{ext}')
    
    downloader = DatasetDownloader(str(temp_dir))
    output_dir = temp_dir / 'processed'
    
    metadata = downloader.preprocess_images(
        images_dir,
        output_dir,
        target_size=(64, 64)
    )
    
    # Should process all formats
    assert metadata['total_images'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
