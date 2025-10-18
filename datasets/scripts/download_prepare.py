#!/usr/bin/env python3
"""
Dataset Download and Preparation Script
Downloads and preprocesses datasets for style transfer training.
Supports COCO, ADE20K, and custom style image collections.
"""

import argparse
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import List, Tuple
import json
from PIL import Image
from tqdm import tqdm
import numpy as np


class DatasetDownloader:
    """Handles downloading and preparing datasets for style transfer."""
    
    def __init__(self, data_dir: str = "./datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, dest: Path):
        """Download a file with progress bar."""
        if dest.exists():
            print(f"File already exists: {dest}")
            return
            
        print(f"Downloading {url}...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                print(f"\rProgress: {percent}%", end='')
        
        urllib.request.urlretrieve(url, dest, reporthook)
        print("\nDownload complete!")
        
    def extract_archive(self, archive_path: Path, extract_to: Path):
        """Extract zip or tar archives."""
        print(f"Extracting {archive_path}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print("Extraction complete!")
    
    def download_coco(self, year: str = "2017", subset: str = "train"):
        """Download COCO dataset for content images."""
        base_url = f"http://images.cocodataset.org/zips/{subset}{year}.zip"
        dest_file = self.data_dir / f"coco_{subset}{year}.zip"
        extract_dir = self.data_dir / "coco" / f"{subset}{year}"
        
        self.download_file(base_url, dest_file)
        self.extract_archive(dest_file, extract_dir)
        
        return extract_dir
    
    def download_wikiart_styles(self):
        """Download WikiArt style images (sample collection)."""
        # Note: For production, use proper WikiArt API or dataset
        # This is a placeholder for demonstration
        styles_dir = self.data_dir / "wikiart_styles"
        styles_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample style categories
        style_categories = ["sci-fi", "fantasy", "modern", "abstract", "impressionism"]
        
        print("WikiArt requires manual download or API access.")
        print(f"Please download style images to: {styles_dir}")
        print(f"Organize by subdirectories: {', '.join(style_categories)}")
        
        return styles_dir
    
    def preprocess_images(
        self,
        source_dir: Path,
        output_dir: Path,
        target_size: Tuple[int, int] = (256, 256),
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ):
        """
        Preprocess images: resize, normalize, and split into train/val/test.
        
        Args:
            source_dir: Directory containing source images
            output_dir: Directory for processed images
            target_size: Target image size (width, height)
            split_ratios: (train, val, test) split ratios
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in source_dir.rglob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images in {source_dir}")
        
        # Shuffle and split
        np.random.shuffle(image_files)
        train_end = int(len(image_files) * split_ratios[0])
        val_end = train_end + int(len(image_files) * split_ratios[1])
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Process and save images
        for split_name, files in splits.items():
            print(f"\nProcessing {split_name} split ({len(files)} images)...")
            for i, img_path in enumerate(tqdm(files)):
                try:
                    # Open and resize image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size, Image.LANCZOS)
                    
                    # Save processed image
                    output_path = output_dir / split_name / f"{split_name}_{i:06d}.jpg"
                    img.save(output_path, quality=95)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Save dataset metadata
        metadata = {
            'source_dir': str(source_dir),
            'target_size': target_size,
            'split_ratios': split_ratios,
            'splits': {k: len(v) for k, v in splits.items()},
            'total_images': len(image_files)
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset preparation complete!")
        print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return metadata
    
    def prepare_style_images(self, style_dir: Path, output_dir: Path):
        """Prepare style images for training."""
        style_dir = Path(style_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        style_metadata = {}
        
        # Process each style category
        for category_dir in style_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            category_output = output_dir / category_name
            category_output.mkdir(parents=True, exist_ok=True)
            
            print(f"\nProcessing style category: {category_name}")
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            style_images = [
                f for f in category_dir.rglob('*')
                if f.suffix.lower() in image_extensions
            ]
            
            processed_images = []
            for i, img_path in enumerate(tqdm(style_images)):
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Keep original size for style images or resize to standard
                    img = img.resize((512, 512), Image.LANCZOS)
                    
                    output_path = category_output / f"{category_name}_{i:03d}.jpg"
                    img.save(output_path, quality=95)
                    processed_images.append(str(output_path.relative_to(output_dir)))
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            style_metadata[category_name] = {
                'num_images': len(processed_images),
                'images': processed_images
            }
        
        # Save style metadata
        with open(output_dir / 'style_metadata.json', 'w') as f:
            json.dump(style_metadata, f, indent=2)
        
        print(f"\nStyle preparation complete!")
        for category, meta in style_metadata.items():
            print(f"{category}: {meta['num_images']} images")
        
        return style_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for style transfer training"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'wikiart', 'custom'],
        default='coco',
        help='Dataset to download/prepare'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./datasets',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        help='Source directory for custom dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Target image size (width height)'
    )
    parser.add_argument(
        '--split-ratios',
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help='Train/val/test split ratios'
    )
    parser.add_argument(
        '--prepare-styles',
        action='store_true',
        help='Prepare style images instead of content images'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.dataset == 'coco':
        print("Downloading COCO dataset...")
        extract_dir = downloader.download_coco(year="2017", subset="train")
        
        if args.output_dir:
            print("\nPreprocessing images...")
            downloader.preprocess_images(
                extract_dir,
                Path(args.output_dir),
                target_size=tuple(args.target_size),
                split_ratios=tuple(args.split_ratios)
            )
    
    elif args.dataset == 'wikiart':
        styles_dir = downloader.download_wikiart_styles()
        
    elif args.dataset == 'custom':
        if not args.source_dir or not args.output_dir:
            parser.error("--source-dir and --output-dir required for custom dataset")
        
        if args.prepare_styles:
            downloader.prepare_style_images(
                Path(args.source_dir),
                Path(args.output_dir)
            )
        else:
            downloader.preprocess_images(
                Path(args.source_dir),
                Path(args.output_dir),
                target_size=tuple(args.target_size),
                split_ratios=tuple(args.split_ratios)
            )


if __name__ == '__main__':
    main()
