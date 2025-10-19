#!/usr/bin/env python3
"""
Create a small training subset from COCO dataset for fast testing.
"""

import shutil
from pathlib import Path
import random

def create_small_dataset(source_dir, output_dir, num_images=1000, seed=42):
    """Copy a random subset of images for quick training."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_images = []
    for ext in image_extensions:
        all_images.extend(source_dir.glob(f'*{ext}'))
        all_images.extend(source_dir.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(all_images)} total images in {source_dir}")
    
    # Random sample
    random.seed(seed)
    selected = random.sample(all_images, min(num_images, len(all_images)))
    
    print(f"Copying {len(selected)} images to {output_dir}...")
    
    for i, img_path in enumerate(selected, 1):
        shutil.copy2(img_path, output_dir / img_path.name)
        if i % 100 == 0:
            print(f"  Copied {i}/{len(selected)} images")
    
    print(f"✓ Done! Created subset with {len(selected)} images")
    return len(selected)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Create small dataset subset")
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_small_dataset(args.source, args.output, args.num_images, args.seed)
