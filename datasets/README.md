# Datasets for Style Transfer Training

This directory contains scripts and datasets for training style transfer models.

## Directory Structure

```
datasets/
├── scripts/
│   └── download_prepare.py    # Dataset download and preprocessing script
├── sample/
│   ├── styles/                # Sample style images
│   └── content/               # Sample content images
├── coco/                      # COCO dataset (content images)
├── wikiart_styles/            # WikiArt style images
└── processed/                 # Preprocessed datasets
```

## Quick Start

### 1. Download COCO Dataset (Content Images)

```bash
cd datasets/scripts
python download_prepare.py --dataset coco --data-dir ../
```

This downloads the COCO 2017 training set (~18GB) for content images.

### 2. Preprocess COCO Images

```bash
python download_prepare.py \
    --dataset custom \
    --source-dir ../coco/train2017 \
    --output-dir ../processed/coco_256 \
    --target-size 256 256 \
    --split-ratios 0.8 0.1 0.1
```

This resizes images to 256x256 and splits into train/val/test sets.

### 3. Prepare Style Images

Organize your style images by category:

```
wikiart_styles/
├── sci-fi/
│   ├── style_001.jpg
│   ├── style_002.jpg
│   └── ...
├── fantasy/
│   ├── style_001.jpg
│   └── ...
└── modern/
    └── ...
```

Then prepare them:

```bash
python download_prepare.py \
    --dataset custom \
    --source-dir ../wikiart_styles \
    --output-dir ../processed/styles \
    --prepare-styles
```

## Dataset Options

### COCO Dataset
- **Size**: ~18GB (train2017)
- **Images**: 118,287 training images
- **Use**: Content images for training
- **Download**: Automatic via script

### WikiArt Dataset
- **Use**: Style images (various artistic styles)
- **Download**: Manual (requires WikiArt API or web scraping)
- **Categories**: Impressionism, Abstract, Sci-fi, Fantasy, etc.

### Custom Datasets
- Place images in organized directories
- Use `--dataset custom` option
- Support for JPG, PNG, BMP, TIFF formats

## Preprocessing Options

### Image Resizing
```bash
--target-size 256 256  # Width and height
```

Common sizes:
- `256 256` - Fast training, mobile-friendly
- `512 512` - Better quality, slower
- `1024 1024` - High quality, resource-intensive

### Data Splitting
```bash
--split-ratios 0.8 0.1 0.1  # train/val/test
```

Default split:
- Training: 80%
- Validation: 10%
- Testing: 10%

## Sample Dataset

The `sample/` directory contains a small dataset for quick testing:

```bash
# Use sample dataset
python download_prepare.py \
    --dataset custom \
    --source-dir ../sample/content \
    --output-dir ../processed/sample \
    --target-size 256 256
```

## Dataset Metadata

Each processed dataset includes a `metadata.json` file:

```json
{
  "source_dir": "/path/to/source",
  "target_size": [256, 256],
  "split_ratios": [0.8, 0.1, 0.1],
  "splits": {
    "train": 94629,
    "val": 11828,
    "test": 11830
  },
  "total_images": 118287
}
```

Style datasets include `style_metadata.json`:

```json
{
  "sci-fi": {
    "num_images": 50,
    "images": ["sci-fi/sci-fi_000.jpg", ...]
  },
  "fantasy": {
    "num_images": 45,
    "images": ["fantasy/fantasy_000.jpg", ...]
  }
}
```

## Adding New Datasets

1. **Organize source images** in a directory structure
2. **Run preprocessing** with `--dataset custom`
3. **Verify output** - check metadata.json and sample images
4. **Update training configs** to point to new dataset

## Storage Requirements

| Dataset | Raw Size | Processed Size (256x256) |
|---------|----------|-------------------------|
| COCO train2017 | ~18GB | ~4GB |
| WikiArt (1000 styles) | ~2GB | ~500MB |
| Sample | ~50MB | ~10MB |

## Best Practices

1. **Use high-quality style images** - Clean, clear artistic examples
2. **Diverse content images** - COCO provides good variety
3. **Consistent preprocessing** - Same size/format for training
4. **Validate splits** - Ensure no data leakage between train/val/test
5. **Version control** - Track dataset versions and preprocessing configs

## Troubleshooting

### Out of Memory
- Reduce `--target-size`
- Process in smaller batches
- Use streaming/lazy loading in training scripts

### Slow Download
- Check internet connection
- Use download mirrors
- Download manually and extract

### Missing Dependencies
```bash
pip install Pillow tqdm numpy
```

## Attribution

- **COCO Dataset**: Lin et al. "Microsoft COCO: Common Objects in Context" (2014)
- **WikiArt**: Licensed artwork from wikiart.org
- Always check licenses for dataset usage and distribution

## Next Steps

After preparing datasets:
1. Review `training/README.md` for training instructions
2. Configure training scripts with dataset paths
3. Start training with PyTorch or TensorFlow

## Support

For issues or questions:
- Check dataset paths and permissions
- Verify image formats and sizes
- Review `metadata.json` for processing details
