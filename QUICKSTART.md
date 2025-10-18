# Quick Start Guide

This guide gets you training your first style transfer model in under 5 minutes.

## Prerequisites

- Python 3.9+ installed
- 8GB+ RAM recommended
- GPU optional (CPU training works but is slower)

## Setup (2 minutes)

```bash
# Clone the repository
git clone <your-repo-url>
cd style-transfer-models

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (choose one)
pip install -r requirements.txt  # Full install (PyTorch + TensorFlow)
# OR
pip install torch torchvision coremltools  # PyTorch only
# OR
pip install tensorflow coremltools  # TensorFlow only
```

## Option 1: Test with Sample Data (30 seconds)

```bash
# Quick test with sample images (no dataset download needed)
# First, add some sample images to datasets/sample/content/ and datasets/sample/styles/
python training/pytorch/train.py \
  --content-dir datasets/sample/content \
  --style-image datasets/sample/styles/starry_night.jpg \
  --style-name starry-night \
  --epochs 2 \
  --batch-size 1 \
  --checkpoint-dir checkpoints/test
```

## Option 2: Download COCO Dataset (1 minute download time)

```bash
# Download and prepare COCO dataset
cd style-transfer-models
python datasets/scripts/download_prepare.py \
  --dataset coco \
  --output-dir datasets/coco \
  --target-size 256 256
```

## Train Your First Model (2-10 minutes depending on hardware)

### PyTorch

```bash
python training/pytorch/train.py \
  --content-dir datasets/coco/train \
  --style-image datasets/sample/styles/starry_night.jpg \
  --style-name starry-night \
  --epochs 10 \
  --batch-size 4 \
  --checkpoint-dir checkpoints/starry_night \
  --lr 1e-3
```

### TensorFlow

```bash
python training/tensorflow/train.py \
  --content-dir datasets/coco/train \
  --style-image datasets/sample/styles/starry_night.jpg \
  --style-name starry-night \
  --epochs 10 \
  --batch-size 4 \
  --checkpoint-dir checkpoints/starry_night \
  --lr 1e-3
```

## Convert to Core ML (30 seconds)

```bash
python conversion/convert_coreml.py \
  --model-path checkpoints/starry_night/best_model.pth \
  --framework pytorch \
  --output models/starry_night.mlmodel \
  --name "Starry Night" \
  --description "Van Gogh starry night style" \
  --author "Your Name"
```

## Generate Manifest (10 seconds)

```bash
python manifest/generate_manifest.py generate \
  --models-dir models/ \
  --output manifest/styles.json
```

## Validate Everything Works

```bash
# Validate the Core ML model
python conversion/convert_coreml.py \
  --model-path checkpoints/starry_night/best_model.pth \
  --framework pytorch \
  --output models/starry_night.mlmodel \
  --validate

# Validate the manifest
python manifest/generate_manifest.py validate \
  --manifest manifest/styles.json
```

## What You Just Created

- ✅ Trained style transfer model
- ✅ Core ML model file (`models/starry_night.mlmodel`)
- ✅ Updated manifest (`manifest/styles.json`)
- ✅ Ready to integrate into your iOS/visionOS app!

## Next Steps

1. **Try different styles**: Add your own style images to `datasets/sample/styles/`
2. **Optimize for device**: See `conversion/README.md` for quantization options
3. **Batch training**: Train multiple styles using the scripts in `training/README.md`
4. **Performance tuning**: Adjust hyperparameters for quality vs. speed tradeoffs
5. **iOS integration**: Copy `models/*.mlmodel` and `manifest/styles.json` to your Xcode project

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--image-size` (try 128 or 256)

### Training Too Slow
- Reduce `--epochs` (2-5 for quick tests)
- Use GPU if available
- Reduce dataset size with `--max-images`

### Model Quality Issues
- Train for more epochs (20-50 for production)
- Use larger dataset (10k+ images)
- Adjust loss weights (`--content-weight`, `--style-weight`)

### Core ML Conversion Fails
- Ensure model is in eval mode
- Check PyTorch/TensorFlow version compatibility
- Try without quantization first (`--quantize none`)

## Getting Help

- 📖 Full documentation: See `README.md`
- 🎓 Training guide: See `training/README.md`
- 🔄 Conversion guide: See `conversion/README.md`
- 🐛 Issues: Check the GitHub issues page
- 💬 Questions: Open a discussion on GitHub

---

**Time to first model**: ~5 minutes
**Time to production-ready model**: ~1 hour
**Ready to ship!** 🚀
