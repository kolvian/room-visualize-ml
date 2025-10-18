# Style Transfer Model Training

Comprehensive guide for training fast neural style transfer models using PyTorch or TensorFlow.

## Table of Contents

- [Overview](#overview)
- [PyTorch Training](#pytorch-training)
- [TensorFlow Training](#tensorflow-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)

## Overview

This directory contains two complete training implementations:

- **PyTorch** (`pytorch/`): Flexible, research-friendly, excellent debugging
- **TensorFlow** (`tensorflow/`): Production-ready, mobile-optimized

Both implementations support:
- Fast neural style transfer (Johnson et al., 2016)
- Perceptual loss with VGG features
- Content, style, and total variation losses
- Configurable hyperparameters
- Checkpointing and resuming
- Validation and early stopping
- Sample output visualization

## PyTorch Training

### Basic Usage

```bash
cd pytorch

python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --style-image ../../datasets/processed/styles/sci-fi/sci-fi_001.jpg \
    --style-name sci-fi \
    --epochs 10 \
    --batch-size 8
```

### Full Options

```bash
python train.py \
    # Dataset
    --content-dir PATH          # Content images directory (required)
    --style-image PATH          # Style image path (required)
    --style-name NAME           # Style identifier (required)
    --val-dir PATH              # Validation images directory
    
    # Model
    --num-residual-blocks 5     # Number of residual blocks (default: 5)
    
    # Training
    --epochs 10                 # Number of epochs (default: 2)
    --batch-size 8              # Batch size (default: 4)
    --image-size 256            # Training image size (default: 256)
    --lr 1e-3                   # Learning rate (default: 1e-3)
    --content-weight 1.0        # Content loss weight (default: 1.0)
    --style-weight 1e5          # Style loss weight (default: 1e5)
    --tv-weight 1e-6            # Total variation loss weight (default: 1e-6)
    --grad-clip 0               # Gradient clipping (0 = disabled)
    
    # Optimization
    --optimizer adam            # Optimizer: adam or sgd (default: adam)
    --lr-scheduler              # Use learning rate scheduler
    --early-stopping            # Enable early stopping
    --patience 5                # Early stopping patience (default: 5)
    
    # System
    --device auto               # Device: auto, cuda, mps, cpu (default: auto)
    --num-workers 4             # Data loading workers (default: 4)
    --seed 42                   # Random seed (default: 42)
    
    # Checkpoints
    --output-dir ./output       # Output directory (default: ./output)
    --checkpoint-dir ../../models/checkpoints
    --resume PATH               # Resume from checkpoint
    --save-interval 1           # Save checkpoint every N epochs
    
    # Logging
    --log-interval 100          # Log every N batches
    --sample-interval 500       # Save samples every N batches
```

### Example Configurations

**Fast Training (Quick Experimentation)**
```bash
python train.py \
    --content-dir ../../datasets/processed/sample/train \
    --style-image ../../datasets/sample/styles/sci-fi.jpg \
    --style-name sci-fi-quick \
    --epochs 2 \
    --batch-size 4 \
    --image-size 128 \
    --save-interval 1
```

**High Quality (Production)**
```bash
python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --val-dir ../../datasets/processed/coco_256/val \
    --style-image ../../datasets/processed/styles/fantasy/fantasy_001.jpg \
    --style-name fantasy-hq \
    --epochs 20 \
    --batch-size 8 \
    --image-size 512 \
    --lr 1e-3 \
    --content-weight 1.0 \
    --style-weight 5e4 \
    --tv-weight 1e-6 \
    --lr-scheduler \
    --early-stopping \
    --num-workers 8 \
    --save-interval 2
```

**Mobile Optimized**
```bash
python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --val-dir ../../datasets/processed/coco_256/val \
    --style-image ../../datasets/processed/styles/modern/modern_001.jpg \
    --style-name modern-mobile \
    --epochs 15 \
    --batch-size 16 \
    --image-size 256 \
    --lr 1e-3 \
    --content-weight 1.0 \
    --style-weight 1e5 \
    --tv-weight 1e-6 \
    --num-residual-blocks 3 \
    --lr-scheduler \
    --save-interval 3
```

### Output Structure

After training, the output directory contains:

```
output/sci-fi/
├── training.log              # Training logs
├── config.json              # Training configuration
├── training_history.csv     # Loss history
└── samples/                 # Sample outputs
    ├── epoch_0_step_0_content.jpg
    ├── epoch_0_step_0_styled.jpg
    └── ...
```

Checkpoints are saved to:

```
models/checkpoints/
├── sci-fi_epoch_1.pth       # Epoch checkpoints
├── sci-fi_epoch_2.pth
├── sci-fi_best.pth          # Best model
└── sci-fi_latest.pth        # Latest (for resuming)
```

### Resume Training

```bash
python train.py \
    --resume ../../models/checkpoints/sci-fi_latest.pth \
    --content-dir ../../datasets/processed/coco_256/train \
    --style-image ../../datasets/processed/styles/sci-fi/sci-fi_001.jpg \
    --style-name sci-fi \
    --epochs 20
```

## TensorFlow Training

### Basic Usage

```bash
cd tensorflow

python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --style-image ../../datasets/processed/styles/fantasy/fantasy_001.jpg \
    --style-name fantasy \
    --epochs 10 \
    --batch-size 8
```

### Full Options

Similar to PyTorch, with TensorFlow-specific options:

```bash
python train.py \
    # ... (same dataset, model, training options as PyTorch)
    
    # System (TensorFlow-specific)
    --gpu-memory-limit 4096     # GPU memory limit in MB
    
    # ... (same checkpoint and logging options)
```

### TensorFlow Callbacks

TensorFlow training includes:
- **TensorBoard**: Real-time visualization (`tensorboard --logdir output/fantasy/logs`)
- **CSV Logger**: Loss history in CSV format
- **Model Checkpointing**: Automatic best model saving
- **Early Stopping**: Based on validation loss
- **Learning Rate Scheduler**: Adaptive learning rate

### View Training Progress

```bash
# Start TensorBoard
tensorboard --logdir output/fantasy/logs

# Open browser to http://localhost:6006
```

## Hyperparameter Tuning

### Content Weight (`--content-weight`)

Controls content preservation:

- **Low (0.1 - 0.5)**: Strong stylization, less content detail
- **Medium (1.0 - 2.0)**: Balanced (recommended)
- **High (5.0 - 10.0)**: Strong content, subtle style

### Style Weight (`--style-weight`)

Controls style strength:

- **Low (1e3 - 1e4)**: Subtle style transfer
- **Medium (1e5)**: Strong stylization (recommended)
- **High (1e6 - 1e7)**: Very strong style, may lose content

### Total Variation Weight (`--tv-weight`)

Controls output smoothness:

- **Low (1e-7)**: More detail, may be noisy
- **Medium (1e-6)**: Balanced (recommended)
- **High (1e-5)**: Smoother, may lose detail

### Learning Rate (`--lr`)

- **Start**: 1e-3 (recommended)
- **Use scheduler**: Reduce by 0.5 when validation loss plateaus
- **Too high**: Training instability, poor convergence
- **Too low**: Slow training, may not converge

### Batch Size (`--batch-size`)

- **Small (2-4)**: Less memory, more updates, slower
- **Medium (8-16)**: Good balance (recommended)
- **Large (32+)**: Faster training, requires more memory

## Advanced Techniques

### Multi-Style Training

Train multiple styles in parallel:

```bash
# Train multiple styles
for style in sci-fi fantasy modern; do
    python pytorch/train.py \
        --content-dir ../datasets/processed/coco_256/train \
        --style-image ../datasets/processed/styles/${style}/${style}_001.jpg \
        --style-name ${style} \
        --epochs 10 \
        --batch-size 8 &
done
wait
```

### Custom Loss Weights

For specific artistic effects:

```bash
# Strong content preservation
python train.py ... --content-weight 5.0 --style-weight 5e4

# Painterly effect
python train.py ... --content-weight 0.5 --style-weight 1e6 --tv-weight 1e-5

# Sharp, detailed
python train.py ... --content-weight 2.0 --style-weight 8e4 --tv-weight 1e-7
```

### Gradient Clipping

Prevent training instability:

```bash
python train.py ... --grad-clip 5.0
```

### Different Image Sizes

```bash
# Small (faster, lower quality)
python train.py ... --image-size 128

# Standard (balanced)
python train.py ... --image-size 256

# Large (slower, higher quality)
python train.py ... --image-size 512
```

## Training Tips

1. **Start Small**: Test with 2 epochs on sample data
2. **Monitor Losses**: Content and style should both decrease
3. **Check Samples**: Visual quality more important than loss values
4. **Use Validation**: Prevent overfitting with validation set
5. **Save Frequently**: Use `--save-interval 1` for important trainings
6. **GPU Utilization**: Maximize batch size without OOM
7. **Learning Rate**: Use scheduler for better convergence
8. **Style Image Quality**: High-quality style images produce better results

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
```bash
# Reduce batch size
--batch-size 2

# Reduce image size
--image-size 128

# Reduce workers
--num-workers 2

# For PyTorch: Use CPU
--device cpu
```

### Poor Stylization

**Solutions:**
```bash
# Increase style weight
--style-weight 5e5

# Train longer
--epochs 20

# Use higher quality style image
```

### Loss of Content Detail

**Solutions:**
```bash
# Increase content weight
--content-weight 2.0

# Decrease style weight
--style-weight 5e4

# Reduce TV weight
--tv-weight 1e-7
```

### Training Too Slow

**Solutions:**
```bash
# Increase batch size
--batch-size 16

# More data workers
--num-workers 8

# Smaller image size
--image-size 128

# GPU/MPS acceleration
--device cuda  # or mps on Mac
```

### Unstable Training

**Solutions:**
```bash
# Use gradient clipping
--grad-clip 5.0

# Reduce learning rate
--lr 5e-4

# Use scheduler
--lr-scheduler
```

## Performance Benchmarks

Training time (10 epochs, COCO dataset, 256x256):

| Hardware | Batch Size | Time per Epoch | Total Time |
|----------|------------|----------------|------------|
| RTX 3090 | 16 | ~8 min | ~80 min |
| RTX 3070 | 8 | ~15 min | ~150 min |
| M1 Max (MPS) | 8 | ~20 min | ~200 min |
| M1 (MPS) | 4 | ~30 min | ~300 min |
| CPU (16-core) | 4 | ~90 min | ~900 min |

## Next Steps

After training:

1. **Validate**: Check training logs and sample outputs
2. **Convert**: Use `conversion/convert_coreml.py` to create .mlmodel
3. **Benchmark**: Test performance with `conversion/validate_coreml.ipynb`
4. **Deploy**: Add to manifest and integrate with your app

## References

- Johnson et al. "Perceptual Losses for Real-Time Style Transfer" (ECCV 2016)
- Gatys et al. "A Neural Algorithm of Artistic Style" (2015)
- [PyTorch Fast Neural Style](https://github.com/pytorch/examples/tree/main/fast_neural_style)
