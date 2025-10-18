# Core ML Conversion and Validation

Convert trained style transfer models to Apple's Core ML format for deployment on iOS, iPadOS, and visionOS devices.

## Table of Contents

- [Overview](#overview)
- [Conversion Process](#conversion-process)
- [Validation](#validation)
- [Optimization](#optimization)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

This directory provides tools for:

- **Conversion**: Transform PyTorch/TensorFlow models to Core ML
- **Validation**: Test model correctness and performance
- **Optimization**: Quantization, compute unit selection
- **Benchmarking**: Measure FPS, latency, memory usage

### Files

- `convert_coreml.py`: Main conversion script
- `validate_coreml.ipynb`: Interactive validation notebook
- `README.md`: This file

## Conversion Process

### Prerequisites

```bash
# Install Core ML Tools (macOS required)
pip install coremltools>=7.0

# Verify installation
python -c "import coremltools; print(coremltools.__version__)"
```

### Basic Conversion

#### From PyTorch

```bash
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi_best.pth \
    --output-path ../models/exported/sci-fi.mlmodel \
    --style-name sci-fi
```

#### From TensorFlow

```bash
python convert_coreml.py \
    --framework tensorflow \
    --model-path ../models/checkpoints/fantasy_best.h5 \
    --output-path ../models/exported/fantasy.mlmodel \
    --style-name fantasy
```

### Full Options

```bash
python convert_coreml.py \
    # Required
    --framework {pytorch,tensorflow}  # Source framework
    --model-path PATH                 # Path to trained model (.pth or .h5)
    --output-path PATH                # Output .mlmodel path
    --style-name NAME                 # Style identifier
    
    # Model Configuration
    --input-size H W                  # Input dimensions (default: 256 256)
    --compute-units {ALL,CPU_AND_GPU,CPU_ONLY,CPU_AND_NE}
                                      # Compute units (default: ALL)
    --quantize                        # Apply float16 quantization
    
    # Metadata
    --author NAME                     # Model author
    --version VERSION                 # Model version
    
    # Validation
    --test-image PATH                 # Test image for validation
```

### Compute Units Explained

| Option | Hardware Used | Performance | Battery | Use Case |
|--------|--------------|-------------|---------|----------|
| `ALL` | CPU + GPU + Neural Engine | Fastest | Moderate | Recommended for most apps |
| `CPU_AND_NE` | CPU + Neural Engine | Very Fast | Best | Modern devices (A14+) |
| `CPU_AND_GPU` | CPU + GPU | Fast | Moderate | Older devices |
| `CPU_ONLY` | CPU only | Slowest | Best | Debugging, fallback |

**Recommendations:**
- **ALL**: Best default for general use
- **CPU_AND_NE**: Best for iPhone 12+ and newer iPads
- **CPU_AND_GPU**: Compatibility with older devices
- **CPU_ONLY**: Testing and debugging only

### Quantization

Float16 quantization reduces model size by ~50% with minimal quality loss.

**Without Quantization:**
```bash
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi_best.pth \
    --output-path ../models/exported/sci-fi_fp32.mlmodel \
    --style-name sci-fi
```
- Size: ~13MB
- Quality: Highest
- Speed: Standard

**With Quantization:**
```bash
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi_best.pth \
    --output-path ../models/exported/sci-fi.mlmodel \
    --style-name sci-fi \
    --quantize
```
- Size: ~7MB
- Quality: Nearly identical
- Speed: Often faster (Neural Engine optimized)

**Recommendation**: Always use `--quantize` for production.

## Validation

### Command-Line Validation

Quick validation with test image:

```bash
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi_best.pth \
    --output-path ../models/exported/sci-fi.mlmodel \
    --style-name sci-fi \
    --quantize \
    --test-image ../datasets/sample/content/test_001.jpg
```

Output:
```
✓ Model loaded successfully
  Input: input_image, shape: [1, 3, 256, 256]
  Output: stylized_image, shape: [1, 3, 256, 256]

Testing inference with: ../datasets/sample/content/test_001.jpg
✓ Inference successful
  Output shape: (1, 3, 256, 256)
  Output range: [0.002, 0.998]

✓ Conversion info saved to: ../models/exported/sci-fi.json
```

### Interactive Validation (Recommended)

Open the Jupyter notebook for comprehensive validation:

```bash
jupyter notebook validate_coreml.ipynb
```

The notebook provides:

1. **Model Inspection**
   - Metadata and version info
   - Input/output specifications
   - Framework and architecture details

2. **Inference Testing**
   - Load and preprocess test images
   - Run model inference
   - Visualize original vs. styled images

3. **Performance Benchmarking**
   - Warmup runs
   - Multiple iterations for statistical analysis
   - FPS, latency, and timing distributions
   - Performance plots

4. **Quality Assessment**
   - Mean Squared Error (MSE)
   - Peak Signal-to-Noise Ratio (PSNR)
   - Color distribution analysis
   - Visual quality metrics

5. **Validation Report**
   - JSON report with all metrics
   - Pass/fail verdicts:
     - Real-time capable (≥15 FPS)
     - Mobile optimized (≤100ms latency)
     - Quality acceptable (PSNR ≥20dB)
   - Styled output images

### Notebook Configuration

Edit the configuration cell:

```python
# Model to validate
MODEL_PATH = "../models/exported/sci-fi.mlmodel"
TEST_IMAGE_PATH = "../datasets/sample/content/test_001.jpg"

# Performance testing
NUM_WARMUP_RUNS = 5
NUM_BENCHMARK_RUNS = 20

# Output directory
OUTPUT_DIR = Path("./validation_results")
```

### Sample Validation Report

```json
{
  "model_path": "../models/exported/sci-fi.mlmodel",
  "timestamp": "2025-10-18 15:30:00",
  "model_info": {
    "framework": "pytorch",
    "input_shape": [1, 3, 256, 256],
    "quantized": true
  },
  "performance": {
    "mean_ms": 45.2,
    "std_ms": 3.1,
    "fps": 22.1,
    "min_ms": 42.0,
    "max_ms": 51.3
  },
  "quality": {
    "mse": 856.4,
    "psnr": 28.7
  },
  "verdict": {
    "realtime_capable": true,
    "mobile_optimized": true,
    "quality_acceptable": true
  }
}
```

## Optimization

### Optimization Strategies

1. **Quantization** (Recommended)
   - Use `--quantize` flag
   - ~50% size reduction
   - Minimal quality impact
   - Often faster on Neural Engine

2. **Input Size**
   - Smaller = faster, lower quality
   - 256x256: Good balance (recommended)
   - 512x512: Higher quality, slower
   - 128x128: Very fast, lower quality

3. **Compute Units**
   - `ALL` or `CPU_AND_NE` for best performance
   - Neural Engine optimized for quantized models

4. **Model Architecture**
   - Reduce `--num-residual-blocks` during training
   - 3 blocks: Faster, lower quality
   - 5 blocks: Standard (recommended)
   - 7+ blocks: Slower, diminishing returns

### Example: Maximum Performance

```bash
# Train with fewer residual blocks
cd ../training/pytorch
python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --style-image ../../datasets/processed/styles/sci-fi/sci-fi_001.jpg \
    --style-name sci-fi-fast \
    --epochs 10 \
    --batch-size 8 \
    --num-residual-blocks 3

# Convert with quantization and Neural Engine
cd ../../conversion
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi-fast_best.pth \
    --output-path ../models/exported/sci-fi-fast.mlmodel \
    --style-name sci-fi-fast \
    --input-size 256 256 \
    --compute-units CPU_AND_NE \
    --quantize
```

Expected performance: 30-35ms latency, 28-30 FPS

### Example: Maximum Quality

```bash
# Train with more residual blocks and larger images
cd ../training/pytorch
python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --style-image ../../datasets/processed/styles/fantasy/fantasy_001.jpg \
    --style-name fantasy-hq \
    --epochs 20 \
    --batch-size 4 \
    --image-size 512 \
    --num-residual-blocks 7

# Convert without quantization
cd ../../conversion
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/fantasy-hq_best.pth \
    --output-path ../models/exported/fantasy-hq.mlmodel \
    --style-name fantasy-hq \
    --input-size 512 512 \
    --compute-units ALL
```

Expected performance: 80-100ms latency, ~10 FPS

## Deployment

### iOS Integration

1. **Add Model to Xcode Project**
   - Drag `.mlmodel` file into Xcode project
   - Xcode auto-generates Swift interface

2. **Load Model**
   ```swift
   import CoreML
   
   guard let model = try? SciFi(configuration: MLModelConfiguration()) else {
       fatalError("Failed to load model")
   }
   ```

3. **Prepare Input**
   ```swift
   func stylizeImage(_ image: UIImage) -> UIImage? {
       // Resize to 256x256
       let size = CGSize(width: 256, height: 256)
       let resized = image.resize(to: size)
       
       // Convert to MLMultiArray or CVPixelBuffer
       guard let buffer = resized.pixelBuffer(width: 256, height: 256) else {
           return nil
       }
       
       // Run inference
       guard let output = try? model.prediction(input_image: buffer) else {
           return nil
       }
       
       // Convert output to UIImage
       return output.stylized_image.image
   }
   ```

4. **Optimize Performance**
   ```swift
   // Use Metal for pixel buffer conversion
   // Process on background queue
   // Cache model instance
   ```

### visionOS Integration

```swift
import CoreML
import RealityKit

class StyleTransferEngine {
    private let model: MLModel
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try SciFi(configuration: config)
    }
    
    func applyStyle(to texture: TextureResource) async throws -> TextureResource {
        // Convert texture to CVPixelBuffer
        let buffer = texture.toCVPixelBuffer()
        
        // Run model
        let output = try model.prediction(input_image: buffer)
        
        // Convert back to TextureResource
        return try TextureResource(from: output.stylized_image)
    }
}
```

### Manifest Integration

After conversion, update the manifest:

```bash
cd ../manifest

# Generate/update manifest
python generate_manifest.py generate \
    --models-dir ../models/exported \
    --output styles.json

# Validate
python generate_manifest.py validate \
    --manifest styles.json \
    --schema styles_schema.json
```

Include `styles.json` in your app bundle for dynamic style loading.

## Troubleshooting

### Conversion Fails

**Error: "Import torch could not be resolved"**
```bash
# Install PyTorch
pip install torch torchvision

# Or for Apple Silicon
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Error: "Model checkpoint loading failed"**
```bash
# Verify checkpoint structure
python -c "import torch; print(torch.load('path/to/model.pth').keys())"

# Checkpoint should contain 'model_state_dict' or be a direct state dict
```

**Error: "coremltools conversion failed"**
```bash
# Update coremltools
pip install --upgrade coremltools

# Try different compute units
python convert_coreml.py ... --compute-units CPU_ONLY
```

### Validation Issues

**Error: "Model output range incorrect"**
- Expected: [0, 1]
- Check model's final activation (should be sigmoid or tanh scaled)
- Verify preprocessing in training

**Error: "Inference too slow"**
```bash
# Solutions:
1. Use --quantize
2. Try --compute-units CPU_AND_NE
3. Reduce --input-size
4. Verify device has Neural Engine (A14+)
```

### Quality Issues

**Stylized output looks wrong**
- Verify test image preprocessing matches training
- Check input/output shapes match
- Test with original framework first (PyTorch/TF) before conversion

**Colors are off**
- Check normalization (should be [0, 1], not [0, 255])
- Verify RGB vs BGR channel order
- Test with simple solid color input

## Performance Tips

1. **Use Neural Engine**
   - Quantize models
   - Use `CPU_AND_NE` compute units
   - Target A14+ devices

2. **Batch Processing**
   - Process multiple frames together if possible
   - Amortizes model loading overhead

3. **Caching**
   - Cache model instance
   - Reuse MLMultiArray buffers
   - Pre-allocate output buffers

4. **Threading**
   - Run inference on background queue
   - Avoid blocking UI thread

5. **Monitoring**
   - Use Instruments to profile
   - Monitor memory usage
   - Check thermal state

## Next Steps

After successful conversion and validation:

1. **Generate Manifest**: Add model to `styles.json`
2. **Create Thumbnail**: Generate preview image for UI
3. **Integrate with App**: Add to Xcode project
4. **Test on Device**: Validate on target hardware
5. **Optimize**: Fine-tune based on device testing

## References

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Core ML Tools](https://github.com/apple/coremltools)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Vision Framework](https://developer.apple.com/documentation/vision)
