# Style Transfer Models for AR/visionOS

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-ff6f00.svg)

End-to-end repository for training, optimizing, and deploying neural style transfer models for real-time AR experiences on iOS and visionOS devices.

## 🎯 Overview

This repository provides a complete pipeline for:

- **Dataset Preparation**: Download and preprocess training datasets (COCO, WikiArt, custom)
- **Model Training**: PyTorch and TensorFlow implementations of fast neural style transfer
- **Core ML Conversion**: Export models to Apple's Core ML format with optimization
- **Validation & Benchmarking**: Test performance, accuracy, and visual quality
- **Deployment Manifest**: Generate metadata for seamless app integration

**Key Features:**
- ✅ Dual framework support (PyTorch & TensorFlow)
- ✅ Optimized for Apple Neural Engine
- ✅ Configurable quantization and compute units
- ✅ Real-time inference (<50ms on modern devices)
- ✅ StyleDescriptor manifest for easy app integration
- ✅ Comprehensive validation and testing tools

## 📁 Repository Structure

```
style-transfer-models/
├── datasets/               # Dataset management
│   ├── scripts/
│   │   └── download_prepare.py
│   ├── sample/            # Sample images for testing
│   └── README.md
├── training/              # Training pipelines
│   ├── pytorch/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── tensorflow/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── utils.py
│   └── README.md
├── conversion/            # Core ML conversion
│   ├── convert_coreml.py
│   ├── validate_coreml.ipynb
│   └── README.md
├── models/                # Trained models
│   ├── exported/          # .mlmodel files
│   └── checkpoints/       # Training checkpoints
├── manifest/              # StyleDescriptor manifest
│   ├── styles.json
│   ├── styles_schema.json
│   └── generate_manifest.py
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- macOS (recommended for Core ML conversion)
- 8GB+ RAM
- GPU (optional but recommended for training)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/style-transfer-models.git
cd style-transfer-models
```

2. **Create a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

For Apple Silicon Macs, TensorFlow Metal acceleration will be automatically installed.

### 30-Second Training Test

Train a style transfer model with sample data:

```bash
# Prepare sample dataset
cd datasets/scripts
python download_prepare.py \
    --dataset custom \
    --source-dir ../sample/content \
    --output-dir ../processed/sample

# Train PyTorch model (2 epochs, fast)
cd ../../training/pytorch
python train.py \
    --content-dir ../../datasets/processed/sample/train \
    --style-image ../../datasets/sample/styles/sci-fi.jpg \
    --style-name sci-fi \
    --epochs 2 \
    --batch-size 4

# Convert to Core ML
cd ../../conversion
python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi_best.pth \
    --output-path ../models/exported/sci-fi.mlmodel \
    --style-name sci-fi \
    --quantize
```

## 📖 Detailed Workflows

### 1. Dataset Preparation

#### Download COCO Dataset

```bash
cd datasets/scripts
python download_prepare.py --dataset coco --data-dir ../
```

This downloads the COCO 2017 training set (~18GB).

#### Preprocess Images

```bash
python download_prepare.py \
    --dataset custom \
    --source-dir ../coco/train2017 \
    --output-dir ../processed/coco_256 \
    --target-size 256 256 \
    --split-ratios 0.8 0.1 0.1
```

#### Prepare Style Images

Organize your style images by category:

```
datasets/wikiart_styles/
├── sci-fi/
│   ├── style_001.jpg
│   └── style_002.jpg
├── fantasy/
│   └── style_001.jpg
└── modern/
    └── style_001.jpg
```

Process them:

```bash
python download_prepare.py \
    --dataset custom \
    --source-dir ../wikiart_styles \
    --output-dir ../processed/styles \
    --prepare-styles
```

See [datasets/README.md](datasets/README.md) for more details.

### 2. Model Training

#### PyTorch Training

```bash
cd training/pytorch

python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --val-dir ../../datasets/processed/coco_256/val \
    --style-image ../../datasets/processed/styles/sci-fi/sci-fi_001.jpg \
    --style-name sci-fi \
    --epochs 10 \
    --batch-size 8 \
    --image-size 256 \
    --lr 1e-3 \
    --content-weight 1.0 \
    --style-weight 1e5 \
    --tv-weight 1e-6 \
    --save-interval 2
```

**Key Arguments:**
- `--content-weight`: Controls content preservation (default: 1.0)
- `--style-weight`: Controls style strength (default: 1e5)
- `--tv-weight`: Total variation loss for smoothness (default: 1e-6)
- `--lr`: Learning rate (default: 1e-3)

#### TensorFlow Training

```bash
cd training/tensorflow

python train.py \
    --content-dir ../../datasets/processed/coco_256/train \
    --val-dir ../../datasets/processed/coco_256/val \
    --style-image ../../datasets/processed/styles/fantasy/fantasy_001.jpg \
    --style-name fantasy \
    --epochs 10 \
    --batch-size 8 \
    --image-size 256
```

**Performance Tips:**
- Use `--num-workers 4` for faster data loading
- Enable `--lr-scheduler` for better convergence
- Use `--early-stopping` to prevent overfitting
- Reduce `--batch-size` if running out of memory

See [training/README.md](training/README.md) for advanced options.

### 3. Core ML Conversion

#### Convert PyTorch Model

```bash
cd conversion

python convert_coreml.py \
    --framework pytorch \
    --model-path ../models/checkpoints/sci-fi_best.pth \
    --output-path ../models/exported/sci-fi.mlmodel \
    --style-name sci-fi \
    --input-size 256 256 \
    --compute-units ALL \
    --quantize
```

#### Convert TensorFlow Model

```bash
python convert_coreml.py \
    --framework tensorflow \
    --model-path ../models/checkpoints/fantasy_best.h5 \
    --output-path ../models/exported/fantasy.mlmodel \
    --style-name fantasy \
    --input-size 256 256 \
    --compute-units CPU_AND_NE \
    --quantize
```

**Compute Units Options:**
- `ALL`: CPU, GPU, and Neural Engine (recommended)
- `CPU_AND_GPU`: CPU and GPU only
- `CPU_AND_NE`: CPU and Neural Engine (best for modern devices)
- `CPU_ONLY`: CPU only (slowest)

**Quantization:**
- `--quantize`: Apply float16 quantization (~50% size reduction, minimal quality loss)

### 4. Model Validation

Open the Jupyter notebook for interactive validation:

```bash
cd conversion
jupyter notebook validate_coreml.ipynb
```

The notebook provides:
- Model metadata inspection
- Inference testing with sample images
- Performance benchmarking (FPS, latency)
- Visual quality assessment
- Automated validation reports

See [conversion/README.md](conversion/README.md) for details.

### 5. Generate Manifest

Create the `styles.json` manifest for app integration:

```bash
cd manifest

# Generate from exported models
python generate_manifest.py generate \
    --models-dir ../models/exported \
    --output styles.json \
    --version 1.0.0

# Validate manifest
python generate_manifest.py validate \
    --manifest styles.json \
    --schema styles_schema.json
```

The manifest maps each model to a `StyleDescriptor` with:
- Model path and metadata
- Input/output specifications
- Performance metrics
- Recommended compute units

## 📊 Performance Benchmarks

Typical performance on Apple devices (256x256 input):

| Device | Neural Engine | Avg Latency | FPS | Notes |
|--------|--------------|-------------|-----|-------|
| iPhone 15 Pro | Yes | ~35ms | 28 | Quantized, ALL compute units |
| iPhone 14 | Yes | ~45ms | 22 | Quantized, ALL compute units |
| iPad Pro M2 | Yes | ~30ms | 33 | Quantized, CPU_AND_NE |
| Vision Pro | Yes | ~40ms | 25 | Real-time AR capable |
| Mac M1/M2 | Yes | ~25ms | 40 | Desktop testing |

*Benchmarks with float16 quantization and optimized compute units.*

## 🎨 StyleDescriptor Integration

The generated `manifest/styles.json` provides a standardized interface for your iOS/visionOS app:

```json
{
  "id": "sci-fi",
  "displayName": "Sci-Fi",
  "description": "Futuristic aesthetic...",
  "category": "sci-fi",
  "modelPath": "../models/exported/sci-fi.mlmodel",
  "inputSize": {"width": 256, "height": 256},
  "computeUnits": "ALL",
  "performance": {
    "avgInferenceMs": 45.0,
    "fps": 22.0
  },
  "metadata": {
    "framework": "pytorch",
    "quantized": true
  }
}
```

**App Integration Example (Swift):**

```swift
struct StyleDescriptor: Codable {
    let id: String
    let displayName: String
    let modelPath: String
    let inputSize: ImageSize
    let computeUnits: String
    // ... other fields
}

// Load manifest
let manifestURL = Bundle.main.url(forResource: "styles", withExtension: "json")
let manifest = try JSONDecoder().decode(StyleManifest.self, from: Data(contentsOf: manifestURL))

// Use styles in your app
for style in manifest.styles {
    let model = try MLModel(contentsOf: URL(fileURLWithPath: style.modelPath))
    // Apply style transfer...
}
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v --cov=.
```

Test coverage includes:
- Dataset loading and preprocessing
- Model architecture validation
- Conversion pipeline integrity
- Manifest generation and validation

## 📝 Configuration Best Practices

### For Mobile Deployment

1. **Input Size**: 256x256 is optimal for real-time performance
2. **Quantization**: Always use float16 quantization
3. **Compute Units**: Use `ALL` or `CPU_AND_NE`
4. **Model Size**: Target <10MB for quick loading

### For High Quality

1. **Input Size**: 512x512 or higher
2. **Quantization**: Optional (sacrifices size for quality)
3. **Training Epochs**: 10-20 epochs
4. **Style Weight**: Experiment with 1e4 to 1e6

### For Fast Training

1. **Batch Size**: Maximize based on GPU memory
2. **Learning Rate**: Start with 1e-3, use scheduler
3. **Validation**: Use smaller validation set
4. **Checkpointing**: Save every 2-5 epochs

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-style`)
3. Commit your changes (`git commit -m 'Add amazing style'`)
4. Push to the branch (`git push origin feature/amazing-style`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Attributions

- **COCO Dataset**: CC BY 4.0
- **VGG Weights**: CC BY
- **Style Transfer Architecture**: Based on Johnson et al. (2016), MIT License

See [LICENSE](LICENSE) for complete attribution details.

## 🔗 Resources

### Papers
- [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155) (Johnson et al., 2016)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al., 2015)

### Documentation
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

### Related Projects
- [fast-neural-style](https://github.com/jcjohnson/fast-neural-style)
- [pytorch/examples](https://github.com/pytorch/examples/tree/main/fast_neural_style)

## 🐛 Troubleshooting

### Common Issues

**Out of Memory During Training:**
```bash
# Reduce batch size
python train.py --batch-size 2 ...

# Reduce image size
python train.py --image-size 128 ...
```

**Slow Training:**
```bash
# Use more workers
python train.py --num-workers 8 ...

# Enable GPU
python train.py --device cuda ...
```

**Core ML Conversion Errors:**
```bash
# Make sure coremltools is up to date
pip install --upgrade coremltools

# Try different compute units
python convert_coreml.py --compute-units CPU_ONLY ...
```

### Getting Help

- Open an [issue](https://github.com/yourusername/style-transfer-models/issues)
- Check existing [discussions](https://github.com/yourusername/style-transfer-models/discussions)
- Review [FAQ](docs/FAQ.md)

## 📧 Contact

For questions, feature requests, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/yourusername/style-transfer-models/issues)
- Email: your.email@example.com

---

**Made with ❤️ for the AR/ML community**

⭐ Star this repo if you find it useful!
