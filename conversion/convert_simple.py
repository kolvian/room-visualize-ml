#!/usr/bin/env python3
"""
Simple Core ML converter for style transfer models.
Works around coremltools compatibility issues.
"""

import argparse
import torch
import sys
from pathlib import Path

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training' / 'pytorch'))
from model import StyleTransferNet


def convert_to_torchscript(model_path, output_path, input_size=256):
    """Convert PyTorch model to TorchScript (mobile-compatible format)."""
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = StyleTransferNet()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"Model loaded. Converting to TorchScript...")
    
    # Create example input
    example_input = torch.rand(1, 3, input_size, input_size)
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save TorchScript model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.jit.save(optimized_model, str(output_path))
    
    print(f"\n✓ Model saved to: {output_path}")
    print(f"   Format: TorchScript (optimized for inference)")
    print(f"   Input size: {input_size}x{input_size}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to mobile format')
    parser.add_argument('--model-path', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output-path', required=True, help='Output path for model')
    parser.add_argument('--input-size', type=int, default=256, help='Input image size')
    
    args = parser.parse_args()
    
    # Convert to TorchScript
    torchscript_path = args.output_path.replace('.mlmodel', '.pt')
    convert_to_torchscript(args.model_path, torchscript_path, args.input_size)
    
    print(f"\n📱 Note: TorchScript format created (.pt file)")
    print(f"   For Core ML conversion, you may need:")
    print(f"   1. Python 3.9-3.11 (current: {sys.version_info.major}.{sys.version_info.minor})")
    print(f"   2. Compatible coremltools version")
    print(f"\n   The .pt file can be used with PyTorch Mobile on iOS!")


if __name__ == '__main__':
    main()
