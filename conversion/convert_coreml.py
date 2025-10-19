#!/usr/bin/env python3
"""
Core ML Conversion Script

Convert trained PyTorch or TensorFlow style transfer models to Core ML format
for deployment on iOS/visionOS devices.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import coremltools as ct
    from coremltools.models import MLModel
except ImportError:
    print("ERROR: coremltools not installed. Install with: pip install coremltools")
    exit(1)


class ModelConverter:
    """Handles conversion of style transfer models to Core ML."""
    
    def __init__(self, framework='pytorch'):
        """
        Args:
            framework: 'pytorch' or 'tensorflow'
        """
        self.framework = framework
    
    def convert_pytorch_model(
        self,
        model_path,
        output_path,
        input_shape=(1, 3, 256, 256),
        compute_units='ALL',
        quantize=False
    ):
        """
        Convert PyTorch model to Core ML.
        
        Args:
            model_path: Path to .pth checkpoint
            output_path: Path for output .mlmodel
            input_shape: Input shape (B, C, H, W)
            compute_units: 'ALL', 'CPU_AND_GPU', 'CPU_ONLY', 'CPU_AND_NE'
            quantize: Whether to apply float16 quantization
        
        Returns:
            Path to converted model
        """
        try:
            import torch
            import torchvision
        except ImportError:
            print("ERROR: PyTorch not installed")
            return None
        
        # Import model architecture
        import sys
        training_path = Path(__file__).parent.parent / 'training' / 'pytorch'
        sys.path.insert(0, str(training_path))
        from model import StyleTransferNet
        
        print(f"Loading PyTorch model from: {model_path}")
        
        # Load model
        model = StyleTransferNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        print(f"Model loaded. Converting to Core ML...")
        
        # Create example input
        example_input = torch.rand(input_shape)
        
        # Trace model
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to Core ML
        _, _, h, w = input_shape
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name='input_image',
                shape=input_shape,
                dtype=np.float32
            )],
            outputs=[ct.TensorType(name='stylized_image')],
            compute_units=self._get_compute_units(compute_units),
            minimum_deployment_target=ct.target.iOS15,
        )
        
        # Quantize if requested
        if quantize:
            print("Applying float16 quantization...")
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
                mlmodel, nbits=16
            )
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path))
        
        print(f"✓ Model converted successfully: {output_path}")
        return output_path
    
    def convert_tensorflow_model(
        self,
        model_path,
        output_path,
        input_shape=(1, 256, 256, 3),
        compute_units='ALL',
        quantize=False
    ):
        """
        Convert TensorFlow/Keras model to Core ML.
        
        Args:
            model_path: Path to .h5 or SavedModel
            output_path: Path for output .mlmodel
            input_shape: Input shape (B, H, W, C)
            compute_units: 'ALL', 'CPU_AND_GPU', 'CPU_ONLY', 'CPU_AND_NE'
            quantize: Whether to apply float16 quantization
        
        Returns:
            Path to converted model
        """
        try:
            import tensorflow as tf
        except ImportError:
            print("ERROR: TensorFlow not installed")
            return None
        
        print(f"Loading TensorFlow model from: {model_path}")
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        print(f"Model loaded. Converting to Core ML...")
        
        # Convert to Core ML
        _, h, w, c = input_shape
        
        mlmodel = ct.convert(
            model,
            inputs=[ct.TensorType(
                name='input_image',
                shape=input_shape,
                dtype=np.float32
            )],
            outputs=[ct.TensorType(name='stylized_image')],
            compute_units=self._get_compute_units(compute_units),
            minimum_deployment_target=ct.target.iOS15,
        )
        
        # Quantize if requested
        if quantize:
            print("Applying float16 quantization...")
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
                mlmodel, nbits=16
            )
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path))
        
        print(f"✓ Model converted successfully: {output_path}")
        return output_path
    
    @staticmethod
    def _get_compute_units(compute_units_str):
        """Convert string to CoreML compute units enum."""
        compute_units_map = {
            'ALL': ct.ComputeUnit.ALL,
            'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
            'CPU_ONLY': ct.ComputeUnit.CPU_ONLY,
            'CPU_AND_NE': ct.ComputeUnit.CPU_AND_NE,
        }
        return compute_units_map.get(compute_units_str.upper(), ct.ComputeUnit.ALL)
    
    def add_metadata(
        self,
        model_path,
        style_name,
        author='Style Transfer Models',
        description='Fast neural style transfer model',
        version='1.0'
    ):
        """
        Add metadata to Core ML model.
        
        Args:
            model_path: Path to .mlmodel file
            style_name: Name of the style
            author: Model author
            description: Model description
            version: Model version
        """
        print(f"Adding metadata to {model_path}...")
        
        # Load the model
        model = ct.models.MLModel(str(model_path))
        
        # Set metadata using the spec
        spec = model.get_spec()
        spec.description.metadata.shortDescription = f"{style_name.title()} style transfer model"
        spec.description.metadata.author = author
        spec.description.metadata.versionString = version
        spec.description.metadata.userDefined['style'] = style_name
        spec.description.metadata.userDefined['framework'] = self.framework
        
        # Update input/output descriptions
        spec.description.input[0].shortDescription = 'Input image (RGB, normalized to [0, 1])'
        spec.description.output[0].shortDescription = 'Stylized output image (RGB, [0, 1])'
        
        # Save updated model (preserves weights for mlpackage)
        model = ct.models.MLModel(spec, weights_dir=str(model_path) if str(model_path).endswith('.mlpackage') else None)
        model.save(str(model_path))
        
        print(f"✓ Metadata added successfully")
    
    def validate_model(self, model_path, test_image_path=None):
        """
        Validate Core ML model.
        
        Args:
            model_path: Path to .mlmodel file
            test_image_path: Optional path to test image
        
        Returns:
            Validation results dict
        """
        print(f"\nValidating model: {model_path}")
        
        model = MLModel(str(model_path))
        spec = model.get_spec()
        
        # Get model info
        input_desc = spec.description.input[0]
        output_desc = spec.description.output[0]
        
        print(f"✓ Model loaded successfully")
        print(f"  Input: {input_desc.name}, shape: {input_desc.type.multiArrayType.shape}")
        print(f"  Output: {output_desc.name}, shape: {output_desc.type.multiArrayType.shape}")
        
        # Test inference if test image provided
        if test_image_path:
            print(f"\nTesting inference with: {test_image_path}")
            
            # Load and preprocess image
            img = Image.open(test_image_path).convert('RGB')
            
            # Get input shape
            input_shape = input_desc.type.multiArrayType.shape
            if self.framework == 'pytorch':
                # PyTorch: (B, C, H, W)
                h, w = int(input_shape[2]), int(input_shape[3])
            else:
                # TensorFlow: (B, H, W, C)
                h, w = int(input_shape[1]), int(input_shape[2])
            
            img = img.resize((w, h))
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Prepare input
            if self.framework == 'pytorch':
                # Convert to (1, 3, H, W)
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)
            else:
                # Keep as (1, H, W, 3)
                img_array = np.expand_dims(img_array, axis=0)
            
            # Run inference
            output = model.predict({'input_image': img_array})
            
            print(f"✓ Inference successful")
            print(f"  Output shape: {output['stylized_image'].shape}")
            print(f"  Output range: [{output['stylized_image'].min():.3f}, {output['stylized_image'].max():.3f}]")
        
        return {
            'valid': True,
            'input_shape': list(input_desc.type.multiArrayType.shape),
            'output_shape': list(output_desc.type.multiArrayType.shape)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Convert style transfer models to Core ML format'
    )
    
    parser.add_argument(
        '--framework',
        type=str,
        required=True,
        choices=['pytorch', 'tensorflow'],
        help='Source framework'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.pth or .h5)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Output path for .mlmodel file'
    )
    parser.add_argument(
        '--style-name',
        type=str,
        required=True,
        help='Name of the style (e.g., sci-fi, fantasy)'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Input image size (height width)'
    )
    parser.add_argument(
        '--compute-units',
        type=str,
        default='ALL',
        choices=['ALL', 'CPU_AND_GPU', 'CPU_ONLY', 'CPU_AND_NE'],
        help='Compute units for model execution'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply float16 quantization'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        help='Test image path for validation'
    )
    parser.add_argument(
        '--author',
        type=str,
        default='Style Transfer Models',
        help='Model author'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='1.0',
        help='Model version'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = ModelConverter(framework=args.framework)
    
    # Convert model
    h, w = args.input_size
    
    if args.framework == 'pytorch':
        input_shape = (1, 3, h, w)
        output_path = converter.convert_pytorch_model(
            args.model_path,
            args.output_path,
            input_shape=input_shape,
            compute_units=args.compute_units,
            quantize=args.quantize
        )
    else:  # tensorflow
        input_shape = (1, h, w, 3)
        output_path = converter.convert_tensorflow_model(
            args.model_path,
            args.output_path,
            input_shape=input_shape,
            compute_units=args.compute_units,
            quantize=args.quantize
        )
    
    if output_path:
        # Add metadata - DISABLED for mlpackage to preserve weights
        # TODO: Fix metadata addition for mlpackage format
        # converter.add_metadata(
        #     output_path,
        #     style_name=args.style_name,
        #     author=args.author,
        #     version=args.version
        # )
        
        # Validate model
        validation_results = converter.validate_model(
            output_path,
            test_image_path=args.test_image
        )
        
        # Save conversion info
        conversion_info = {
            'framework': args.framework,
            'source_model': str(args.model_path),
            'output_model': str(output_path),
            'style_name': args.style_name,
            'input_size': args.input_size,
            'compute_units': args.compute_units,
            'quantized': args.quantize,
            'validation': validation_results
        }
        
        info_path = Path(args.output_path).with_suffix('.json')
        with open(info_path, 'w') as f:
            json.dump(conversion_info, f, indent=2)
        
        print(f"\n✓ Conversion info saved to: {info_path}")
        print(f"\n🎉 Conversion complete! Model ready for iOS/visionOS deployment.")


if __name__ == '__main__':
    main()
