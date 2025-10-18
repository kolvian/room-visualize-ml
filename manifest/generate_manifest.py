#!/usr/bin/env python3
"""
Manifest Generator

Generate and update the styles.json manifest from exported Core ML models.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import os

try:
    import coremltools as ct
    from coremltools.models import MLModel
    HAS_COREML = True
except ImportError:
    HAS_COREML = False
    print("Warning: coremltools not installed. Model inspection features disabled.")


def get_model_info(model_path):
    """Extract information from Core ML model."""
    if not HAS_COREML:
        return None
    
    try:
        model = MLModel(str(model_path))
        spec = model.get_spec()
        
        # Get input/output shapes
        input_desc = spec.description.input[0]
        output_desc = spec.description.output[0]
        
        input_shape = list(input_desc.type.multiArrayType.shape)
        output_shape = list(output_desc.type.multiArrayType.shape)
        
        # Determine framework from shape
        if len(input_shape) == 4:
            if input_shape[1] == 3:  # (B, C, H, W) - PyTorch
                framework = 'pytorch'
                h, w = int(input_shape[2]), int(input_shape[3])
            else:  # (B, H, W, C) - TensorFlow
                framework = 'tensorflow'
                h, w = int(input_shape[1]), int(input_shape[2])
        else:
            framework = 'unknown'
            h, w = 256, 256
        
        # Get metadata
        metadata = {}
        if hasattr(spec.description, 'metadata') and spec.description.metadata.userDefined:
            metadata = dict(spec.description.metadata.userDefined)
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return {
            'inputSize': {'width': w, 'height': h},
            'outputSize': {'width': w, 'height': h},
            'framework': metadata.get('framework', framework),
            'author': model.author or 'Unknown',
            'version': model.version or '1.0',
            'modelSizeMB': round(file_size_mb, 2),
            'metadata': metadata
        }
    
    except Exception as e:
        print(f"Error reading model {model_path}: {e}")
        return None


def create_style_descriptor(
    style_id,
    display_name,
    category,
    model_path,
    description='',
    compute_units='ALL',
    performance=None,
    notes='',
    enabled=True
):
    """Create a StyleDescriptor object."""
    
    # Get model info if available
    model_info = get_model_info(model_path) if Path(model_path).exists() else None
    
    descriptor = {
        'id': style_id,
        'displayName': display_name,
        'description': description or f"{display_name} style transfer",
        'category': category,
        'modelPath': str(Path(model_path).relative_to(Path.cwd())),
        'thumbnailPath': f"../models/exported/thumbnails/{style_id}.jpg",
        'inputSize': model_info['inputSize'] if model_info else {'width': 256, 'height': 256},
        'outputSize': model_info['outputSize'] if model_info else {'width': 256, 'height': 256},
        'computeUnits': compute_units,
        'enabled': enabled
    }
    
    # Add performance if provided
    if performance:
        descriptor['performance'] = performance
    elif model_info:
        descriptor['performance'] = {
            'modelSizeMB': model_info['modelSizeMB']
        }
    
    # Add metadata
    if model_info:
        descriptor['metadata'] = {
            'author': model_info['author'],
            'version': model_info['version'],
            'framework': model_info['framework'],
            'quantized': 'quantized' in str(model_path).lower(),
            'created': datetime.now().isoformat() + 'Z'
        }
    
    # Add notes if provided
    if notes:
        descriptor['notes'] = notes
    
    return descriptor


def generate_manifest(models_dir, output_path, version='1.0.0'):
    """
    Generate manifest from directory of .mlmodel files.
    
    Args:
        models_dir: Directory containing .mlmodel files
        output_path: Path to output manifest JSON
        version: Manifest version
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return
    
    # Find all .mlmodel files
    model_files = list(models_dir.glob('*.mlmodel'))
    
    if not model_files:
        print(f"Warning: No .mlmodel files found in {models_dir}")
    
    # Create style descriptors
    styles = []
    
    for model_path in sorted(model_files):
        style_id = model_path.stem
        
        # Infer category from style name
        category_map = {
            'sci-fi': 'sci-fi',
            'scifi': 'sci-fi',
            'fantasy': 'fantasy',
            'modern': 'modern',
            'abstract': 'abstract',
            'impressionist': 'impressionism',
            'impressionism': 'impressionism',
            'expressionist': 'expressionism',
            'expressionism': 'expressionism'
        }
        
        category = category_map.get(style_id.lower(), 'other')
        display_name = style_id.replace('-', ' ').replace('_', ' ').title()
        
        print(f"Processing: {style_id}")
        
        descriptor = create_style_descriptor(
            style_id=style_id,
            display_name=display_name,
            category=category,
            model_path=model_path
        )
        
        styles.append(descriptor)
    
    # Create manifest
    manifest = {
        'version': version,
        'updated': datetime.now().isoformat() + 'Z',
        'styles': styles
    }
    
    # Save manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest generated: {output_path}")
    print(f"  Version: {version}")
    print(f"  Styles: {len(styles)}")


def update_manifest(manifest_path, style_id, updates):
    """
    Update an existing style in the manifest.
    
    Args:
        manifest_path: Path to manifest JSON
        style_id: ID of style to update
        updates: Dictionary of fields to update
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Find and update style
    found = False
    for style in manifest['styles']:
        if style['id'] == style_id:
            style.update(updates)
            found = True
            print(f"✓ Updated style: {style_id}")
            break
    
    if not found:
        print(f"Warning: Style '{style_id}' not found in manifest")
        return
    
    # Update manifest timestamp
    manifest['updated'] = datetime.now().isoformat() + 'Z'
    
    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Manifest updated: {manifest_path}")


def validate_manifest(manifest_path, schema_path=None):
    """
    Validate manifest against JSON schema.
    
    Args:
        manifest_path: Path to manifest JSON
        schema_path: Optional path to JSON schema
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Basic validation
    required_fields = ['version', 'updated', 'styles']
    for field in required_fields:
        if field not in manifest:
            print(f"✗ Missing required field: {field}")
            return False
    
    # Validate each style
    required_style_fields = ['id', 'displayName', 'category', 'modelPath', 'inputSize', 'computeUnits']
    
    for i, style in enumerate(manifest['styles']):
        for field in required_style_fields:
            if field not in style:
                print(f"✗ Style {i} missing required field: {field}")
                return False
        
        # Check if model file exists
        model_path = Path(manifest_path).parent / style['modelPath']
        if not model_path.exists():
            print(f"⚠ Warning: Model file not found for '{style['id']}': {model_path}")
    
    # JSON Schema validation (if jsonschema available)
    if schema_path:
        try:
            import jsonschema
            
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            jsonschema.validate(manifest, schema)
            print("✓ Manifest validates against schema")
        
        except ImportError:
            print("Warning: jsonschema not installed, skipping schema validation")
        except jsonschema.ValidationError as e:
            print(f"✗ Schema validation error: {e.message}")
            return False
    
    print(f"✓ Manifest is valid")
    print(f"  Version: {manifest['version']}")
    print(f"  Styles: {len(manifest['styles'])}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate and manage style transfer model manifest'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate manifest from models')
    gen_parser.add_argument(
        '--models-dir',
        type=str,
        default='../models/exported',
        help='Directory containing .mlmodel files'
    )
    gen_parser.add_argument(
        '--output',
        type=str,
        default='./styles.json',
        help='Output manifest path'
    )
    gen_parser.add_argument(
        '--version',
        type=str,
        default='1.0.0',
        help='Manifest version'
    )
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update style in manifest')
    update_parser.add_argument(
        '--manifest',
        type=str,
        default='./styles.json',
        help='Manifest path'
    )
    update_parser.add_argument(
        '--style-id',
        type=str,
        required=True,
        help='Style ID to update'
    )
    update_parser.add_argument(
        '--updates',
        type=str,
        required=True,
        help='JSON string of updates'
    )
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate manifest')
    val_parser.add_argument(
        '--manifest',
        type=str,
        default='./styles.json',
        help='Manifest path'
    )
    val_parser.add_argument(
        '--schema',
        type=str,
        default='./styles_schema.json',
        help='Schema path'
    )
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_manifest(args.models_dir, args.output, args.version)
    
    elif args.command == 'update':
        updates = json.loads(args.updates)
        update_manifest(args.manifest, args.style_id, updates)
    
    elif args.command == 'validate':
        validate_manifest(args.manifest, args.schema)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
