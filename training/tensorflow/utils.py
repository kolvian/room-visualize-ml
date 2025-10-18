"""
Utility functions for TensorFlow style transfer training
"""

import os
import json
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


def setup_logger(log_file=None):
    """Setup logging configuration."""
    logger = logging.getLogger('style_transfer_tf')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_dataset(
    image_dir,
    batch_size=4,
    image_size=256,
    shuffle=True,
    augment=True
):
    """
    Create TensorFlow dataset from image directory.
    
    Args:
        image_dir: Directory containing images
        batch_size: Batch size
        image_size: Target image size
        shuffle: Whether to shuffle dataset
        augment: Whether to apply data augmentation
    
    Returns:
        tf.data.Dataset
    """
    # Get image paths
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        str(f) for f in image_dir.rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_files))
    
    # Load and preprocess images
    def load_and_preprocess(path):
        # Load image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize
        img = tf.image.resize(img, [image_size, image_size])
        
        # Data augmentation
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        return img
    
    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_style_image(style_path, image_size=512):
    """
    Load and preprocess a style image.
    
    Args:
        style_path: Path to style image
        image_size: Size to resize style image
    
    Returns:
        Style image tensor (1, H, W, 3) in [0, 1]
    """
    img = tf.io.read_file(style_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = img / 255.0
    img = tf.expand_dims(img, 0)
    
    return img


def save_image_tensor(tensor, path):
    """
    Save image tensor to file.
    
    Args:
        tensor: Image tensor (H, W, 3) or (1, H, W, 3) in [0, 1]
        path: Output path
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Convert to numpy and clip
    img_array = np.array(tensor)
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    
    # Save image
    img = Image.fromarray(img_array)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_model(model, save_path, save_format='h5'):
    """
    Save Keras model.
    
    Args:
        model: Keras model
        save_path: Path to save model
        save_format: 'h5' or 'tf' (SavedModel format)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_format == 'h5':
        model.save(save_path, save_format='h5')
    else:
        model.save(save_path)
    
    print(f"Model saved to: {save_path}")


def load_model(model_path):
    """Load Keras model."""
    return keras.models.load_model(model_path, compile=False)


def save_training_config(config, output_path):
    """Save training configuration to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert non-serializable objects
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_serializable[key] = str(value)
        else:
            config_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


class CheckpointCallback(keras.callbacks.Callback):
    """Custom callback for saving checkpoints with style name."""
    
    def __init__(self, checkpoint_dir, style_name, save_freq=1):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.style_name = style_name
        self.save_freq = save_freq
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"{self.style_name}_epoch_{epoch+1}.h5"
            self.model.model.save(checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")
        
        # Save best model
        current_loss = logs.get('loss', float('inf'))
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.checkpoint_dir / f"{self.style_name}_best.h5"
            self.model.model.save(best_path)
            print(f"\nSaved best model: {best_path} (loss: {current_loss:.4f})")


class SampleCallback(keras.callbacks.Callback):
    """Callback to save sample outputs during training."""
    
    def __init__(self, sample_dir, content_images, save_freq=1):
        super().__init__()
        self.sample_dir = Path(sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.content_images = content_images
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # Generate styled images
            styled = self.model.model(self.content_images, training=False)
            
            # Save samples
            for i in range(min(4, len(styled))):
                save_image_tensor(
                    styled[i],
                    self.sample_dir / f"epoch_{epoch+1}_sample_{i}.jpg"
                )


def setup_gpus(gpu_memory_limit=None):
    """
    Setup GPU configuration.
    
    Args:
        gpu_memory_limit: Memory limit in MB (None = unlimited)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if specified
            if gpu_memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=gpu_memory_limit
                    )]
                )
            
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPUs available, using CPU")


def visualize_batch(content, styled, save_dir, epoch, step):
    """
    Save visualization of content and styled images.
    
    Args:
        content: Content image tensor (B, H, W, 3)
        styled: Styled image tensor (B, H, W, 3)
        save_dir: Directory to save visualizations
        epoch: Current epoch
        step: Current step
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save first image from batch
    save_image_tensor(content[0], save_dir / f"epoch_{epoch}_step_{step}_content.jpg")
    save_image_tensor(styled[0], save_dir / f"epoch_{epoch}_step_{step}_styled.jpg")
