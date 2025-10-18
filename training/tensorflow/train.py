#!/usr/bin/env python3
"""
TensorFlow Style Transfer Training Script

Train fast neural style transfer models using TensorFlow/Keras.
"""

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from model import (
    build_style_transfer_model,
    PerceptualLoss,
    StyleTransferTrainer
)
from utils import (
    setup_logger,
    create_dataset,
    load_style_image,
    save_model,
    save_training_config,
    count_parameters,
    CheckpointCallback,
    SampleCallback,
    setup_gpus
)


def main():
    parser = argparse.ArgumentParser(
        description='Train TensorFlow style transfer model'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--content-dir',
        type=str,
        required=True,
        help='Directory containing content images'
    )
    parser.add_argument(
        '--style-image',
        type=str,
        required=True,
        help='Path to style image'
    )
    parser.add_argument(
        '--style-name',
        type=str,
        required=True,
        help='Name of the style (e.g., sci-fi, fantasy)'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        help='Directory containing validation images'
    )
    
    # Model arguments
    parser.add_argument(
        '--num-residual-blocks',
        type=int,
        default=5,
        help='Number of residual blocks in model'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='Training image size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--content-weight',
        type=float,
        default=1.0,
        help='Content loss weight'
    )
    parser.add_argument(
        '--style-weight',
        type=float,
        default=1e5,
        help='Style loss weight'
    )
    parser.add_argument(
        '--tv-weight',
        type=float,
        default=1e-6,
        help='Total variation loss weight'
    )
    
    # System arguments
    parser.add_argument(
        '--gpu-memory-limit',
        type=int,
        help='GPU memory limit in MB'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Checkpoint arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for checkpoints and samples'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='../../models/checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='Save checkpoint every N epochs'
    )
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir = Path(args.output_dir) / args.style_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(args.output_dir / 'training.log')
    logger.info(f"Starting training for style: {args.style_name}")
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    tf.random.set_seed(args.seed)
    
    # Setup GPUs
    setup_gpus(args.gpu_memory_limit)
    
    # Load style image
    logger.info(f"Loading style image: {args.style_image}")
    style_image = load_style_image(args.style_image)
    
    # Create model
    logger.info("Creating model...")
    style_transfer_model = build_style_transfer_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_residual_blocks=args.num_residual_blocks
    )
    logger.info(f"Model parameters: {count_parameters(style_transfer_model):,}")
    
    # Create perceptual loss
    perceptual_loss = PerceptualLoss(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight
    )
    
    # Precompute style features
    logger.info("Precomputing style features...")
    perceptual_loss.set_style_features(style_image)
    
    # Create trainer model
    trainer = StyleTransferTrainer(style_transfer_model, perceptual_loss)
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    
    # Compile model
    trainer.compile(optimizer=optimizer)
    
    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        style_transfer_model.load_weights(args.resume)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = create_dataset(
        args.content_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        augment=True
    )
    
    val_dataset = None
    if args.val_dir:
        val_dataset = create_dataset(
            args.val_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            shuffle=False,
            augment=False
        )
    
    # Create sample content images for visualization
    sample_iterator = iter(train_dataset)
    sample_content = next(sample_iterator)
    
    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            args.checkpoint_dir,
            args.style_name,
            save_freq=args.save_interval
        ),
        SampleCallback(
            args.output_dir / 'samples',
            sample_content,
            save_freq=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=args.output_dir / 'logs',
            update_freq='epoch'
        ),
        keras.callbacks.CSVLogger(
            args.output_dir / 'training_history.csv'
        )
    ]
    
    # Add early stopping if validation set is provided
    if val_dataset:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                verbose=1
            )
        )
    
    # Save training config
    config = vars(args)
    config['model_parameters'] = count_parameters(style_transfer_model)
    save_training_config(config, args.output_dir / 'config.json')
    
    # Training
    logger.info("Starting training...")
    history = trainer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / f"{args.style_name}_final.h5"
    save_model(style_transfer_model, final_path)
    
    logger.info("\nTraining complete!")
    logger.info(f"Final model saved to: {final_path}")
    
    # Print training summary
    final_loss = history.history['loss'][-1]
    logger.info(f"Final training loss: {final_loss:.4f}")
    
    if val_dataset:
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Final validation loss: {final_val_loss:.4f}")


if __name__ == '__main__':
    main()
