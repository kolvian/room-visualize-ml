#!/usr/bin/env python3
"""
PyTorch Style Transfer Training Script

Train fast neural style transfer models for specific artistic styles.
Supports checkpointing, visualization, and configurable hyperparameters.
"""

import argparse
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import StyleTransferNet, PerceptualLoss
from utils import (
    StyleTransferDataset,
    get_training_transforms,
    get_validation_transforms,
    load_style_image,
    save_checkpoint,
    load_checkpoint,
    save_image_tensor,
    setup_logger,
    AverageMeter,
    get_device,
    count_parameters,
    save_training_config,
    visualize_batch,
    EarlyStopping
)


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    style_features,
    device,
    epoch,
    logger,
    args
):
    """Train for one epoch."""
    model.train()
    
    losses = {
        'total': AverageMeter(),
        'content': AverageMeter(),
        'style': AverageMeter(),
        'tv': AverageMeter()
    }
    
    start_time = time.time()
    
    for batch_idx, content in enumerate(dataloader):
        content = content.to(device)
        
        # Forward pass
        styled = model(content)
        
        # Compute loss
        loss, loss_dict = criterion(styled, content, style_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update metrics
        batch_size = content.size(0)
        losses['total'].update(loss_dict['total'], batch_size)
        losses['content'].update(loss_dict['content'], batch_size)
        losses['style'].update(loss_dict['style'], batch_size)
        losses['tv'].update(loss_dict['tv'], batch_size)
        
        # Logging
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                f'Epoch: [{epoch}][{batch_idx}/{len(dataloader)}] '
                f'Loss: {losses["total"].avg:.4f} '
                f'(Content: {losses["content"].avg:.4f}, '
                f'Style: {losses["style"].avg:.4f}, '
                f'TV: {losses["tv"].avg:.4f}) '
                f'Time: {elapsed:.2f}s'
            )
        
        # Save sample outputs
        if batch_idx % args.sample_interval == 0:
            visualize_batch(
                content,
                styled,
                args.output_dir / 'samples',
                epoch,
                batch_idx
            )
    
    return losses['total'].avg


def validate(model, dataloader, criterion, style_features, device):
    """Validate the model."""
    model.eval()
    
    total_loss = AverageMeter()
    
    with torch.no_grad():
        for content in dataloader:
            content = content.to(device)
            styled = model(content)
            loss, loss_dict = criterion(styled, content, style_features)
            total_loss.update(loss_dict['total'], content.size(0))
    
    return total_loss.avg


def main():
    parser = argparse.ArgumentParser(
        description='Train PyTorch style transfer model'
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
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=0,
        help='Gradient clipping threshold (0 = no clipping)'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--lr-scheduler',
        action='store_true',
        help='Use learning rate scheduler'
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Use early stopping'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )
    
    # System arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
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
    
    # Logging arguments
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Log training info every N batches'
    )
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=500,
        help='Save sample outputs every N batches'
    )
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir = Path(args.output_dir) / args.style_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(args.output_dir / 'training.log')
    logger.info(f"Starting training for style: {args.style_name}")
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Load style image and precompute features
    logger.info(f"Loading style image: {args.style_image}")
    style_image = load_style_image(args.style_image, device=device)
    
    # Create model and loss
    logger.info("Creating model...")
    model = StyleTransferNet(num_residual_blocks=args.num_residual_blocks)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    criterion = PerceptualLoss(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight
    ).to(device)
    
    # Precompute style features
    logger.info("Precomputing style features...")
    with torch.no_grad():
        style_features = criterion.precompute_style_features(style_image)
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
    
    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = StyleTransferDataset(
        args.content_dir,
        transform=get_training_transforms(args.image_size)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info(f"Training samples: {len(train_dataset)}")
    
    val_loader = None
    if args.val_dir:
        val_dataset = StyleTransferDataset(
            args.val_dir,
            transform=get_validation_transforms(args.image_size)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Save training config
    config = vars(args)
    config['device'] = str(device)
    config['model_parameters'] = count_parameters(model)
    save_training_config(config, args.output_dir / 'config.json')
    
    # Training loop
    best_loss = float('inf')
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            style_features, device, epoch, logger, args
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, style_features, device)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_loss)
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_loss):
                    logger.info("Early stopping triggered!")
                    break
            
            current_loss = val_loss
        else:
            current_loss = train_loss
        
        # Save checkpoint
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, 0, current_loss,
                args.checkpoint_dir, args.style_name, is_best
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("\nTraining complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final model saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
