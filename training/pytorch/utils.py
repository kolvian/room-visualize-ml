"""
Utility functions for PyTorch style transfer training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import logging


def setup_logger(log_file=None):
    """Setup logging configuration."""
    logger = logging.getLogger('style_transfer')
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


class StyleTransferDataset(Dataset):
    """Dataset for style transfer training."""
    
    def __init__(
        self,
        content_dir,
        transform=None,
        max_samples=None
    ):
        """
        Args:
            content_dir: Directory containing content images
            transform: Optional transform to apply to images
            max_samples: Maximum number of samples to use (None = use all)
        """
        self.content_dir = Path(content_dir)
        self.transform = transform
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = [
            f for f in self.content_dir.rglob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {content_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_training_transforms(image_size=256, crop_size=256):
    """Get training data transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def get_validation_transforms(image_size=256):
    """Get validation data transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def load_style_image(style_path, image_size=512, device='cuda'):
    """
    Load and preprocess a style image.
    
    Args:
        style_path: Path to style image
        image_size: Size to resize style image
        device: Device to load tensor on
    
    Returns:
        Style image tensor (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    style_image = Image.open(style_path).convert('RGB')
    style_tensor = transform(style_image).unsqueeze(0).to(device)
    
    return style_tensor


def save_checkpoint(
    model,
    optimizer,
    epoch,
    iteration,
    loss,
    checkpoint_dir,
    style_name,
    is_best=False
):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"{style_name}_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / f"{style_name}_best.pth"
        torch.save(checkpoint, best_path)
    
    # Save latest checkpoint (for resuming)
    latest_path = checkpoint_dir / f"{style_name}_latest.pth"
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Tuple of (epoch, iteration, loss)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    iteration = checkpoint.get('iteration', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return epoch, iteration, loss


def save_image_tensor(tensor, path):
    """
    Save image tensor to file.
    
    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        path: Output path
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Ensure tensor is on CPU and in [0, 1] range
    tensor = tensor.cpu().clamp(0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    
    # Save image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device(gpu_id=0):
    """Get device for training."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_config(config, output_path):
    """Save training configuration to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for JSON serialization
    json_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            json_config[key] = str(value)
        else:
            json_config[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_config, f, indent=2)


def load_training_config(config_path):
    """Load training configuration from JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def denormalize_imagenet(tensor):
    """Denormalize tensor from ImageNet stats to [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def visualize_batch(content, styled, save_dir, epoch, step):
    """
    Save visualization of content and styled images.
    
    Args:
        content: Content image tensor (B, 3, H, W)
        styled: Styled image tensor (B, 3, H, W)
        save_dir: Directory to save visualizations
        epoch: Current epoch
        step: Current step
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save first image from batch
    save_image_tensor(content, save_dir / f"epoch_{epoch}_step_{step}_content.jpg")
    save_image_tensor(styled, save_dir / f"epoch_{epoch}_step_{step}_styled.jpg")
