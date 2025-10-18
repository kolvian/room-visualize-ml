"""
PyTorch Style Transfer Model
Implements Fast Neural Style Transfer based on Johnson et al. (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Convolution layer with optional instance normalization and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=True):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """Upsampling layer using nearest neighbor + convolution (better than transpose conv)."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode='nearest')
        x = self.conv(x)
        x = self.norm(x)
        return x


class StyleTransferNet(nn.Module):
    """
    Fast Style Transfer Network (Johnson et al. 2016)
    
    Architecture:
    - 3 convolutional layers for encoding
    - 5 residual blocks
    - 3 upsampling layers for decoding
    - Output layer with tanh activation
    """
    
    def __init__(self, num_residual_blocks=5):
        super(StyleTransferNet, self).__init__()
        
        # Encoder: Downsampling layers
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        residual_layers = []
        for _ in range(num_residual_blocks):
            residual_layers.append(ResidualBlock(128))
        self.residual = nn.Sequential(*residual_layers)
        
        # Decoder: Upsampling layers
        self.decoder = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, kernel_size=9, stride=1, norm=False),
        )
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W) normalized to [0, 1]
        
        Returns:
            Stylized image tensor (B, 3, H, W) in range [0, 1]
        """
        out = self.encoder(x)
        out = self.residual(out)
        out = self.decoder(out)
        # Use tanh and scale to [0, 1]
        out = torch.tanh(out) * 0.5 + 0.5
        return out


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for perceptual loss.
    Extracts features from relu1_2, relu2_2, relu3_3, relu4_3 layers.
    """
    
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pretrained VGG16
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*list(vgg.features[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features[9:16])) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.features[16:23])) # relu4_3
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W) normalized to ImageNet stats
        
        Returns:
            Dictionary of feature maps at different layers
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        return {
            'relu1_2': h1,
            'relu2_2': h2,
            'relu3_3': h3,
            'relu4_3': h4
        }


def gram_matrix(features):
    """
    Compute Gram matrix for style representation.
    
    Args:
        features: Feature tensor (B, C, H, W)
    
    Returns:
        Gram matrix (B, C, C)
    """
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    # Normalize by number of elements
    gram = gram / (channels * height * width)
    return gram


class PerceptualLoss(nn.Module):
    """
    Perceptual loss combining content loss, style loss, and total variation loss.
    """
    
    def __init__(
        self,
        content_weight=1.0,
        style_weight=1e5,
        tv_weight=1e-6,
        style_layers=None,
        content_layers=None
    ):
        super(PerceptualLoss, self).__init__()
        
        self.feature_extractor = VGG16FeatureExtractor()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.style_layers = style_layers or ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_layers = content_layers or ['relu2_2']
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, output, content, style_features):
        """
        Args:
            output: Stylized image (B, 3, H, W) in [0, 1]
            content: Content image (B, 3, H, W) in [0, 1]
            style_features: Precomputed style features from style image
        
        Returns:
            Total loss, dict of individual losses
        """
        # Normalize to ImageNet stats
        output_norm = self.normalize_imagenet(output)
        content_norm = self.normalize_imagenet(content)
        
        # Extract features
        output_features = self.feature_extractor(output_norm)
        content_features = self.feature_extractor(content_norm)
        
        # Content loss
        content_loss = 0
        for layer in self.content_layers:
            content_loss += self.mse_loss(
                output_features[layer],
                content_features[layer]
            )
        content_loss = self.content_weight * content_loss
        
        # Style loss
        style_loss = 0
        for layer in self.style_layers:
            output_gram = gram_matrix(output_features[layer])
            style_gram = style_features[layer]
            style_loss += self.mse_loss(output_gram, style_gram)
        style_loss = self.style_weight * style_loss
        
        # Total variation loss (for smoothness)
        tv_loss = self.tv_weight * self.total_variation_loss(output)
        
        # Total loss
        total_loss = content_loss + style_loss + tv_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'content': content_loss.item(),
            'style': style_loss.item(),
            'tv': tv_loss.item()
        }
        
        return total_loss, loss_dict
    
    @staticmethod
    def normalize_imagenet(tensor):
        """Normalize tensor to ImageNet statistics."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        return (tensor - mean) / std
    
    @staticmethod
    def total_variation_loss(img):
        """Compute total variation loss for smoothness."""
        batch_size, channels, height, width = img.size()
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w
    
    def precompute_style_features(self, style_image):
        """
        Precompute Gram matrices for style image.
        
        Args:
            style_image: Style image tensor (B, 3, H, W) in [0, 1]
        
        Returns:
            Dictionary of Gram matrices for each style layer
        """
        style_norm = self.normalize_imagenet(style_image)
        style_features = self.feature_extractor(style_norm)
        
        style_grams = {}
        for layer in self.style_layers:
            style_grams[layer] = gram_matrix(style_features[layer])
        
        return style_grams


if __name__ == '__main__':
    # Test model
    model = StyleTransferNet()
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
