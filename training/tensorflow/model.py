"""
TensorFlow Style Transfer Model
Implements Fast Neural Style Transfer using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def conv_layer(x, filters, kernel_size, strides, activation=True):
    """Convolutional layer with instance normalization."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False
    )(x)
    x = layers.LayerNormalization()(x)  # Instance norm approximation
    if activation:
        x = layers.ReLU()(x)
    return x


def residual_block(x, filters):
    """Residual block with two convolutional layers."""
    residual = x
    
    x = conv_layer(x, filters, kernel_size=3, strides=1)
    x = conv_layer(x, filters, kernel_size=3, strides=1, activation=False)
    
    x = layers.Add()([x, residual])
    return x


def upsample_layer(x, filters, kernel_size, strides):
    """Upsampling layer using resize + convolution."""
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=1,
        padding='same',
        use_bias=False
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_style_transfer_model(input_shape=(256, 256, 3), num_residual_blocks=5):
    """
    Build fast style transfer model.
    
    Architecture:
    - Encoder: 3 convolutional layers (downsampling)
    - Transformer: 5 residual blocks
    - Decoder: 3 upsampling layers
    - Output: tanh activation scaled to [0, 1]
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_residual_blocks: Number of residual blocks
    
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    x = conv_layer(inputs, 32, kernel_size=9, strides=1)
    x = conv_layer(x, 64, kernel_size=3, strides=2)
    x = conv_layer(x, 128, kernel_size=3, strides=2)
    
    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, 128)
    
    # Decoder
    x = upsample_layer(x, 64, kernel_size=3, strides=2)
    x = upsample_layer(x, 32, kernel_size=3, strides=2)
    
    # Output layer
    x = layers.Conv2D(3, kernel_size=9, strides=1, padding='same')(x)
    outputs = layers.Activation('tanh')(x)
    outputs = layers.Lambda(lambda x: (x + 1.0) / 2.0)(outputs)  # Scale to [0, 1]
    
    model = Model(inputs=inputs, outputs=outputs, name='StyleTransferNet')
    return model


class VGGFeatureExtractor(Model):
    """
    VGG19 feature extractor for perceptual loss.
    Extracts features from multiple layers.
    """
    
    def __init__(self, layer_names=None):
        super(VGGFeatureExtractor, self).__init__()
        
        if layer_names is None:
            layer_names = [
                'block1_conv2',
                'block2_conv2',
                'block3_conv3',
                'block4_conv3'
            ]
        
        # Load pretrained VGG19
        vgg = keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False
        
        # Extract layers
        outputs = [vgg.get_layer(name).output for name in layer_names]
        
        self.feature_model = Model(
            inputs=vgg.input,
            outputs=outputs,
            name='VGG19_Features'
        )
        self.layer_names = layer_names
    
    def call(self, x):
        """
        Extract features from VGG19.
        
        Args:
            x: Input tensor (B, H, W, 3) normalized to ImageNet stats
        
        Returns:
            List of feature tensors
        """
        return self.feature_model(x)


def preprocess_vgg(image):
    """Preprocess image for VGG (ImageNet normalization)."""
    # Convert from [0, 1] to [0, 255]
    image = image * 255.0
    # Apply VGG preprocessing (RGB to BGR and zero-center)
    return keras.applications.vgg19.preprocess_input(image)


def gram_matrix(features):
    """
    Compute Gram matrix for style representation.
    
    Args:
        features: Feature tensor (B, H, W, C)
    
    Returns:
        Gram matrix (B, C, C)
    """
    batch_size, height, width, channels = tf.shape(features)[0], \
        tf.shape(features)[1], tf.shape(features)[2], tf.shape(features)[3]
    
    # Reshape to (B, H*W, C)
    features = tf.reshape(features, [batch_size, height * width, channels])
    
    # Compute Gram matrix: (B, C, H*W) @ (B, H*W, C) -> (B, C, C)
    features_t = tf.transpose(features, perm=[0, 2, 1])
    gram = tf.matmul(features_t, features)
    
    # Normalize
    gram = gram / tf.cast(height * width * channels, tf.float32)
    
    return gram


class PerceptualLoss(keras.losses.Loss):
    """
    Perceptual loss combining content, style, and total variation losses.
    """
    
    def __init__(
        self,
        content_weight=1.0,
        style_weight=1e5,
        tv_weight=1e-6,
        style_layers=None,
        content_layers=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        if style_layers is None:
            style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        if content_layers is None:
            content_layers = ['block2_conv2']
        
        all_layers = list(set(style_layers + content_layers))
        self.feature_extractor = VGGFeatureExtractor(all_layers)
        
        self.style_layer_indices = [all_layers.index(layer) for layer in style_layers]
        self.content_layer_indices = [all_layers.index(layer) for layer in content_layers]
        
        self.style_features = None
    
    def set_style_features(self, style_image):
        """
        Precompute style features (Gram matrices) from style image.
        
        Args:
            style_image: Style image tensor (B, H, W, 3) in [0, 1]
        """
        style_preprocessed = preprocess_vgg(style_image)
        features = self.feature_extractor(style_preprocessed)
        
        self.style_features = [
            gram_matrix(features[i]) for i in self.style_layer_indices
        ]
    
    def call(self, y_true, y_pred):
        """
        Compute perceptual loss.
        
        Args:
            y_true: Content image (B, H, W, 3) in [0, 1]
            y_pred: Stylized image (B, H, W, 3) in [0, 1]
        
        Returns:
            Total loss
        """
        # Preprocess for VGG
        content_preprocessed = preprocess_vgg(y_true)
        output_preprocessed = preprocess_vgg(y_pred)
        
        # Extract features
        content_features = self.feature_extractor(content_preprocessed)
        output_features = self.feature_extractor(output_preprocessed)
        
        # Content loss
        content_loss = 0.0
        for idx in self.content_layer_indices:
            content_loss += tf.reduce_mean(
                tf.square(output_features[idx] - content_features[idx])
            )
        content_loss = self.content_weight * content_loss
        
        # Style loss
        style_loss = 0.0
        if self.style_features is not None:
            for i, idx in enumerate(self.style_layer_indices):
                output_gram = gram_matrix(output_features[idx])
                style_gram = self.style_features[i]
                style_loss += tf.reduce_mean(tf.square(output_gram - style_gram))
        style_loss = self.style_weight * style_loss
        
        # Total variation loss
        tv_loss = self.tv_weight * self.total_variation_loss(y_pred)
        
        # Total loss
        total_loss = content_loss + style_loss + tv_loss
        
        return total_loss
    
    @staticmethod
    def total_variation_loss(image):
        """Compute total variation loss for smoothness."""
        # Differences along height and width
        diff_h = image[:, 1:, :, :] - image[:, :-1, :, :]
        diff_w = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        tv = tf.reduce_mean(tf.abs(diff_h)) + tf.reduce_mean(tf.abs(diff_w))
        return tv


class StyleTransferTrainer(Model):
    """Wrapper model for training with custom loss tracking."""
    
    def __init__(self, style_transfer_model, perceptual_loss):
        super().__init__()
        self.model = style_transfer_model
        self.perceptual_loss = perceptual_loss
    
    def call(self, inputs):
        return self.model(inputs)
    
    def train_step(self, data):
        content = data
        
        with tf.GradientTape() as tape:
            styled = self.model(content, training=True)
            loss = self.perceptual_loss(content, styled)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {'loss': loss}
    
    def test_step(self, data):
        content = data
        styled = self.model(content, training=False)
        loss = self.perceptual_loss(content, styled)
        
        return {'loss': loss}


if __name__ == '__main__':
    # Test model
    print("Building model...")
    model = build_style_transfer_model(input_shape=(256, 256, 3))
    model.summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = tf.random.normal((1, 256, 256, 3))
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
