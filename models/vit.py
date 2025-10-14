# models/vit.py

import torch
from torch import nn
from torchvision.models import vision_transformer

def create_vit_tiny_patch4_32(num_classes=10):
    """
    Creates an instance of a Vision Transformer in a "Tiny" configuration,
    specifically adapted for 32x32 images with 4x4 patches.
    
    This configuration is modeled after popular implementations, e.g., from the `timm` library.
    """
    # We use the flexible VisionTransformer class from torchvision
    model = vision_transformer.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,  # This corresponds to our 'n_embd'
        mlp_dim=192 * 4, # The MLP dimension is typically 4x the embedding dimension
        num_classes=num_classes,
        representation_size=None, # We don't use an extra projection head
    )
    return model