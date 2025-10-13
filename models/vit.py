# models/vit.py

import torch
from torch import nn
from torchvision.models import vision_transformer

def create_vit_tiny_patch4_32(num_classes=10):
    """
    Tworzy instancję Vision Transformer w konfiguracji "Tiny",
    dostosowaną do obrazów 32x32 z łatkami 4x4.
    
    Ta konfiguracja jest wzorowana na popularnych implementacjach, np. z biblioteki `timm`.
    """
    # Używamy elastycznej klasy VisionTransformer z torchvision
    model = vision_transformer.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,  # To jest nasz 'n_embd'
        mlp_dim=192 * 4, # Standardowo wymiar MLP to 4 * wymiar embeddingu
        num_classes=num_classes,
        representation_size=None, # Nie używamy dodatkowej głowicy projekcyjnej
    )
    return model