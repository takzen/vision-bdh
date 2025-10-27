# interpretability/models_for_viz/vision_bdh_v2.py
"""
MODIFIED VERSION FOR VISUALIZATION ONLY!

This is a modified copy of models/vit.py that returns 
attention maps for interpretability analysis.

DO NOT use this for training - use the original in models/ instead.

Original: models/vit.py
"""

import torch
from torch import nn
from torchvision.models import vision_transformer
from functools import partial

# ==============================================================================
# 1. Custom Encoder Block (This is our building block)
# ==============================================================================

class EncoderBlockWithAttention(vision_transformer.EncoderBlock):
    """
    An encoder block that is identical to the original, but its forward
    method is overridden to also return the attention weights.
    """
    def forward(self, input: torch.Tensor):
        # Standard ViT encoder block logic
        x = self.ln_1(input)
        # Get attention weights by setting need_weights=True
        x, attn_weights = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.dropout(x)
        x = x + input # First residual connection

        y = self.ln_2(x)
        y = self.mlp(y)
        output = x + y # Second residual connection
        
        # Return both the output and the captured attention weights
        return output, attn_weights

# ==============================================================================
# 2. Factory function (Creates the standard model)
# ==============================================================================

def create_vit_tiny_patch4_32(num_classes=10):
    """
    Creates an instance of a standard Vision Transformer, identical to the
    one used during training.
    """
    model = vision_transformer.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=192 * 4,
        num_classes=num_classes,
        representation_size=None,
    )
    return model

# ==============================================================================
# 3. The conversion function (This is the core of the final fix)
# ==============================================================================

def convert_vit_to_attention_model(trained_vit_model):
    """
    Takes a trained, standard ViT model and converts it into a new model
    that can return attention maps. It achieves this by manually replacing
    the encoder blocks and copying all trained weights.
    """
    # --- STEP A: Manually replace the encoder layers ---
    
    # Get configuration from the already trained model
    num_layers = len(trained_vit_model.encoder.layers)
    num_heads = trained_vit_model.encoder.layers[0].self_attention.num_heads
    hidden_dim = trained_vit_model.hidden_dim
    mlp_dim = trained_vit_model.encoder.layers[0].mlp[0].out_features
    
    # Create a new nn.Sequential module containing our custom blocks
    new_encoder_layers = nn.Sequential(
        *[EncoderBlockWithAttention(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=0.0,
            attention_dropout=0.0
        ) for _ in range(num_layers)]
    )
    
    # Replace the old encoder layers with our new ones
    trained_vit_model.encoder.layers = new_encoder_layers
    
    # --- STEP B: Manually replace the forward method ---
    
    def forward_with_attention(self, x: torch.Tensor, return_attention: bool = False):
        """A new forward method that will be attached to the model."""
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        
        attention_maps = []
        # Iterate through the NEWLY replaced encoder layers
        for block in self.encoder.layers:
            x, attn_weights = block(x) # Now each block returns weights
            if return_attention:
                attention_maps.append(attn_weights.detach())
        
        x = self.encoder.ln(x)
        cls_token_final = x[:, 0]
        logits = self.heads(cls_token_final)
        
        if return_attention:
            return logits, attention_maps
        return logits

    # This is a technique called "monkey-patching". We are dynamically
    # replacing the original forward method of the model instance.
    trained_vit_model.forward = partial(forward_with_attention, trained_vit_model)
    
    # Return the modified model
    return trained_vit_model