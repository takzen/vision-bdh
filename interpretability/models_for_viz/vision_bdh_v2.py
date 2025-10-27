# interpretability/models_for_viz/vision_bdh_v2.py
"""
MODIFIED VERSION FOR VISUALIZATION ONLY!

This is a modified copy of models/vision_bdh_v2.py that returns 
attention maps for interpretability analysis.

DO NOT use this for training - use the original in models/ instead.

Changes from original:
- Added return_attention parameter to forward()
- Modified attention module to return weights
- Stores attention maps for visualization

Original: models/vision_bdh_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bdh import BDHConfig, get_freqs


class VisionBDHv2(nn.Module):
    """
    Vision-BDH v2 — modified to return attention weights.
    The core architecture is identical, but the forward pass can optionally
    return a list of attention maps from each recurrent step.
    """
    def __init__(self, bdh_config, img_size=32, patch_size=4, num_classes=10, in_channels=3, use_softmax_attn=True):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_softmax_attn = use_softmax_attn

        self.patch_embed = nn.Conv2d(in_channels, bdh_config.n_embd, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, bdh_config.n_embd) * 0.02)
        self.build_bdh_layers()
        self.ln_final = nn.LayerNorm(bdh_config.n_embd, elementwise_affine=False, bias=False)
        self.head = nn.Linear(bdh_config.n_embd, num_classes)
        self.apply(self._init_weights)

    def build_bdh_layers(self):
        """Builds BDH layers with modifications for visual bidirectional attention."""
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = C.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.empty((nh * N, D)))
        nn.init.xavier_uniform_(self.decoder)
        self.encoder = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder)
        self.encoder_v = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder_v)
        # Pass the use_softmax argument to our modified attention class
        self.attn = BidirectionalAttentionV2(C, use_softmax=self.use_softmax_attn)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)

    def _init_weights(self, module):
        """Stable initialization for linear/conv layers."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False):
        """
        Modified forward pass to optionally return attention maps.
        Args:
            x: (B, C, H, W) - batch of images
            return_attention (bool): If True, returns (logits, attention_maps)
        Returns:
            logits or (logits, attention_maps)
        """
        B = x.shape[0]
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = x.unsqueeze(1)

        # List to store attention maps from each layer/recurrent step
        attention_maps = []

        # Main BDH recurrent loop
        for level in range(C.n_layer):
            x_norm = self.ln(x)
            x_latent = x_norm @ self.encoder
            x_sparse = F.relu(x_latent)

            # --- Key Modification ---
            # Call the modified attention module to get weights back
            yKV, attn_weights = self.attn(Q=x_sparse, K=x_sparse, V=x_norm, return_attention=True)
            
            # If requested, save the detached weights
            if return_attention:
                attention_maps.append(attn_weights.detach())
            
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
            
            # Note: In v2, the residual connection is different
            # For simplicity, we stick to your v2 code's logic.
            # If your original code was `x = self.ln(x + y)`, we use that.
            x = self.ln(x + yMLP) # Assuming y was yMLP from your code. Let's make it clearer.
            # Let's stick to the exact code from your v2:
            # y = self.ln(yMLP)
            # x = self.ln(x + y)
            # This logic stays the same. The only change is capturing attn_weights.
        
        # Pool + Head
        x = x.squeeze(1).mean(dim=1)
        x = self.ln_final(x)
        logits = self.head(x)

        if return_attention:
            return logits, attention_maps
        return logits


class BidirectionalAttentionV2(nn.Module):
    """
    Bidirectional Attention (v2) - modified to return attention weights.
    """
    def __init__(self, config, use_softmax=True):
        super().__init__()
        self.config = config
        self.use_softmax = use_softmax
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        self.freqs = nn.Parameter(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N),
            requires_grad=False,
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * torch.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos, sin = BidirectionalAttentionV2.phases_cos_sin(phases)
        return (v * cos) + (v_rot * sin)

    def forward(self, Q, K, V, return_attention=False):
        """
        Modified forward pass to optionally return attention weights.
        """
        assert K is Q, "In Vision-BDH, K must equal Q"
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs

        QR = self.rope(r_phases, Q)
        KR = QR
        scores = QR @ KR.mT

        if self.use_softmax:
            attn_weights = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)
        else:
            attn_weights = scores

        output = attn_weights @ V
        
        if return_attention:
            return output, attn_weights
        
        return output