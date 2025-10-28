# models/vision_bdh_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bdh import BDHConfig
import math


class ScaledLayerNorm(nn.Module):
    """
    A LayerNorm module that applies a fixed scaling factor after normalization.
    The scaling factor is calculated as 1/sqrt(depth_idx) to stabilize deep networks.
    """
    def __init__(self, normalized_shape, depth_idx: int):
        super().__init__()
        # Use standard LayerNorm with learnable affine parameters for flexibility
        self.ln = nn.LayerNorm(normalized_shape) 
        self.scale = 1.0 / math.sqrt(depth_idx)

    def forward(self, x):
        return self.ln(x) * self.scale


class VisionBDHv3(nn.Module):
    """
    Vision-BDH v3 — The version incorporating community-driven improvements for SOTA performance.

    Key features:
    - ScaledLayerNorm used per recurrent depth for better gradient scaling.
    - Final LayerNorm before the classification head is removed.
    - Softmax in the attention mechanism is disabled by default for linear attention.
    """
    def __init__(
        self,
        bdh_config,
        img_size=32,
        patch_size=4,
        num_classes=10,
        in_channels=3,
        use_softmax_attn=False, # Default is False based on best results
    ):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_softmax_attn = use_softmax_attn

        self.patch_embed = nn.Conv2d(in_channels, bdh_config.n_embd, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, bdh_config.n_embd) * 0.02)
        self.build_bdh_layers()

        self.ln_layers = nn.ModuleList([
            ScaledLayerNorm(bdh_config.n_embd, depth_idx=i + 1)
            for i in range(bdh_config.n_layer)
        ])
        
        # This final LN is applied only once after the loop
        self.ln_final = ScaledLayerNorm(bdh_config.n_embd, depth_idx=bdh_config.n_layer + 1)
        
        self.head = nn.Linear(bdh_config.n_embd, num_classes)
        self.apply(self._init_weights)

    def build_bdh_layers(self):
        """Builds the core BDH parameters."""
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
        self.attn = BidirectionalAttentionV2(C, use_softmax=self.use_softmax_attn)
        self.drop = nn.Dropout(C.dropout)

    def _init_weights(self, module):
        """Applies stable Xavier initialization."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        B = x.shape[0]
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # 1. Patch + Positional Embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = x.unsqueeze(1)

        # 2. BDH recurrent loop with depth-specific ScaledLayerNorm
        for level in range(C.n_layer):
            ln = self.ln_layers[level]
            
            # --- CLEANED UP FORWARD LOGIC (Pre-LN style) ---
            # Store original x for residual connection
            residual = x
            
            # Normalize input
            x_norm = ln(x)
            
            # Main block logic
            x_latent = x_norm @ self.encoder
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x_norm)
            
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder

            # Add the result to the original input (residual connection)
            x = residual + yMLP
            # --- END OF CLEANED UP LOGIC ---

        # 3. Final normalization, Pooling + Classification Head
        x = self.ln_final(x)
        x = x.squeeze(1).mean(dim=1)
        logits = self.head(x)
        return logits


class BidirectionalAttentionV2(nn.Module):
    # ... (ta klasa pozostaje bez zmian) ...
    def __init__(self, config, use_softmax=True):
        super().__init__()
        self.config = config
        self.use_softmax = use_softmax
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        from models.bdh import get_freqs
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

    def forward(self, Q, K, V):
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
            scores = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)
        return scores @ V