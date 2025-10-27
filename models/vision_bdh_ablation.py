# Copyright 2024 Krzysztof Pika
# This file is part of the Vision-BDH project.
#
# Licensed under the MIT License.
#
# This code is an adaptation of the original Baby Dragon Hatchling (BDH)
# architecture for computer vision tasks. The core concepts are derived from
# the BDH model by Pathway Technology, Inc., which is also licensed under the
# MIT License and is subject to the following copyright notice:
#
# ----------------------------------------------------------------------
#
# Copyright 2025 Pathway Technology, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ----------------------------------------------------------------------

"""
Vision-BDH v2 with Ablation Support

This is a COPY of vision_bdh_v2.py extended with configurable 
normalization strategies for ablation studies.

WHY SEPARATE FILE:
- vision_bdh_v2.py: Frozen for reproducibility of published results
- vision_bdh_ablation.py: Active development for new experiments

DIFFERENCES FROM vision_bdh_v2.py:
- Added norm_style parameter (pre_ln, post_ln, double_ln)
- Separate forward methods for each strategy
- Type hints and comprehensive documentation

For production training: use vision_bdh_v2.py
For ablation studies: use this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bdh import BDHConfig
from typing import Literal


class VisionBDHv2(nn.Module):
    """
    Vision-BDH v2 — Enhanced version with improved stability and configurable ablations.
    
    Key improvements over v1:
    - Xavier uniform initialization (better gradient flow)
    - Optional softmax attention (numerical stability)
    - Pre-LayerNorm placement (training stability)
    - Configurable normalization style for ablation studies
    
    Args:
        bdh_config: BDH configuration object
        img_size: Input image size (default: 32)
        patch_size: Patch size for embedding (default: 4)
        num_classes: Number of output classes (default: 10)
        in_channels: Number of input channels (default: 3)
        use_softmax_attn: Whether to use softmax in attention (default: True)
        norm_style: LayerNorm placement strategy (default: 'pre_ln')
            - 'pre_ln': Pre-LayerNorm (recommended, more stable)
            - 'post_ln': Post-LayerNorm (original Transformer style)
            - 'double_ln': Double LayerNorm (normalize update + result)
    """

    def __init__(
        self, 
        bdh_config: BDHConfig, 
        img_size: int = 32, 
        patch_size: int = 4, 
        num_classes: int = 10, 
        in_channels: int = 3, 
        use_softmax_attn: bool = True,
        norm_style: Literal['pre_ln', 'post_ln', 'double_ln'] = 'pre_ln'
    ):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_softmax_attn = use_softmax_attn
        self.norm_style = norm_style

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels,
            bdh_config.n_embd,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Positional embedding for patch tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, bdh_config.n_embd) * 0.02
        )

        # BDH layers (core recurrent module)
        self.build_bdh_layers()

        # Classification head
        self.ln_final = nn.LayerNorm(bdh_config.n_embd, elementwise_affine=False, bias=False)
        self.head = nn.Linear(bdh_config.n_embd, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    # ---------------------------------------------------------------------- #
    #                            Model Construction
    # ---------------------------------------------------------------------- #

    def build_bdh_layers(self):
        """Builds BDH layers with modifications for visual bidirectional attention."""
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = C.mlp_internal_dim_multiplier * D // nh

        # Xavier initialization for better gradient flow
        self.decoder = nn.Parameter(torch.empty((nh * N, D)))
        nn.init.xavier_uniform_(self.decoder)

        self.encoder = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder)

        self.encoder_v = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder_v)

        self.attn = BidirectionalAttentionV2(C, use_softmax=self.use_softmax_attn)
        
        # LayerNorms - create based on norm_style
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        if self.norm_style == 'double_ln':
            # Separate LayerNorm for update (used in double_ln mode)
            self.ln_update = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        self.drop = nn.Dropout(C.dropout)

    # ---------------------------------------------------------------------- #
    #                            Initialization
    # ---------------------------------------------------------------------- #

    def _init_weights(self, module):
        """Stable initialization for linear/conv layers."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------------- #
    #                            Forward Pass
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Vision-BDH v2.
        
        Args:
            x: Input images, shape (B, C, H, W)
            
        Returns:
            logits: Class predictions, shape (B, num_classes)
        """
        B = x.shape[0]
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # 1. Patch embedding -> flatten -> (B, num_patches, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = x.unsqueeze(1)  # (B, 1, T, D)

        # 2. Main BDH recurrent loop
        for level in range(C.n_layer):
            x = self._forward_bdh_block(x, B, D, nh, N)

        # 3. Pool + Head
        x = x.squeeze(1).mean(dim=1)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits

    def _forward_bdh_block(
        self, 
        x: torch.Tensor, 
        B: int, 
        D: int, 
        nh: int, 
        N: int
    ) -> torch.Tensor:
        """
        Single BDH block forward pass with configurable normalization.
        
        Supports three normalization strategies:
        
        1. pre_ln (recommended, default):
           - Normalize input before processing
           - More stable gradients
           - Used in modern Transformers (GPT-2+)
           
        2. post_ln (original Transformer):
           - Normalize after residual connection
           - Original "Attention is All You Need" style
           
        3. double_ln (experimental):
           - Normalize both update and result
           - Extra computational cost
           - May provide additional stability
        """
        if self.norm_style == 'pre_ln':
            return self._forward_pre_ln(x, B, D, nh, N)
        elif self.norm_style == 'post_ln':
            return self._forward_post_ln(x, B, D, nh, N)
        elif self.norm_style == 'double_ln':
            return self._forward_double_ln(x, B, D, nh, N)
        else:
            raise ValueError(f"Unknown norm_style: {self.norm_style}")

    def _forward_pre_ln(
        self, 
        x: torch.Tensor, 
        B: int, 
        D: int, 
        nh: int, 
        N: int
    ) -> torch.Tensor:
        """
        Pre-LayerNorm forward pass (recommended).
        
        Flow: LN -> Encoder -> Attention -> Encoder_v -> MLP -> Residual
        
        Benefits:
        - More stable training (gradient flow)
        - Widely used in modern Transformers
        - Default for v2
        """
        # Normalize before processing
        x_norm = self.ln(x)
        
        # Sparse projection
        x_latent = x_norm @ self.encoder
        x_sparse = F.relu(x_latent)

        # Bidirectional attention (Q = K)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x_norm)
        yKV = self.ln(yKV)

        # Value projection and gating
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = self.drop(xy_sparse)

        # MLP projection
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
        
        # Residual connection (no additional LN)
        x = x + yMLP
        
        return x

    def _forward_post_ln(
        self, 
        x: torch.Tensor, 
        B: int, 
        D: int, 
        nh: int, 
        N: int
    ) -> torch.Tensor:
        """
        Post-LayerNorm forward pass (original Transformer style).
        
        Flow: Encoder -> Attention -> Encoder_v -> MLP -> Residual -> LN
        
        Used in original "Attention is All You Need" paper.
        May be less stable for deep networks.
        """
        # Process without initial normalization
        x_latent = x @ self.encoder
        x_sparse = F.relu(x_latent)

        # Bidirectional attention (Q = K)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
        
        # Value projection and gating
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = self.drop(xy_sparse)

        # MLP projection
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
        
        # Residual connection + normalization
        x = self.ln(x + yMLP)
        
        return x

    def _forward_double_ln(
        self, 
        x: torch.Tensor, 
        B: int, 
        D: int, 
        nh: int, 
        N: int
    ) -> torch.Tensor:
        """
        Double LayerNorm forward pass (experimental).
        
        Flow: Encoder -> Attention -> Encoder_v -> MLP -> LN(update) -> Residual -> LN
        
        Normalizes both the update and the result.
        Higher computational cost, may provide extra stability.
        """
        x_latent = x @ self.encoder
        x_sparse = F.relu(x_latent)

        # Bidirectional attention (Q = K)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
        
        # Value projection and gating
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = self.drop(xy_sparse)

        # MLP projection
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
        
        # Normalize the update
        y_normalized = self.ln_update(yMLP)
        
        # Residual connection + normalization
        x = self.ln(x + y_normalized)
        
        return x


# ========================================================================== #
#                             Bidirectional Attention
# ========================================================================== #

class BidirectionalAttentionV2(nn.Module):
    """
    Bidirectional Attention for Vision-BDH.
    
    Key features:
    - Q=K constraint (activation similarity)
    - RoPE positional encoding
    - Optional softmax normalization
    """

    def __init__(self, config: BDHConfig, use_softmax: bool = True):
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
    def phases_cos_sin(phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert phases to cosine and sine for RoPE."""
        phases = (phases % 1) * (2 * torch.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding (RoPE)."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos, sin = BidirectionalAttentionV2.phases_cos_sin(phases)
        return (v * cos) + (v_rot * sin)

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Bidirectional attention forward pass.
        
        Args:
            Q: Query tensor (B, nh, T, N)
            K: Key tensor (must equal Q due to Q=K constraint)
            V: Value tensor (B, 1, T, D)
            
        Returns:
            Attended output (B, 1, T, D)
        """
        assert K is Q, "In Vision-BDH, K must equal Q (Q=K constraint)"
        _, _, T, _ = Q.size()

        # Generate position-dependent phases for RoPE
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs

        # Apply RoPE to Q and K
        QR = self.rope(r_phases, Q)
        KR = QR  # Q=K constraint preserved
        
        # Compute attention scores
        scores = QR @ KR.mT  # (B, nh, T, T)

        if self.use_softmax:
            # Scaled softmax attention (standard Transformer)
            scores = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)

        return scores @ V