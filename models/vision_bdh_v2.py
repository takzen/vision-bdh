import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bdh import BDHConfig


class VisionBDHv2(nn.Module):
    """
    Vision-BDH v2 — incremental, stable, and slightly optimized version of VisionBDH.
    
    Core architecture identical to v1, but with:
    - [v2 change] Optional softmax attention for better numerical stability
    - [v2 change] Improved initialization (Xavier uniform)
    - [v2 change] Slightly adjusted LayerNorm placement
    - [v2 change] More transparent dropout control
    """

    def __init__(self, bdh_config, img_size=32, patch_size=4, num_classes=10, in_channels=3, use_softmax_attn=True):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_softmax_attn = use_softmax_attn  # [v2 change]

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

        # [v2 change] Xavier initialization for better gradient flow
        self.decoder = nn.Parameter(torch.empty((nh * N, D)))
        nn.init.xavier_uniform_(self.decoder)

        self.encoder = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder)

        self.encoder_v = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder_v)

        self.attn = BidirectionalAttentionV2(C, use_softmax=self.use_softmax_attn)  # [v2 change]
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)

    # ---------------------------------------------------------------------- #
    #                            Initialization
    # ---------------------------------------------------------------------- #

    def _init_weights(self, module):
        """[v2 change] More stable initialization for linear/conv layers."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------------- #
    #                            Forward Pass
    # ---------------------------------------------------------------------- #

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - batch of images
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # 1. Patch embedding -> flatten -> (B, num_patches, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed  # Add positional embedding
        x = x.unsqueeze(1)  # (B, 1, T, D)

        # 2. Main BDH recurrent loop
        for level in range(C.n_layer):
            x = self.ln(x)  # [v2 change] moved before encoder for stability
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            # Bidirectional attention (Q = K)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)  # Gated-like residual

        # 3. Pool + Head
        x = x.squeeze(1).mean(dim=1)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits


# ========================================================================== #
#                             Bidirectional Attention
# ========================================================================== #

class BidirectionalAttentionV2(nn.Module):
    """
    Bidirectional Attention (v2) for Vision-BDH.
    Identical to original but adds optional softmax normalization for stability.
    """

    def __init__(self, config, use_softmax=True):
        super().__init__()
        self.config = config
        self.use_softmax = use_softmax  # [v2 change]
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
        KR = QR  # Q=K constraint preserved
        scores = QR @ KR.mT  # (B, nh, T, T)

        if self.use_softmax:  # [v2 change]
            scores = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)

        return scores @ V
