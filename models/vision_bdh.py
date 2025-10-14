import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.bdh import BDHConfig


class VisionBDH(nn.Module):
    """
    An adapter to convert the language-oriented BDH model into a Vision Transformer.
    
    Key modifications:
    1. Patch embedding instead of token embedding.
    2. Bidirectional attention instead of causal attention.
    3. Classification head instead of a language modeling head.
    """
    
    def __init__(self, bdh_config, img_size=32, patch_size=4, num_classes=10, in_channels=3):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: A Conv2d layer acts as a linear projection for each patch.
        self.patch_embed = nn.Conv2d(
            in_channels, 
            bdh_config.n_embd, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding for patches (an additional embedding, alongside BDH's RoPE).
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, bdh_config.n_embd) * 0.02
        )
        
        # Build the BDH layers manually (without the original embedding and lm_head).
        self.build_bdh_layers()
        
        # Classification head.
        self.ln_final = nn.LayerNorm(bdh_config.n_embd, elementwise_affine=False, bias=False)
        self.head = nn.Linear(bdh_config.n_embd, num_classes)
        
        self.apply(self._init_weights)
    
    def build_bdh_layers(self):
        """Builds the core BDH parameters with a modification for bidirectional attention."""
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = C.mlp_internal_dim_multiplier * D // nh
        
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        
        # Attention module with modifications for vision tasks.
        self.attn = BidirectionalAttention(C)
        
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - a batch of images.
        Returns:
            logits: (B, num_classes).
        """
        B = x.shape[0]
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        # 1. Patch embedding: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, num_patches) -> (B, num_patches, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. Add positional embedding.
        x = x + self.pos_embed
        
        # 3. Reshape for BDH core: (B, num_patches, D) -> (B, 1, num_patches, D)
        x = x.unsqueeze(1)
        x = self.ln(x)
        
        # 4. BDH layers loop (n_layer times).
        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # (B, nh, T, N)
            
            # Bidirectional attention!
            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)
            
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        # 5. Global average pooling + classification.
        x = x.squeeze(1)      # (B, num_patches, D)
        x = x.mean(dim=1)     # (B, D) - average across all patches
        x = self.ln_final(x)
        logits = self.head(x) # (B, num_classes)
        
        return logits


class BidirectionalAttention(nn.Module):
    """
    A modification of the BDH Attention module: removes causal masking (.tril)
    to enable full bidirectional attention for image patches.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Copied from the original BDH.Attention
        from models.bdh import get_freqs
        self.freqs = nn.Parameter(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N),
            requires_grad=False
        )
    
    @staticmethod
    def phases_cos_sin(phases):
        import math
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin
    
    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = BidirectionalAttention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
    
    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, _ = Q.size()
        
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        QR = self.rope(r_phases, Q)
        KR = QR
        
        # KEY CHANGE: Full attention matrix instead of .tril(diagonal=-1)
        scores = QR @ KR.mT  # (B, nh, T, T) - all patches can see all other patches
        
        # Optional: add softmax for stability (not present in original BDH)
        # scores = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)
        
        return scores @ V