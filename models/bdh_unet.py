"""
BDH-UNet: Semantic Segmentation with Baby Dragon Hatchling blocks (STABLE VERSION)

CRITICAL FIXES:
- Replaced TransposedConv2d with Upsample + Conv2d (numerical stability!)
- Xavier initialization for decoder
- Proper bias settings

Architecture Philosophy:
- Encoder: BDH blocks (sparse activations, Q=K attention)
- Decoder: STABLE upsampling (Upsample + Conv, not TransposedConv!)
- Skip connections: Concatenation (U-Net style)
- Pre-LN + raw attention (proven best combo!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryPositionalEmbedding(nn.Module):
    """RoPE for 2D spatial positions (H×W patches)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cat([emb.cos(), emb.sin()], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([
        x1 * cos[..., ::2] - x2 * sin[..., ::2],
        x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
    ], dim=-1)


class BDHBlock(nn.Module):
    """
    Single BDH layer with Pre-LayerNorm and raw attention (no softmax)
    
    Key features from Vision-BDH v2 optimized:
    - Sparse activations (ReLU)
    - Q=K constraint
    - Pre-LayerNorm (stability)
    - Raw attention scores (no softmax)
    - Multiplicative gating
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.ln = nn.LayerNorm(dim)
        self.encoder = nn.Linear(dim, dim)
        self.qk_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        
        # Xavier initialization (stable training)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.qk_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.gate.weight)
    
    def forward(self, x: torch.Tensor, rope_emb: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        x_norm = self.ln(x)
        x_latent = F.relu(self.encoder(x_norm))
        
        qk = self.qk_proj(x_latent).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x_latent).view(B, N, self.num_heads, self.head_dim)
        
        cos = rope_emb[:, :self.head_dim].cos().unsqueeze(0).unsqueeze(2)
        sin = rope_emb[:, :self.head_dim].sin().unsqueeze(0).unsqueeze(2)
        qk = apply_rotary_pos_emb(qk, cos, sin)
        
        qk = qk.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_scores = torch.matmul(qk, qk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_out = torch.matmul(attn_scores, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)
        
        gate_values = torch.sigmoid(self.gate(x_norm))
        y = gate_values * attn_out
        
        return x + y


class BDHEncoder(nn.Module):
    """
    BDH Encoder: Processes image patches with recurrent BDH blocks
    """
    def __init__(self, in_channels: int = 3, base_dim: int = 64, 
                 depths: list = [2, 2, 4, 2], num_heads: int = 4):
        super().__init__()
        self.depths = depths
        
        self.patch_embed = nn.Conv2d(in_channels, base_dim, kernel_size=4, stride=4)
        
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.rope_layers = nn.ModuleList()
        
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]
        
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            self.rope_layers.append(RotaryPositionalEmbedding(dim))
            
            stage = nn.ModuleList([
                BDHBlock(dim, num_heads) for _ in range(depth)
            ])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(dim, dims[i+1], kernel_size=2, stride=2),
                    nn.BatchNorm2d(dims[i+1])
                )
                self.downsample_layers.append(downsample)
    
    def forward(self, x: torch.Tensor) -> list:
        features = []
        
        x = self.patch_embed(x)
        
        for i, stage in enumerate(self.stages):
            B, C, H, W = x.shape
            
            x_flat = x.flatten(2).transpose(1, 2)
            rope_emb = self.rope_layers[i](H * W)
            
            for block in stage:
                x_flat = block(x_flat, rope_emb)
            
            x = x_flat.transpose(1, 2).view(B, C, H, W)
            features.append(x)
            
            if i < len(self.stages) - 1:
                x = self.downsample_layers[i](x)
        
        return features


class DecoderBlock(nn.Module):
    """
    STABLE Decoder Block using Upsample + Conv (not TransposedConv2d!)
    
    This is critical for numerical stability!
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # STABLE upsampling (no TransposedConv2d!)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Refinement after skip connection
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Ensure spatial dimensions match
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        
        return x


class BDHUNet(nn.Module):
    """
    BDH-UNet: Semantic Segmentation with BDH encoder (STABLE VERSION)
    
    Architecture:
    - Encoder: BDH blocks (sparse activations, Q=K attention)
    - Decoder: STABLE upsampling (Upsample + Conv, not TransposedConv!)
    - Skip connections: U-Net style concatenation
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 21, 
                 base_dim: int = 64, depths: list = [2, 2, 4, 2]):
        super().__init__()
        
        # BDH Encoder
        self.encoder = BDHEncoder(in_channels, base_dim, depths)
        
        # Decoder stages (STABLE version)
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]
        
        self.decoder_stages = nn.ModuleList()
        for i in range(len(dims)-1, 0, -1):
            self.decoder_stages.append(
                DecoderBlock(dims[i], dims[i-1])
            )
        
        # Final upsampling to original resolution (×4) - STABLE!
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        
        # Segmentation head
        self.seg_head = nn.Conv2d(base_dim, num_classes, kernel_size=1)
        
        # Initialize decoder weights
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        """Initialize decoder weights with Xavier (more stable than Kaiming)"""
        for name, m in self.named_modules():
            if any(x in name for x in ['decoder', 'final_upsample', 'seg_head']):
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            seg_map: [B, num_classes, H, W] segmentation logits
        """
        # Encoder with skip connections
        enc_features = self.encoder(x)
        
        # Decoder with skip connections
        x = enc_features[-1]
        
        for i, decoder_block in enumerate(self.decoder_stages):
            skip_idx = len(enc_features) - 2 - i
            skip = enc_features[skip_idx]
            x = decoder_block(x, skip)
        
        # Final upsampling to input resolution
        x = self.final_upsample(x)
        
        # Segmentation head
        seg_map = self.seg_head(x)
        
        return seg_map


# Quick test
if __name__ == "__main__":
    model = BDHUNet(in_channels=3, num_classes=21, base_dim=64)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✅ BDH-UNet STABLE ready for experiments!")