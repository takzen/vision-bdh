# models/vision_bdh.py

import torch
from torch import nn
import torch.nn.functional as F

# Importujemy oryginalny model BDH i jego konfigurację
from .bdh import BDH, BDHConfig

class VisionBDH(nn.Module):
    """
    Vision Transformer (ViT) wrapper for the Baby Dragon Hatchling (BDH) model.
    This class handles patching, embedding, and adapting the BDH core for image classification.
    """
    def __init__(self, bdh_config: BDHConfig, img_size=224, patch_size=16, in_channels=3, num_classes=1000):
        super().__init__()
        self.bdh_config = bdh_config
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Sprawdzamy, czy wymiary się zgadzają
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (img_size // patch_size) ** 2
        
        # --- Krok 1: "Podwozie" ViT ---
        
        # Warstwa, która zamienia obraz na spłaszczone łatki i osadza je w przestrzeni D
        # To jest wydajny sposób na zrobienie "patching + linear projection"
        self.patch_projection = nn.Conv2d(
            in_channels, bdh_config.n_embd, kernel_size=patch_size, stride=patch_size
        )
        
        # Specjalny, uczący się token [CLS], który zbierze informacje o całym obrazie
        self.cls_token = nn.Parameter(torch.randn(1, 1, bdh_config.n_embd))
        
        # Uczące się osadzenia pozycyjne dla każdej łatki + tokenu [CLS]
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, bdh_config.n_embd))

        # --- Krok 2: "Silnik" BDH ---
        # Używamy oryginalnego modelu BDH jako rdzenia
        self.bdh_core = BDH(bdh_config)
        
        # --- Krok 3: "Deska rozdzielcza" ---
        # Głowica klasyfikacyjna, która patrzy tylko na wyjście z tokenu [CLS]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(bdh_config.n_embd),
            nn.Linear(bdh_config.n_embd, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przejście w przód dla VisionBDH.
        Args:
            x: Tensor obrazu wejściowego o kształcie (B, C, H, W).
        Returns:
            Tensor logitów o kształcie (B, num_classes).
        """
        B = x.shape[0] # Rozmiar batcha

        # 1. Tworzenie sekwencji łatek (patch embeddings)
        x = self.patch_projection(x)  # Kształt: (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # Kształt: (B, num_patches, D)

        # 2. Dodanie tokenu [CLS] na początek sekwencji
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Kształt: (B, num_patches + 1, D)

        # 3. Dodanie osadzeń pozycyjnych
        x = x + self.positional_embedding

        # 4. Przetwarzanie sekwencji przez rdzeń BDH
        # Wywołujemy zmodyfikowane przejście w przód, omijając embeddingi BDH
        processed_sequence = self.forward_features(x)

        # 5. Izolacja tokenu [CLS] i klasyfikacja
        # Bierzemy wyjście tylko dla pierwszego tokenu ([CLS])
        cls_output = processed_sequence[:, 0]
        logits = self.mlp_head(cls_output)
        
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ta funkcja replikuje pętlę z oryginalnego BDH.forward(), ale operuje
        na gotowych embeddingach, a nie na indeksach tokenów.
        
        Args:
            x: Tensor osadzonych łatek (patch embeddings) o kształcie (B, T, D).
        """
        # Musimy dodać wymiar dla "głowic", którego oczekuje BDH
        x = x.unsqueeze(1) # Kształt: (B, 1, T, D)

        # Normalizacja warstwowa na wejściu, tak jak w oryginale
        x = self.bdh_core.ln(x)

        # Główna pętla z oryginalnego modelu BDH
        for _ in range(self.bdh_config.n_layer):
            x_latent = x @ self.bdh_core.encoder
            x_sparse = F.relu(x_latent)

            yKV = self.bdh_core.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.bdh_core.ln(yKV)

            y_latent = yKV @ self.bdh_core.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse

            xy_sparse = self.bdh_core.drop(xy_sparse)
            
            # Zmieniamy kształt do projekcji
            B, _, T, _ = xy_sparse.shape
            nh = self.bdh_config.n_head
            N = self.bdh_config.mlp_internal_dim_multiplier * self.bdh_config.n_embd // nh
            
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.bdh_core.decoder
            )
            y = self.bdh_core.ln(yMLP)
            x = self.bdh_core.ln(x + y)
        
        # Usuwamy dodatkowy wymiar "głowicy"
        x = x.squeeze(1) # Kształt: (B, T, D)
        return x