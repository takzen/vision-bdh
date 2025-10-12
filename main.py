# main.py

import torch
from models.bdh import BDHConfig
from models.vision_bdh import VisionBDH

def test_vision_bdh():
    """
    Funkcja testująca, która tworzy model VisionBDH i przepuszcza przez niego
    sztuczny obraz, aby sprawdzić, czy wymiary się zgadzają.
    """
    print("--- Testing VisionBDH ---")

    # 1. Stwórz konfigurację dla rdzenia BDH
    # Użyjemy małej konfiguracji do testów
    config = BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        vocab_size=256 # Ta wartość jest ignorowana, ale wymagana przez BDHConfig
    )
    print(f"Using BDH config: n_layer={config.n_layer}, n_embd={config.n_embd}")

    # 2. Stwórz instancję naszego modelu VisionBDH
    # Przekazujemy config BDH oraz parametry obrazu
    model = VisionBDH(
        bdh_config=config,
        img_size=224,
        patch_size=16,
        num_classes=10 # np. dla zbioru CIFAR-10
    )
    print(f"VisionBDH model created successfully.")
    
    # Obliczmy parametry
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")

    # 3. Stwórz sztuczny obraz wejściowy
    # Kształt: (batch_size, kanały, wysokość, szerokość)
    dummy_image = torch.randn(2, 3, 224, 224)
    print(f"Created a dummy image tensor with shape: {dummy_image.shape}")

    # 4. Przepuść obraz przez model
    print("Performing a forward pass...")
    try:
        logits = model(dummy_image)
        print("Forward pass successful!")
        # Sprawdź kształt wyjściowy
        print(f"Output logits shape: {logits.shape}")
        assert logits.shape == (2, 10)
        print("Output shape is correct.")
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")

if __name__ == "__main__":
    test_vision_bdh()