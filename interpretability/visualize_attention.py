# interpretability/visualize_attention.py

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Add project root to path to allow importing models ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# --- Import MODIFIED models from the local subfolder ---
from interpretability.models_for_viz.vision_bdh_v2 import VisionBDHv2, BDHConfig
from interpretability.models_for_viz.vit import create_vit_tiny_patch4_32

# ==============================================================================
# STEP 1: CONFIGURATION
# ==============================================================================
BDH_V2_CHECKPOINT_PATH = os.path.join(project_root, "checkpoints_v2_cifar10/final_model_best_v2.pth")
VIT_TINY_CHECKPOINT_PATH = os.path.join(project_root, "checkpoints_vit_tiny_cifar10/final_model.pth")

OUTPUT_DIR = os.path.join(project_root, "attention_maps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 4
IMG_SIZE = 32

# --- Select an image to analyze ---
CLASS_TO_VISUALIZE = "bird"
IMAGE_INDEX_IN_CLASS = 7 # Pick the 6th cat image (0-indexed)

# ==============================================================================
# STEP 2: LOAD MODELS AND DATA
# ==============================================================================

def load_models():
    """
    Loads the trained models. For ViT, it loads the standard model
    and then converts it to a version that can output attention maps.
    """
    print("Loading models...")
    
    def fix_compiled_keys(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[10:]
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    # --- Load VisionBDH v2 (remains the same) ---
    bdh_config = BDHConfig(n_layer=6, n_embd=192, n_head=6, mlp_internal_dim_multiplier=32)
    bdh_model = VisionBDHv2(bdh_config=bdh_config, num_classes=10)
    if not os.path.exists(BDH_V2_CHECKPOINT_PATH):
        raise FileNotFoundError(f"BDH model checkpoint not found at: {BDH_V2_CHECKPOINT_PATH}")
    bdh_state_dict = torch.load(BDH_V2_CHECKPOINT_PATH, map_location=DEVICE)
    bdh_state_dict_fixed = fix_compiled_keys(bdh_state_dict)
    bdh_model.load_state_dict(bdh_state_dict_fixed)
    bdh_model.to(DEVICE)
    bdh_model.eval()
    print("✓ Vision-BDH v2 loaded.")

    # --- Load and Convert ViT-Tiny ---
    # 1. Create the original model structure
    from models.vit import create_vit_tiny_patch4_32 as create_original_vit
    vit_model_original = create_original_vit(num_classes=10)
    
    if not os.path.exists(VIT_TINY_CHECKPOINT_PATH):
        raise FileNotFoundError(f"ViT-Tiny model checkpoint not found at: {VIT_TINY_CHECKPOINT_PATH}")
    
    # 2. Load the saved state_dict into the original structure
    vit_state_dict = torch.load(VIT_TINY_CHECKPOINT_PATH, map_location=DEVICE)
    vit_state_dict_fixed = fix_compiled_keys(vit_state_dict)
    vit_model_original.load_state_dict(vit_state_dict_fixed)
    
    # 3. Now, convert the trained model to our visualization-ready version
    from interpretability.models_for_viz.vit import convert_vit_to_attention_model
    vit_model_for_viz = convert_vit_to_attention_model(vit_model_original)
    
    vit_model_for_viz.to(DEVICE)
    vit_model_for_viz.eval()
    print("✓ ViT-Tiny loaded and converted for visualization.")
    
    return bdh_model, vit_model_for_viz

def get_image():
    """Loads the specified image from the CIFAR-10 test set."""
    print(f"\nLoading image {IMAGE_INDEX_IN_CLASS} of class '{CLASS_TO_VISUALIZE}'...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = CIFAR10(root=os.path.join(project_root, "data"), train=False, download=True, transform=transform)
    class_idx = test_dataset.class_to_idx[CLASS_TO_VISUALIZE]
    indices_of_class = [i for i, label in enumerate(test_dataset.targets) if label == class_idx]
    if IMAGE_INDEX_IN_CLASS >= len(indices_of_class):
        raise IndexError(f"Image index out of bounds for class '{CLASS_TO_VISUALIZE}'. Max is {len(indices_of_class) - 1}.")
    image_index = indices_of_class[IMAGE_INDEX_IN_CLASS]
    image_tensor, label = test_dataset[image_index]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    unnormalized_test_set = CIFAR10(root=os.path.join(project_root, "data"), train=False, download=True, transform=transforms.ToTensor())
    original_image, _ = unnormalized_test_set[image_index]
    
    print(f"✓ Image loaded. Shape for model: {image_tensor.shape}")
    return image_tensor, original_image, CLASS_TO_VISUALIZE

# ==============================================================================
# STEP 3: ATTENTION EXTRACTION AND VISUALIZATION
# ==============================================================================

def get_attention_maps(model, image_tensor):
    """Runs the model and returns the attention maps, averaged over heads."""
    model.eval()
    with torch.no_grad():
        logits, attention_maps = model(image_tensor, return_attention=True)
    
    # Stack list of tensors into a single tensor
    # Shape: (num_layers, batch_size, num_heads, num_patches, num_patches)
    attention_tensor = torch.stack(attention_maps)
    
    # Squeeze the batch dimension (since batch_size is 1)
    attention_tensor = attention_tensor.squeeze(1)
    
    # Average over the heads to get one map per layer
    # Shape: (num_layers, num_patches, num_patches)
    attention_tensor_avg_heads = attention_tensor.mean(dim=1)
    
    return attention_tensor_avg_heads.cpu().numpy()

def plot_attention_maps(attentions, original_image, model_name, class_name):
    """
    Plots the attention maps for selected layers (first, middle, last).
    The map shows the attention from the center patch to all other patches.
    """
    num_layers = attentions.shape[0]
    # Attention map shape is (num_patches, num_patches). num_patches = side * side
    num_patches_side = int(np.sqrt(attentions.shape[-1]))

    # Select layers to visualize
    layers_to_plot = [0, num_layers // 2, num_layers - 1]
    
    fig, axes = plt.subplots(1, len(layers_to_plot) + 1, figsize=(16, 4))
    fig.suptitle(f'Attention Maps for {model_name} on a "{class_name}" image', fontsize=16, fontweight='bold')
    
    # Plot original image
    axes[0].imshow(original_image.permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # We will visualize the attention FROM the center patch TO all other patches
    center_patch_idx = (num_patches_side * num_patches_side) // 2
    
    # Find global min and max for consistent color scaling
    vmin = min(attentions[l, center_patch_idx, :].min() for l in layers_to_plot)
    vmax = max(attentions[l, center_patch_idx, :].max() for l in layers_to_plot)

    for i, layer_idx in enumerate(layers_to_plot):
        # Extract attention from the center patch and reshape to 2D
        attn_map = attentions[layer_idx, center_patch_idx, :].reshape(num_patches_side, num_patches_side)
        
        ax = axes[i + 1]
        ax.set_title(f"Layer {layer_idx + 1}")
        
        # Plot the attention map as a heatmap
        im = ax.imshow(attn_map, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    
    save_path = os.path.join(OUTPUT_DIR, f"attention_{model_name.lower().replace(' ', '_')}_{class_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved attention map plot to {save_path}")
    # plt.show() # Uncomment to display plots interactively
    plt.close(fig)

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    bdh_model, vit_model = load_models()
    image_to_analyze, original_image_for_plot, class_name = get_image()

    print("\n-------------------------------------------")
    print("Extracting and visualizing attention maps...")
    print("-------------------------------------------")

    # --- Get and plot for BDH ---
    bdh_attentions = get_attention_maps(bdh_model, image_to_analyze)
    plot_attention_maps(bdh_attentions, original_image_for_plot, "Vision-BDH v2", class_name)

    # --- Get and plot for ViT ---
    vit_attentions = get_attention_maps(vit_model, image_to_analyze)
    # ViT has a CLS token at the beginning of the sequence, so we must remove it for visualization
    vit_attentions_no_cls = vit_attentions[:, 1:, 1:]
    plot_attention_maps(vit_attentions_no_cls, original_image_for_plot, "ViT-Tiny", class_name)

    print("\n-------------------------------------------")
    print("Visualization complete. Check the 'attention_maps' folder.")
    print("-------------------------------------------")