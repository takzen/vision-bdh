# analyze.py

import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# STEP 1: GATHER YOUR DATA
# The results for VisionBDH are filled in from your training logs.
# ==============================================================================
results = {
    'Vision-BDH': {
        'val_accuracy_per_epoch': [
            35.61,  # Epoch 1
            50.81,  # Epoch 2
            56.13,  # Epoch 3
            59.85,  # Epoch 4
            65.67,  # Epoch 5
            66.33,  # Epoch 6
            69.74,  # Epoch 7
            71.83,  # Epoch 8
            72.66,  # Epoch 9
            73.13   # Epoch 10
        ],
        'final_test_accuracy': 72.51,
        'color': 'blue'
    },
    'ViT-Tiny': {
        # === NOTE: THE FOLLOWING ARE REALISTIC PLACEHOLDER DATA ===
        # === REPLACE WITH YOUR ACTUAL ViT-Tiny RESULTS AFTER TRAINING ===
        'val_accuracy_per_epoch': [
            30.15, 45.21, 51.55, 55.12, 58.34, 60.21, 61.88, 63.05, 64.73, 66.12
        ],
        'final_test_accuracy': 65.96,
        'color': 'orange'
    }
}

# --- Define the directory for saving images ---
IMAGE_DIR = "./images"
os.makedirs(IMAGE_DIR, exist_ok=True) # Creates the directory if it doesn't exist

def plot_learning_curves(results_data):
    """
    Creates and saves a plot of the learning curves (Validation Accuracy vs. Epoch).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in results_data.items():
        num_epochs = len(data['val_accuracy_per_epoch'])
        epochs = np.arange(1, num_epochs + 1)
        ax.plot(epochs, data['val_accuracy_per_epoch'], 
                marker='o', 
                linestyle='-', 
                label=model_name,
                color=data['color'])

    ax.set_title('Learning Curves: Validation Accuracy vs. Epoch', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    
    # Adjust X-axis ticks to match the number of epochs
    max_epochs = max(len(d['val_accuracy_per_epoch']) for d in results_data.values())
    ax.set_xticks(np.arange(1, max_epochs + 1))
    ax.set_ylim(bottom=0, top=100)

    plt.tight_layout()
    
    # --- Save the plot to the 'images' directory ---
    save_path = os.path.join(IMAGE_DIR, 'learning_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"✓ Learning curves plot saved to '{save_path}'")
    plt.show()


def plot_final_accuracy_comparison(results_data):
    """
    Creates and saves a bar chart comparing the final test accuracies.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    model_names = list(results_data.keys())
    final_accuracies = [data['final_test_accuracy'] for data in results_data.values()]
    colors = [data['color'] for data in results_data.values()]

    bars = ax.bar(model_names, final_accuracies, color=colors)

    ax.set_title('Final Test Accuracy Comparison on CIFAR-10', fontsize=16)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(bottom=0, top=max(final_accuracies) + 10) # Dynamic Y-axis

    # Add accuracy labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, 
                f'{yval:.2f}%', 
                va='bottom', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # --- Save the plot to the 'images' directory ---
    save_path = os.path.join(IMAGE_DIR, 'final_accuracy_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"✓ Comparison bar chart saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    print("--- Generating result visualizations ---")
    plot_learning_curves(results)
    print("-" * 40)
    plot_final_accuracy_comparison(results)
    print(f"\nDone! You can now find your plots in the '{IMAGE_DIR}' folder.")