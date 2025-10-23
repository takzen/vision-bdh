# analyze.py

import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# STEP 1: GATHER YOUR DATA
# Results from all three models: Original BDH, Optimized BDH, and ViT-Tiny
# ==============================================================================
results = {
    'Vision-BDH (Original)': {
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
        'params_millions': 6.5,
        'epoch_time_seconds': 7500,
        'color': '#1f77b4',  # Blue
        'linestyle': '--',
        'marker': 's'
    },
    'Vision-BDH (Optimized)': {
        'val_accuracy_per_epoch': [
            38.34,  # Epoch 1
            49.73,  # Epoch 2
            56.03,  # Epoch 3
            61.79,  # Epoch 4
            63.42,  # Epoch 5
            66.85,  # Epoch 6 (interpolated)
            68.65,  # Epoch 7
            70.50,  # Epoch 8 (interpolated)
            72.68,  # Epoch 9
            72.68   # Epoch 10 (same as 9, or update if you have data)
        ],
        'final_test_accuracy': 72.68,
        'params_millions': 4.2,
        'epoch_time_seconds': 50,
        'color': '#2ca02c',  # Green
        'linestyle': '-',
        'marker': 'o'
    },
    'ViT-Tiny': {
        'val_accuracy_per_epoch': [
            30.15,  # Epoch 1
            45.21,  # Epoch 2
            51.55,  # Epoch 3
            55.12,  # Epoch 4
            58.34,  # Epoch 5
            60.21,  # Epoch 6
            61.88,  # Epoch 7
            63.05,  # Epoch 8
            64.73,  # Epoch 9
            65.38   # Epoch 10
        ],
        'final_test_accuracy': 65.96,
        'params_millions': 5.7,
        'epoch_time_seconds': 45,
        'color': '#ff7f0e',  # Orange
        'linestyle': '-',
        'marker': '^'
    }
}

# --- Define the directory for saving images ---
IMAGE_DIR = "./images"
os.makedirs(IMAGE_DIR, exist_ok=True)


def plot_learning_curves(results_data):
    """
    Creates and saves a plot of the learning curves (Validation Accuracy vs. Epoch).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name, data in results_data.items():
        num_epochs = len(data['val_accuracy_per_epoch'])
        epochs = np.arange(1, num_epochs + 1)
        ax.plot(epochs, data['val_accuracy_per_epoch'], 
                marker=data.get('marker', 'o'),
                linestyle=data.get('linestyle', '-'),
                linewidth=2.5,
                markersize=8,
                label=model_name,
                color=data['color'])

    ax.set_title('Learning Curves: Validation Accuracy vs. Epoch', fontsize=18, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Adjust X-axis ticks
    max_epochs = max(len(d['val_accuracy_per_epoch']) for d in results_data.values())
    ax.set_xticks(np.arange(1, max_epochs + 1))
    ax.set_ylim(bottom=25, top=80)

    plt.tight_layout()
    
    save_path = os.path.join(IMAGE_DIR, 'learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curves plot saved to '{save_path}'")
    plt.show()


def plot_final_accuracy_comparison(results_data):
    """
    Creates and saves a bar chart comparing the final test accuracies.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    model_names = list(results_data.keys())
    final_accuracies = [data['final_test_accuracy'] for data in results_data.values()]
    colors = [data['color'] for data in results_data.values()]

    bars = ax.bar(model_names, final_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_title('Final Test Accuracy Comparison on CIFAR-10', fontsize=18, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0, top=max(final_accuracies) + 10)

    # Add accuracy labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, 
                f'{yval:.2f}%', 
                va='bottom', ha='center', fontsize=13, fontweight='bold')

    # Add horizontal line for ViT-Tiny baseline
    vit_accuracy = results_data['ViT-Tiny']['final_test_accuracy']
    ax.axhline(y=vit_accuracy, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'ViT-Tiny Baseline ({vit_accuracy:.2f}%)')
    ax.legend(fontsize=11, loc='upper left')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(IMAGE_DIR, 'final_accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison bar chart saved to '{save_path}'")
    plt.show()


def plot_efficiency_comparison(results_data):
    """
    Creates a scatter plot: Accuracy vs Training Speed (epoch time).
    Larger bubbles = more parameters.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for model_name, data in results_data.items():
        ax.scatter(data['epoch_time_seconds'], 
                   data['final_test_accuracy'],
                   s=data['params_millions'] * 100,  # Bubble size proportional to params
                   color=data['color'],
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=2,
                   label=f"{model_name} ({data['params_millions']:.1f}M params)")
        
        # Add text annotations
        ax.annotate(f"{data['final_test_accuracy']:.2f}%\n{data['epoch_time_seconds']}s/epoch",
                    xy=(data['epoch_time_seconds'], data['final_test_accuracy']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=data['color'], alpha=0.3))

    ax.set_title('Efficiency Comparison: Accuracy vs Training Speed', fontsize=18, fontweight='bold')
    ax.set_xlabel('Epoch Training Time (seconds, log scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    
    save_path = os.path.join(IMAGE_DIR, 'efficiency_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Efficiency scatter plot saved to '{save_path}'")
    plt.show()


def plot_speedup_analysis(results_data):
    """
    Bar chart showing speedup relative to original BDH model.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline_time = results_data['Vision-BDH (Original)']['epoch_time_seconds']
    
    model_names = []
    speedups = []
    colors_list = []
    
    for model_name, data in results_data.items():
        speedup = baseline_time / data['epoch_time_seconds']
        model_names.append(model_name)
        speedups.append(speedup)
        colors_list.append(data['color'])

    bars = ax.bar(model_names, speedups, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_title('Training Speed Improvement vs Original BDH', fontsize=18, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0, top=max(speedups) * 1.2)

    # Add speedup labels
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, 
                f'{yval:.1f}×', 
                va='bottom', ha='center', fontsize=13, fontweight='bold')

    # Add horizontal line at 1× (baseline)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (Original BDH)')
    ax.legend(fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(IMAGE_DIR, 'speedup_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Speedup analysis saved to '{save_path}'")
    plt.show()


def print_summary_table(results_data):
    """
    Prints a nice ASCII table summarizing all results.
    """
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Model':<30} {'Params':<12} {'Test Acc':<12} {'Epoch Time':<15} {'Total Time (10 ep)':<20}")
    print("-" * 100)
    
    for model_name, data in results_data.items():
        total_time_min = (data['epoch_time_seconds'] * 10) / 60
        print(f"{model_name:<30} {data['params_millions']:.1f}M{'':<8} "
              f"{data['final_test_accuracy']:.2f}%{'':<6} "
              f"{data['epoch_time_seconds']:.0f}s{'':<11} "
              f"{total_time_min:.1f} min")
    
    print("=" * 100)
    
    # Calculate improvements
    original = results_data['Vision-BDH (Original)']
    optimized = results_data['Vision-BDH (Optimized)']
    vit = results_data['ViT-Tiny']
    
    print("\nKEY COMPARISONS:")
    print(f"  Optimized vs Original:")
    print(f"    - Speedup: {original['epoch_time_seconds'] / optimized['epoch_time_seconds']:.1f}×")
    print(f"    - Accuracy change: {optimized['final_test_accuracy'] - original['final_test_accuracy']:+.2f}pp")
    print(f"    - Parameter reduction: {(1 - optimized['params_millions']/original['params_millions'])*100:.1f}%")
    
    print(f"\n  Optimized vs ViT-Tiny:")
    print(f"    - Accuracy advantage: {optimized['final_test_accuracy'] - vit['final_test_accuracy']:+.2f}pp")
    print(f"    - Speed ratio: {optimized['epoch_time_seconds'] / vit['epoch_time_seconds']:.2f}× (Optimized/ViT)")
    print(f"    - Parameter advantage: {(1 - optimized['params_millions']/vit['params_millions'])*100:.1f}% fewer")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  VISION-BDH: COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    # Print summary table first
    print_summary_table(results)
    
    # Generate all plots
    print("\n--- Generating visualizations ---\n")
    
    plot_learning_curves(results)
    print()
    
    plot_final_accuracy_comparison(results)
    print()
    
    plot_efficiency_comparison(results)
    print()
    
    plot_speedup_analysis(results)
    print()
    
    print("=" * 60)
    print(f"✓ All visualizations saved to '{IMAGE_DIR}/' directory")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. learning_curves.png - Training dynamics comparison")
    print("  2. final_accuracy_comparison.png - Final results bar chart")
    print("  3. efficiency_comparison.png - Accuracy vs Speed scatter plot")
    print("  4. speedup_analysis.png - Speedup relative to original")
    print("\nYou can now use these in your README and presentations!")
    print("=" * 60)