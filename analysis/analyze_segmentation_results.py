"""
CamVid Segmentation Results Analysis
------------------------------------

Compares results of multiple models (e.g., BDH-UNet and ResNet-UNet) on CamVid dataset.

Generates:
1. Learning curves plots (loss + mIoU)
2. Performance comparison table
3. Convergence speed analysis
4. Overfitting analysis plot
5. Text summary report

Usage:
    python analyze_segmentation_results.py
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ===============================================
# CONFIGURATION
# ===============================================

CHECKPOINT_DIRS = {
    'bdh_unet': './checkpoints_camvid_bdh_rope_fixed',       # BDH U-Net
    'bdh_unet_finetune': './checkpoints_finetune_bdh',       # BDH U-Net fine-tuning
    'resnet34_unet': './checkpoints_camvid_resnet34_unet',   # ResNet34 U-Net
    'resnet50_unet': './checkpoints_camvid_resnet50_unet'    # ResNet50 U-Net
}

SAVE_DIR = './analysis_results_camvid'

COLORS = {
    'bdh_unet': '#2ecc71',          # green
    'bdh_unet_finetune': "#b0c514", # lime
    'resnet34_unet': '#3498db',     # blue
    'resnet50_unet': "#751950",     # purple
}


# ===============================================
# HELPER FUNCTIONS
# ===============================================

def load_training_history(checkpoint_dir: str) -> dict:
    """Load saved training_history.json file"""
    history_path = Path(checkpoint_dir) / 'training_history.json'
    
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


# ===============================================
# 1️⃣ LEARNING CURVES
# ===============================================
def plot_learning_curves(histories: dict, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CamVid Segmentation: Learning Curves Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # --- Training Loss ---
    ax = axes[0, 0]
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 
                label=model_name.replace('_', '-').upper(),
                linewidth=2.5, color=COLORS.get(model_name, None))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    
    # --- Validation Loss ---
    ax = axes[0, 1]
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'],
                label=model_name.replace('_', '-').upper(),
                linewidth=2.5, color=COLORS.get(model_name, None))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    
    # --- Training mIoU ---
    ax = axes[1, 0]
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_miou']) + 1)
        ax.plot(epochs, [x*100 for x in history['train_miou']],
                label=model_name.replace('_', '-').upper(),
                linewidth=2.5, color=COLORS.get(model_name, None))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU [%]')
    ax.set_title('Training mIoU')
    ax.legend()
    
    # --- Validation mIoU ---
    ax = axes[1, 1]
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_miou']) + 1)
        miou_vals = [x*100 for x in history['val_miou']]
        ax.plot(epochs, miou_vals,
                label=model_name.replace('_', '-').upper(),
                linewidth=2.5, color=COLORS.get(model_name, None))
        
        # Mark best epoch
        best_epoch = np.argmax(history['val_miou']) + 1
        best_miou = max(history['val_miou']) * 100
        ax.scatter(best_epoch, best_miou, s=150, color=COLORS.get(model_name, None),
                   marker='*', edgecolors='black', linewidths=1.5)
        ax.annotate(f'{best_miou:.2f}%',
                    xy=(best_epoch, best_miou),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU [%]')
    ax.set_title('Validation mIoU (⭐ Best result)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: learning_curves_comparison.png")


# ===============================================
# 2️⃣ PERFORMANCE COMPARISON TABLE
# ===============================================
def create_performance_table(histories: dict, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for model_name, history in histories.items():
        best_epoch = np.argmax(history['val_miou']) + 1
        best_val_miou = max(history['val_miou']) * 100
        final_val_miou = history['val_miou'][-1] * 100
        final_train_miou = history['train_miou'][-1] * 100
        convergence_epoch = next(
            (i+1 for i, x in enumerate([v*100 for v in history['val_miou']]) if x >= 0.9 * best_val_miou),
            len(history['val_miou'])
        )

        results.append({
            'Model': model_name.replace('_', '-').upper(),
            'Best Val mIoU [%]': f'{best_val_miou:.2f}',
            'Best Epoch': best_epoch,
            'Final Val mIoU [%]': f'{final_val_miou:.2f}',
            'Final Train mIoU [%]': f'{final_train_miou:.2f}',
            'Convergence (90%)': f'{convergence_epoch} ep'
        })
    
    df = pd.DataFrame(results)
    csv_path = Path(save_dir) / 'performance_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: performance_comparison.csv\n")
    print(df)
    return df


# ===============================================
# 3️⃣ CONVERGENCE COMPARISON
# ===============================================
def plot_convergence_comparison(histories: dict, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_miou']) + 1)
        vals = [x*100 for x in history['val_miou']]
        best = max(vals)
        norm = [(x / best) * 100 for x in vals]
        plt.plot(epochs, norm, label=model_name.replace('_', '-').upper(),
                 linewidth=2.5, color=COLORS.get(model_name, None))
    plt.axhline(90, color='red', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(95, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('% of best result')
    plt.title('Convergence Speed Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: convergence_comparison.png")


# ===============================================
# 4️⃣ OVERFITTING ANALYSIS
# ===============================================
def analyze_overfitting(histories: dict, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(histories), figsize=(8*len(histories), 6))

    if len(histories) == 1:
        axes = [axes]

    for idx, (model_name, history) in enumerate(histories.items()):
        ax = axes[idx]
        epochs = range(1, len(history['train_miou']) + 1)
        train = [x*100 for x in history['train_miou']]
        val = [x*100 for x in history['val_miou']]
        gap = [t - v for t, v in zip(train, val)]
        ax.plot(epochs, train, label='Train mIoU', linewidth=2.5, color='green')
        ax.plot(epochs, val, label='Val mIoU', linewidth=2.5, color='blue')
        ax.fill_between(epochs, train, val, color='red', alpha=0.2)
        ax.set_title(f"{model_name.replace('_', '-').upper()} (Avg. gap: {np.mean(gap):.2f}%)")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU [%]')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: overfitting_analysis.png")


# ===============================================
# 5️⃣ TEXT SUMMARY REPORT
# ===============================================
def create_summary_report(histories: dict, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CAMVID SEGMENTATION: ANALYSIS SUMMARY")
    report_lines.append("="*80)
    
    for model_name, history in histories.items():
        best_epoch = np.argmax(history['val_miou']) + 1
        best_val_miou = max(history['val_miou']) * 100
        final_val_miou = history['val_miou'][-1] * 100
        final_train_miou = history['train_miou'][-1] * 100
        report_lines.append(f"\n🧠 Model: {model_name.replace('_', '-').upper()}")
        report_lines.append(f"   Best mIoU: {best_val_miou:.2f}% (Epoch {best_epoch})")
        report_lines.append(f"   Final mIoU: {final_val_miou:.2f}%")
        report_lines.append(f"   Train mIoU: {final_train_miou:.2f}%")
        report_lines.append("-"*60)
    
    best_model = max(histories.keys(), key=lambda k: max(histories[k]['val_miou']))
    best_score = max(histories[best_model]['val_miou']) * 100
    report_lines.append(f"\n🏆 Best model: {best_model.replace('_', '-').upper()} ({best_score:.2f}%)")
    
    report_path = Path(save_dir) / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("\n".join(report_lines))
    print(f"✅ Saved: analysis_report.txt")


# ===============================================
# MAIN
# ===============================================
def main():
    print("\n" + "="*80)
    print("CAMVID SEGMENTATION RESULTS ANALYSIS")
    print("="*80 + "\n")

    histories = {}
    for name, dir_path in CHECKPOINT_DIRS.items():
        try:
            hist = load_training_history(dir_path)
            histories[name] = hist
            print(f"✅ Loaded history: {name}")
        except FileNotFoundError:
            print(f"⚠️  History not found: {name}")
    
    if not histories:
        print("❌ No data to analyze. Make sure training_history.json files exist.")
        return
    
    print("\n📊 Analyzing models...\n")
    plot_learning_curves(histories)
    create_performance_table(histories)
    plot_convergence_comparison(histories)
    analyze_overfitting(histories)
    create_summary_report(histories)
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete! Results saved in: {SAVE_DIR}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()