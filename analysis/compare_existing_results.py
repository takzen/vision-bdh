"""
Comparison of Vision-BDH architectural variants on CIFAR-10.
This script analyzes the impact of normalization style and attention mechanism.
"""
import matplotlib.pyplot as plt
import numpy as np

# --- Final, curated results from all 50-epoch experiments ---
results = {
    'Vision-BDH v2 (Double-LN, Linear Attn)': {
        'test_acc': 81.73,
        'params': 3.2,
        'notes': 'Best performance'
    },
    'Vision-BDH v2 (Double-LN, Softmax)': {
        'test_acc': 80.45,
        'params': 3.2,
        'notes': 'Ablation: Softmax added'
    },
    'Vision-BDH v1 (Hybrid-LN, Linear Attn)': {
        'test_acc': 80.43,
        'params': 3.6,
        'notes': 'Original architecture'
    },
    'Vision-BDH v3 (Pre-LN + ScaledLN, Linear Attn)': {
        'test_acc': 77.27,
        'params': 3.2,
        'notes': 'Ablation: ScaledLN experiment'
    },
    'ViT-Tiny (Baseline)': {
        'test_acc': 76.05,
        'params': 5.7,
        'notes': 'External baseline'
    }
}

# --- Create comparison plot ---
fig, ax = plt.subplots(figsize=(12, 7))
plt.style.use("seaborn-v0_8-whitegrid")

# Sort data by accuracy for a clean, ranked chart
sorted_results = sorted(results.items(), key=lambda item: item[1]['test_acc'])

# Combine model name and parameter count into a single, multi-line label
names_with_params = [f"{item[0]}\n({item[1]['params']:.1f}M params)" for item in sorted_results]
accs = [item[1]['test_acc'] for item in sorted_results]

# A distinct color palette to easily differentiate models
# Best model is green, baseline is gray, experiments are other colors
colors = ['#95a5a6', '#e74c3c', '#e67e22', '#3498db', '#2ecc71']

# Create horizontal bar chart
bars = ax.barh(np.arange(len(names_with_params)), accs, color=colors, alpha=0.9, edgecolor='black', height=0.6)

# Add accuracy labels outside the bars for clarity
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}%', 
            ha='left', va='center', 
            fontweight='bold', color='black', fontsize=11)

# --- STYLING ---
ax.set_yticks(np.arange(len(names_with_params)))
ax.set_yticklabels(names_with_params, fontsize=10) # Slightly smaller font for long names
ax.invert_yaxis()  # Display the best result on top

# Adjust x-axis to prevent labels from being cut off
ax.set_xlim(0, max(accs) + 5) 

# Hide X-axis line and labels, as they are redundant
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax.set_xlabel('')

ax.set_title('Vision-BDH Architecture Ablation Study (CIFAR-10, 50 epochs)', 
             fontsize=16, fontweight='bold')

ax.grid(True, axis='x', linestyle='--', alpha=0.6)

# Remove frame spines for a modern, data-focused look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add a vertical line at 0 for reference
ax.axvline(0, color='black', linewidth=1.5)

plt.tight_layout(pad=1.5)
plt.savefig('vision_bdh_ablation_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: vision_bdh_ablation_comparison.png")
plt.show()

# --- Print summary table ---
print("\n" + "="*80)
print("VISION-BDH ARCHITECTURE ABLATION SUMMARY (CIFAR-10, 50 Epochs)")
print("="*80)
print(f"{'Model Configuration':<50} {'Accuracy (%)':<15} {'Parameters (M)':<15}")
print("-" * 80)

for name, data in sorted(results.items(), key=lambda item: item[1]['test_acc'], reverse=True):
    print(f"{name:<50} {data['test_acc']:<15.2f} {data['params']:.1f}")

print("="*80)

# Calculate and print key findings from the ablation study
best_v2_acc = results['Vision-BDH v2 (Double-LN, Linear Attn)']['test_acc']
softmax_v2_acc = results['Vision-BDH v2 (Double-LN, Softmax)']['test_acc']
v3_acc = results['Vision-BDH v3 (Pre-LN + ScaledLN, Linear Attn)']['test_acc']

print("\nKey Findings from Ablation Study:")
print(f"  1. Impact of Linear Attention (No Softmax): +{best_v2_acc - softmax_v2_acc:.2f}pp")
print(f"     (Comparing v2 w/ Softmax vs. v2 w/o Softmax: {softmax_v2_acc}% → {best_v2_acc}%)")
print(f"  2. Impact of Normalization Style (Double-LN vs. ScaledLN):")
print(f"     ScaledLN (v3) performed significantly worse: {v3_acc}% vs {best_v2_acc}% (-{best_v2_acc - v3_acc:.2f}pp)")
print(f"  3. Conclusion: The combination of Double-LN and Linear Attention is optimal.")
print("="*80)