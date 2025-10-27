# compare_ablations.py
import pandas as pd
import matplotlib.pyplot as plt

styles = ['pre_ln', 'post_ln', 'double_ln']
results = {}

for style in styles:
    df = pd.read_csv(f'checkpoints_ablation_{style}_cifar100/metrics_{style}.csv')
    results[style] = df

# Plot learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for style, df in results.items():
    plt.plot(df['epoch'], df['train_loss'], label=style)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.legend()
plt.title('Training Loss Comparison')

plt.subplot(1, 2, 2)
for style, df in results.items():
    plt.plot(df['epoch'], df['val_accuracy'], label=style)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy Comparison')

plt.tight_layout()
plt.savefig('ablation_comparison.png')