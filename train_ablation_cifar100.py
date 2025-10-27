# train_ablation_cifar100.py
"""
Ablation Study: Normalization Strategy Comparison on CIFAR-100.

This script trains Vision-BDH v2 with different normalization strategies
to systematically evaluate their impact on model performance.

Supported strategies:
- pre_ln: Pre-LayerNorm (recommended, default in v2)
- post_ln: Post-LayerNorm (original Transformer style)
- double_ln: Double LayerNorm (experimental)

Usage:
    python train_ablation_cifar100.py --norm-style pre_ln
    python train_ablation_cifar100.py --norm-style post_ln
    python train_ablation_cifar100.py --norm-style double_ln
"""

import torch
from torch import nn
from torch.optim import AdamW
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import argparse
import os
import glob
import math
import csv
from typing import Literal

# Import the unified model with ablation support
from models.bdh import BDHConfig
from models.vision_bdh_v2 import VisionBDHv2


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate schedule with linear warmup followed by cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Index of last epoch (for resuming)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, grad_clip=1.0):
    """
    Train for one epoch.
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def save_checkpoint(model, optimizer, epoch, val_accuracy, checkpoint_dir, norm_style):
    """Save training checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'norm_style': norm_style  # Save which strategy was used
    }, checkpoint_path)
    return checkpoint_path


def load_best_checkpoint(checkpoint_dir):
    """
    Load the checkpoint with the best validation accuracy.
    
    Returns:
        Path to best checkpoint and its validation accuracy
    """
    best_acc = 0
    best_path = ""
    
    for ckpt_path in glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'val_accuracy' in ckpt and ckpt['val_accuracy'] > best_acc:
            best_acc = ckpt['val_accuracy']
            best_path = ckpt_path
    
    return best_path, best_acc


def main(args):
    """
    Main training function for ablation study.
    """
    # --- Configuration ---
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    INITIAL_LR = args.lr
    WARMUP_STEPS = args.warmup_steps
    GRAD_CLIP = args.grad_clip
    VALIDATION_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create checkpoint directory specific to this ablation
    CHECKPOINT_DIR = f"./checkpoints_ablation_{args.norm_style}_cifar100"
    LOG_FILE = os.path.join(CHECKPOINT_DIR, f"metrics_{args.norm_style}.csv")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # GPU performance optimizations
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # --- Print Configuration ---
    print("=" * 70)
    print(f"     ABLATION STUDY: Vision-BDH with {args.norm_style.upper()}")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Normalization Strategy: {args.norm_style}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {INITIAL_LR}")
    print(f"MLP Multiplier: {args.mlp_multiplier}")
    print("=" * 70)

    # --- Model Configuration ---
    config = BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_multiplier
    )
    
    # Create model with specified normalization strategy
    model = VisionBDHv2(
        bdh_config=config, 
        img_size=32, 
        patch_size=4, 
        num_classes=100,
        norm_style=args.norm_style  # This is the key parameter!
    )
    
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=args.weight_decay)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created with {num_params / 1e6:.2f}M trainable parameters.\n")

    # --- Data Preparation ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = CIFAR100(root="./data_cifar100", train=True, download=True, transform=train_transform)
    train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = CIFAR100(root="./data_cifar100", train=False, download=True, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=args.num_workers)

    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}\n")

    # --- Scheduler & Loss ---
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Initialize CSV Log ---
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_accuracy", "epoch_time_sec", 
                           "learning_rate", "norm_style"])

    # --- Training Loop ---
    print("=" * 70)
    print(f"     Starting Training ({args.norm_style})")
    print("=" * 70 + "\n")

    best_val_acc = 0

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, DEVICE, GRAD_CLIP
        )
        
        # Validate
        val_accuracy = evaluate(model, val_loader, DEVICE)
        epoch_time = time.time() - epoch_start_time
        
        # Track best
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

        # Print summary
        print("-" * 70)
        print(f"Epoch {epoch+1}/{EPOCHS} ({args.norm_style}):")
        print(f"  Train Loss:       {avg_train_loss:.4f}")
        print(f"  Val Accuracy:     {val_accuracy:.2f}%")
        print(f"  Best Val Acc:     {best_val_acc:.2f}%")
        print(f"  Epoch Time:       {epoch_time:.2f}s")
        print(f"  Learning Rate:    {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 70 + "\n")

        # Log to CSV
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                avg_train_loss, 
                val_accuracy, 
                epoch_time, 
                scheduler.get_last_lr()[0],
                args.norm_style
            ])

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, val_accuracy, CHECKPOINT_DIR, args.norm_style
        )
        print(f"✓ Checkpoint saved: {checkpoint_path}\n")

    # --- Final Evaluation ---
    print("\n" + "=" * 70)
    print(f"     Final Test Evaluation ({args.norm_style})")
    print("=" * 70)

    best_path, best_val = load_best_checkpoint(CHECKPOINT_DIR)
    
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        print(f"Best validation accuracy: {best_val:.2f}%\n")
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Warning: No checkpoint found, using final model state.\n")

    test_accuracy = evaluate(model, test_loader, DEVICE)
    
    print("=" * 70)
    print(f"FINAL RESULTS ({args.norm_style}):")
    print(f"  Best Val Accuracy:  {best_val:.2f}%")
    print(f"  Test Accuracy:      {test_accuracy:.2f}%")
    print("=" * 70)
    
    # Save final results
    results_file = os.path.join(CHECKPOINT_DIR, "final_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Normalization Strategy: {args.norm_style}\n")
        f.write(f"Best Validation Accuracy: {best_val:.2f}%\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Total Parameters: {num_params / 1e6:.2f}M\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation study for Vision-BDH normalization strategies on CIFAR-100"
    )
    
    # Ablation parameter
    parser.add_argument(
        '--norm-style', 
        type=str,
        choices=['pre_ln', 'post_ln', 'double_ln'],
        default='pre_ln',
        help='Normalization strategy (pre_ln, post_ln, or double_ln)'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping threshold')
    
    # Model architecture
    parser.add_argument('--n-layer', type=int, default=6, help='Number of BDH layers')
    parser.add_argument('--n-embd', type=int, default=192, help='Embedding dimension')
    parser.add_argument('--n-head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--mlp-multiplier', type=int, default=32, help='MLP internal dimension multiplier')
    
    # System
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)