# main.py
"""
Quick test: BDH-Vision with MLP multiplier = 32 (instead of 128)
Goal: Verify if training speed improves while maintaining good accuracy

Expected results:
- Speed: ~800-1000s per epoch (vs 7500s with multiplier=128)
- Accuracy: ~70-72% after 10 epochs (vs 72.51% with multiplier=128)
"""

import torch
from torch import nn
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import math

from models.bdh import BDHConfig
from models.vision_bdh import VisionBDH


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate schedule with linear warmup followed by cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    """
    Train BDH-Vision with MLP multiplier = 32 (optimized for speed)
    """
    # --- Configuration ---
    EPOCHS = 10
    BATCH_SIZE = 32
    INITIAL_LR = 1e-4 
    WARMUP_STEPS = 500
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_mlp32"
    
    # KEY CHANGE: MLP multiplier 32 instead of 128
    MLP_MULTIPLIER = 32

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print("     Testing BDH-Vision with MLP Multiplier = 32")
    print("=" * 70)
    print(f"Configuration: {EPOCHS} epochs, Batch: {BATCH_SIZE}, LR: {INITIAL_LR}")
    print(f"Device: {DEVICE}")
    print(f"MLP Multiplier: {MLP_MULTIPLIER} (original was 128)")
    print("=" * 70)

    # --- Model Configuration ---
    config = BDHConfig(
        n_layer=6, 
        n_embd=192, 
        n_head=6, 
        vocab_size=256,
        mlp_internal_dim_multiplier=MLP_MULTIPLIER  # <-- KEY CHANGE!
    )
    model = VisionBDH(bdh_config=config, img_size=32, patch_size=4, num_classes=10).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.05)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlp_dim = MLP_MULTIPLIER * 192 // 6  # Calculate actual MLP dimension
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_params / 1e6:.2f}M")
    print(f"  MLP internal dim: {mlp_dim * 6} (per head: {mlp_dim})")
    print(f"  Original model had: 24,576 MLP dim (128 multiplier)")
    print(f"  New model has: {mlp_dim * 6} MLP dim ({MLP_MULTIPLIER} multiplier)")
    print(f"  Reduction: {24576 / (mlp_dim * 6):.1f}x smaller MLP\n")

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
    
    full_train_dataset = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    
    train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # --- Learning Rate Scheduler ---
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps)

    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("\n" + "=" * 70)
    print("     Starting Training")
    print("=" * 70 + "\n")
    
    epoch_times = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                
                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start_time
        
        epoch_times.append(epoch_time)
        val_accuracies.append(val_accuracy)
        
        print("-" * 70)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print("-" * 70)
        
        # --- Save Checkpoint ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'mlp_multiplier': MLP_MULTIPLIER
        }, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}\n")
        
    print("\n" + "=" * 70)
    print("     Training Finished")
    print("=" * 70)

    # --- Final Test Evaluation ---
    print("\n" + "=" * 70)
    print("     Final Evaluation on Test Set")
    print("=" * 70)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    total_training_time = sum(epoch_times)
    
    print("-" * 70)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("-" * 70)
    
    # --- Save Final Model ---
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to {final_model_path}")
    
    # --- Print Comparison ---
    print("\n" + "=" * 70)
    print("     COMPARISON WITH ORIGINAL MODEL")
    print("=" * 70)
    print(f"{'Metric':<30} {'Original (128x)':<20} {'Optimized (32x)':<20}")
    print("-" * 70)
    print(f"{'MLP Dimension':<30} {'24,576':<20} {f'{mlp_dim * 6}':<20}")
    print(f"{'Parameters':<30} {'6.5M':<20} {f'{num_params/1e6:.2f}M':<20}")
    print(f"{'Test Accuracy':<30} {'72.51%':<20} {f'{test_accuracy:.2f}%':<20}")
    print(f"{'Avg Epoch Time':<30} {'~7500s':<20} {f'{avg_epoch_time:.1f}s':<20}")
    print(f"{'Total Training Time':<30} {'~20.8 hours':<20} {f'{total_training_time/3600:.1f} hours':<20}")
    print(f"{'Speedup':<30} {'1.0x (baseline)':<20} {f'{7500/avg_epoch_time:.1f}x faster':<20}")
    print("=" * 70)
    
    # --- Analysis ---
    accuracy_diff = test_accuracy - 72.51
    speedup = 7500 / avg_epoch_time
    
    print("\n" + "=" * 70)
    print("     ANALYSIS")
    print("=" * 70)
    if speedup > 5:
        print(f"✅ MAJOR SPEEDUP: {speedup:.1f}x faster training!")
    elif speedup > 2:
        print(f"✅ Good speedup: {speedup:.1f}x faster training")
    else:
        print(f"⚠️  Limited speedup: {speedup:.1f}x faster (expected more)")
    
    if accuracy_diff > -1:
        print(f"✅ Accuracy maintained: {accuracy_diff:+.2f}pp difference (excellent!)")
    elif accuracy_diff > -3:
        print(f"✅ Acceptable accuracy loss: {accuracy_diff:+.2f}pp difference")
    else:
        print(f"⚠️  Significant accuracy loss: {accuracy_diff:+.2f}pp difference")
    
    print("\nConclusion:")
    if speedup > 5 and accuracy_diff > -2:
        print("🎉 SUCCESS! This configuration offers a great speed/accuracy trade-off.")
        print("   Recommended to use MLP multiplier = 32 for production.")
    elif speedup > 3:
        print("👍 Good result! Consider this configuration for faster iterations.")
    else:
        print("🤔 Results inconclusive. May need further optimization.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()