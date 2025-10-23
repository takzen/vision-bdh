# main.py
"""
Train Vision-BDH for 30 epochs with torch.compile() enabled.
Goal: Find the maximum potential accuracy of the optimized architecture.
"""

import torch
from torch import nn
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import argparse
import os
import glob
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


def main(args):
    """
    Train BDH-Vision with MLP multiplier = 32, for 30 epochs, with compilation.
    """
    # --- Configuration ---
    EPOCHS = 30
    BATCH_SIZE = 32
    INITIAL_LR = 1e-4 
    WARMUP_STEPS = 500
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_mlp32_30epochs"
    MLP_MULTIPLIER = 32

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Performance optimizations for GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print("     Training Vision-BDH (MLP Multiplier = 32, 30 Epochs)")
    print("=" * 70)
    print(f"Configuration: {EPOCHS} epochs, Batch: {BATCH_SIZE}, LR: {INITIAL_LR}")
    print(f"Device: {DEVICE}, Compilation: ENABLED")
    print("=" * 70)

    # --- Model Configuration ---
    config = BDHConfig(
        n_layer=6, 
        n_embd=192, 
        n_head=6, 
        vocab_size=256,
        mlp_internal_dim_multiplier=MLP_MULTIPLIER
    )
    model = VisionBDH(bdh_config=config, img_size=32, patch_size=4, num_classes=10)

    # --- Enable torch.compile() ---
    print("\nCompiling the model... (this may take a moment on the first run)")
    try:
        # =================================================================
        # KLUCZOWA ZMIANA: Dodajemy backend, który działa na Windowsie
        model = torch.compile(model, backend="aot_eager")
        # =================================================================
        print("✓ Model compiled successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Model compilation failed, continuing without it. Error: {e}")

    model.to(DEVICE) # Move to device AFTER compilation
    
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.05)

    # --- Resume Logic ---
    start_epoch = 0
    if args.resume:
        list_of_files = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pth'))
        if not list_of_files:
            print("No checkpoint found to resume from.")
        else:
            latest_checkpoint_path = max(list_of_files, key=os.path.getctime)
            print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created with {num_params / 1e6:.2f}M trainable parameters.")
    
    # --- Data Preparation ---
    # ... (ta sekcja pozostaje bez zmian) ...
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
    # ... (ta sekcja pozostaje bez zmian) ...
    print("\n" + "=" * 70)
    print("     Starting Training")
    print("=" * 70 + "\n")
    
    for epoch in range(start_epoch, EPOCHS):
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
            'val_accuracy': val_accuracy
        }, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}\n")

    # ... (reszta skryptu: finalna ewaluacja, zapis - pozostaje bez zmian) ...
    print("\n" + "=" * 70)
    print("     Training Finished")
    print("=" * 70)

    # --- Final Test Evaluation ---
    # Find best model from checkpoints
    best_accuracy = 0
    best_checkpoint_path = ""
    list_of_files = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pth'))
    for checkpoint_path in list_of_files:
        checkpoint = torch.load(checkpoint_path)
        if 'val_accuracy' in checkpoint and checkpoint['val_accuracy'] > best_accuracy:
            best_accuracy = checkpoint['val_accuracy']
            best_checkpoint_path = checkpoint_path
    
    print(f"\nLoading best model from: {best_checkpoint_path} with accuracy {best_accuracy:.2f}%")
    if best_checkpoint_path:
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "=" * 70)
    print("     Final Evaluation on Test Set (using best model)")
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
    
    print("-" * 70)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("-" * 70)
    
    # --- Save Final Model ---
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model_best.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Best model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionBDH on CIFAR-10")
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()
    
    main(args)