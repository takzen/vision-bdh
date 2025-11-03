import torch
from torch import nn
from torch.optim import AdamW
import timm
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import math
import csv

# ==============================================================================
# Training Script for DeiT-Tiny on CIFAR-100
# ==============================================================================

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
    # --- Configuration (identical to other CIFAR-100 experiments) ---
    EPOCHS = 50
    BATCH_SIZE = 32
    INITIAL_LR = 1e-4
    WARMUP_STEPS = 500
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_deit_tiny_cifar100"
    LOG_FILE = os.path.join(CHECKPOINT_DIR, "metrics_deit_tiny_cifar100.csv")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 60)
    print("     Training DeiT-Tiny on CIFAR-100 (Baseline)")
    print("=" * 60)
    print(f"Configuration: {EPOCHS} epochs, Batch: {BATCH_SIZE}, LR: {INITIAL_LR}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Model ---
    # Load the DeiT-Tiny model from timm, training from scratch for 100 classes.
    # pretrained=False: We train from scratch.
    # num_classes=100: Sets the output layer size.
    # img_size=32: Adapts the model for 32x32 images.
    model = timm.create_model(
        'deit_tiny_patch16_224', 
        pretrained=False, 
        num_classes=100,
        img_size=32
    ).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.05)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeiT-Tiny created with {num_params / 1e6:.2f}M trainable parameters.")

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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # --- Scheduler and Loss Function ---
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("\n" + "=" * 60)
    print("     Starting Training")
    print("=" * 60)
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_accuracy", "epoch_time_sec", "learning_rate"])

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
        
        print("-" * 60)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print("-" * 60)
        
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_accuracy, epoch_time, scheduler.get_last_lr()[0]])

        # --- Checkpoint ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
        
    print("\n" + "=" * 60)
    print("     Training Finished")
    print("=" * 60)

    # --- Final Test Evaluation ---
    print("\n" + "=" * 60)
    print("     Final Evaluation on Test Set (using last epoch model)")
    print("=" * 60)
    
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
    print("-" * 60)
    print(f"Final Test Accuracy (last epoch model): {test_accuracy:.2f}%")
    print("-" * 60)
    
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model_deit_tiny_cifar100.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()