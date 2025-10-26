import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import math
import csv

# ==============================================================================
# Full, standard implementation of ResNet for CIFAR
# (Based on official implementations, e.g., from the PyTorch repository)
# ==============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20(num_classes=10):
    # ResNet-20 has 3 blocks in each of its 3 layers (3*3*2 conv + 2 other = 20 layers)
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


# ==============================================================================
# Training Script (Your logic, adapted)
# ==============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
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
    INITIAL_LR = 1e-4 # Note: ResNets are often trained with a higher LR, but we keep this for consistency
    WARMUP_STEPS = 500
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_resnet20_cifar100"
    LOG_FILE = os.path.join(CHECKPOINT_DIR, "metrics_resnet20_cifar100.csv")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 60)
    print("     Training ResNet-20 on CIFAR-100 (Baseline)")
    print("=" * 60)
    print(f"Configuration: {EPOCHS} epochs, Batch: {BATCH_SIZE}, LR: {INITIAL_LR}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Model ---
    model = ResNet20(num_classes=100).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.05)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-20 created with {num_params / 1e6:.2f}M trainable parameters.")

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

    # --- Final Evaluation (on the last epoch model) ---
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
    
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model_resnet20_cifar100.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()