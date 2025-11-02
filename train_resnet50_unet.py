import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ======================
# Konfiguracja (Zmieniony Encoder na ResNet50)
# ======================
CONFIG = {
    "data_dir": "./data_camvid",
    "img_size": 384,
    "batch_size": 8,
    "epochs": 40,
    "lr": 1e-4,
    "encoder": "resnet50", # ZMIENIONO: z resnet34 na resnet50
    "num_classes": 11,
    "ignore_index": 11,
    "checkpoint_dir": "./checkpoints_camvid_resnet50_unet", # ZMIENIONO NAZWĘ FOLDERU
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_mapping": True
}


# ======================
# Dataset CamVid (Z OPTYMALIZACJĄ LUT)
# ======================
class CamVidDataset(Dataset):
    NUM_CLASSES = CONFIG['num_classes']
    IGNORE_INDEX = CONFIG['ignore_index']

    def __init__(self, root, split='train', transform=None, class_mapping=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.images_dir = self.root / split
        labels_folder1 = self.root / f"{split}_labels"
        labels_folder2 = self.root / f"{split}annot"
        self.masks_dir = labels_folder1 if labels_folder1.exists() else labels_folder2
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Nie znaleziono folderu masek dla split={split}")

        self.images = sorted(self.images_dir.glob('*.png'))
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        # Mapping klas
        if class_mapping is not None:
            self.mapping = class_mapping
            print(f"ℹ️ Split '{split}': używam przekazanego mappingu ({len(self.mapping)} klas)")
        else:
            self.mapping = self._create_class_mapping()
            print(f"ℹ️ Split '{split}': stworzono mapping dla {len(self.mapping)} unikalnych wartości pikseli")
            
        # UWAGA: OPTYMALIZACJA - Tworzenie Look-Up Table (LUT)
        max_val = max(self.mapping.keys()) if self.mapping else 0
        self.lut = np.zeros(max_val + 1, dtype=np.uint8) + self.IGNORE_INDEX
        for old_val, new_val in self.mapping.items():
            if old_val <= max_val:
                self.lut[old_val] = new_val
        print(f"✅ Stworzono LUT (rozmiar {len(self.lut)}) dla szybszego mapowania masek.")


    def _create_class_mapping(self):
        cache_file = self.root / f"class_mapping_{self.split}.json"
        if CONFIG['cache_mapping'] and cache_file.exists():
            print(f"📂 Ładowanie mappingu z cache: {cache_file}")
            with open(cache_file, 'r') as f:
                mapping = json.load(f)
            return {int(k): v for k, v in mapping.items()}

        unique_values = set()
        print(f"🔍 Skanowanie masek dla '{self.split}'...")
        for img_path in tqdm(self.images, desc="Scanning masks"):
            mask_path = self.masks_dir / (f"{img_path.stem}_L.png" if (self.masks_dir / f"{img_path.stem}_L.png").exists() else img_path.name)
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            unique_values.update(np.unique(mask))

        sorted_vals = sorted(unique_values)
        mapping = {int(v): min(i, self.IGNORE_INDEX) for i, v in enumerate(sorted_vals)}

        if CONFIG['cache_mapping']:
            print(f"💾 Zapis mappingu do cache: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        return mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (f"{img_path.stem}_L.png" if (self.masks_dir / f"{img_path.stem}_L.png").exists() else img_path.name)
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # OPTYMALIZACJA - Użycie LUT zamiast pętli
        mask_remap = self.lut[mask] 

        if self.transform:
            transformed = self.transform(image=image, mask=mask_remap)
            image, mask_remap = transformed['image'], transformed['mask']
        else:
            image = torch.as_tensor(image, dtype=torch.float32)
            mask_remap = torch.as_tensor(mask_remap, dtype=torch.long)

        return image, mask_remap.long()


# ======================
# Augmentacje
# ======================
def get_training_augmentation(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_validation_augmentation(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ======================
# Metryki
# ======================
def calculate_iou(pred, target, num_classes=11, ignore_index=11):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    valid = target != ignore_index
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls) & valid
        target_mask = (target == cls) & valid
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

# ======================
# Funkcja Straty (Dice + Cross-Entropy)
# ======================
def get_combined_loss(ignore_index):
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
    
    def combined_loss(y_pred, y_true):
        return 0.5 * ce_loss(y_pred, y_true) + 0.5 * dice_loss(y_pred, y_true)

    return combined_loss


# ======================
# Pętle trening/val
# ======================
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_iou = 0, 0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", ncols=120)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = outputs.argmax(1)
        iou = calculate_iou(preds, masks)
        total_loss += loss.item()
        total_iou += iou

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mIoU': f'{iou:.4f}', 'GPU_MB': f'{mem:.0f}'})
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mIoU': f'{iou:.4f}'})
    return total_loss / len(loader), total_iou / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0
    pbar = tqdm(loader, desc="Validation", ncols=120)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        preds = outputs.argmax(1)
        iou = calculate_iou(preds, masks)
        total_loss += loss.item()
        total_iou += iou
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mIoU': f'{iou:.4f}'})
    return total_loss / len(loader), total_iou / len(loader)


# ======================
# Main
# ======================
def main():
    print("=" * 60)
    print("🖥️  System Info:")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # Użycie ResNet50 z wagami ImageNet
    model = smp.Unet(
        encoder_name=CONFIG['encoder'],
        encoder_weights='imagenet',
        in_channels=3,
        classes=CONFIG['num_classes'] + 1 
    ).to(CONFIG['device'])

    print(f"✅ Model loaded: {CONFIG['encoder']} (Output channels: {CONFIG['num_classes'] + 1})")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    train_dataset = CamVidDataset(CONFIG['data_dir'], split='train', transform=get_training_augmentation(CONFIG['img_size']))
    val_dataset = CamVidDataset(CONFIG['data_dir'], split='val', transform=get_validation_augmentation(CONFIG['img_size']),
                                class_mapping=train_dataset.mapping)

    num_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    # Użycie połączonej straty
    criterion = get_combined_loss(CONFIG['ignore_index'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_iou = 0
    history = defaultdict(list)

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'], epoch)
        val_loss, val_iou = val_epoch(model, val_loader, criterion, CONFIG['device'])
        scheduler.step(val_iou)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   mIoU: {val_iou:.4f} | LR: {lr_now:.6f}")

        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = f"{CONFIG['checkpoint_dir']}/best_e{epoch:02d}_miou{val_iou:.3f}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_iou,
                'class_mapping': train_dataset.mapping,
                'config': CONFIG
            }, ckpt_path)
            print(f"  ✅ New best model saved: {ckpt_path}")
        
        if epoch % 5 == 0:
            ckpt_path = f"{CONFIG['checkpoint_dir']}/checkpoint_e{epoch:02d}.pth"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_miou': val_iou}, ckpt_path)


    with open(f"{CONFIG['checkpoint_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n🏁 Training completed! Best validation mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()