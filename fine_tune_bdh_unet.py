import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ======================
# BDH Config
# ======================
class BDHConfig:
    n_embd = 256
    n_head = 4
    n_layer = 6
    dropout = 0.1
    mlp_internal_dim_multiplier = 16

def get_freqs(N, theta=2**16, dtype=torch.float32):
    dim = N // 2
    inv_freq = theta ** (torch.arange(0, dim, dtype=dtype) / dim)
    return inv_freq

class BidirectionalAttentionV2(nn.Module):
    def __init__(self, config, use_softmax=True):
        super().__init__()
        self.config = config
        self.use_softmax = use_softmax
        D = config.n_embd
        N_mlp = config.mlp_internal_dim_multiplier * D // config.n_head
        RoPE_dim = N_mlp // 2
        self.freqs = nn.Parameter(get_freqs(N_mlp, dtype=torch.float32).view(1, 1, 1, RoPE_dim), requires_grad=False)

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * torch.pi)
        cos_partial = torch.cos(phases)
        sin_partial = torch.sin(phases)
        cos = torch.cat([cos_partial, cos_partial], dim=-1)
        sin = torch.cat([sin_partial, sin_partial], dim=-1)
        return cos, sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos, sin = BidirectionalAttentionV2.phases_cos_sin(phases)
        return (v * cos) + (v_rot * sin)

    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = QR @ KR.mT
        if self.use_softmax:
            scores = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)
        return scores @ V

class VisionBDHv2(nn.Module):
    def __init__(self, bdh_config, img_size=256, patch_size=8, num_classes=10, in_channels=3, use_softmax_attn=False):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_softmax_attn = use_softmax_attn
        self.patch_embed = nn.Conv2d(in_channels, bdh_config.n_embd, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, bdh_config.n_embd) * 0.02)
        self.build_bdh_layers()
        self.ln_final = nn.LayerNorm(bdh_config.n_embd, elementwise_affine=False, bias=False)
        self.head = nn.Linear(bdh_config.n_embd, num_classes)
        self.apply(self._init_weights)

    def build_bdh_layers(self):
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = D * C.mlp_internal_dim_multiplier // nh
        self.decoder = nn.Parameter(torch.empty((nh * N, D)))
        nn.init.xavier_uniform_(self.decoder)
        self.encoder = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder)
        self.encoder_v = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder_v)
        self.attn = BidirectionalAttentionV2(C, use_softmax=self.use_softmax_attn)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        B = x.shape[0]
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = x.unsqueeze(1)
        for level in range(C.n_layer):
            x = self.ln(x)
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            T = x.shape[2]
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder
            y = self.ln(yMLP).squeeze(1)
            x = self.ln(x.squeeze(1) + y).unsqueeze(1)
        return x.squeeze(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits

# ======================
# BDH-UNet
# ======================
class BDH_UNet(nn.Module):
    def __init__(self, bdh_config, num_classes=12, img_size=384, patch_size=8):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.img_size = img_size
        self.encoder = VisionBDHv2(bdh_config, img_size=img_size, patch_size=patch_size, num_classes=bdh_config.n_embd)
        D = bdh_config.n_embd
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(D, D//2, 2, 2),
            nn.BatchNorm2d(D//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(D//2, D//2, 3, 1, 1),
            nn.BatchNorm2d(D//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(D//2, D//4, 2, 2),
            nn.BatchNorm2d(D//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(D//4, D//4, 3, 1, 1),
            nn.BatchNorm2d(D//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(D//4, D//8, 2, 2),
            nn.BatchNorm2d(D//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(D//8, D//8, 3, 1, 1),
            nn.BatchNorm2d(D//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(D//8, num_classes, 1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        feats_tokens = self.encoder.forward_features(x)
        T = feats_tokens.shape[1]
        side = int(T ** 0.5)
        feats = feats_tokens.transpose(1, 2).reshape(B, self.config.n_embd, side, side)
        out = self.decoder(feats)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

# ======================
# Config
# ======================
CONFIG = {
    "data_dir": "./data_camvid",
    "img_size": 256,
    "batch_size": 8,
    "epochs": 10,
    "lr": 5e-4,
    "num_classes": 11,
    "ignore_index": 11,
    "checkpoint_dir": "./checkpoints_finetune_bdh",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patch_size": 8,
    "pretrained_model": "./checkpoints_camvid_bdh_rope_fixed/best_0.532.pth"
}

# ======================
# Dataset
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
        self.images = sorted(self.images_dir.glob('*.png'))
        if class_mapping is not None:
            self.mapping = class_mapping
        else:
            self.mapping = {i:i for i in range(CONFIG['num_classes'])}
        max_val = max(max(self.mapping.keys()), CONFIG['num_classes'])
        self.lut = np.full(max_val+1, self.IGNORE_INDEX, dtype=np.uint8)
        for old_val, new_val in self.mapping.items():
            self.lut[old_val] = new_val

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.name
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:,:,0]
        mask_remap = np.where(mask < len(self.lut), self.lut[mask], self.IGNORE_INDEX)
        if self.transform:
            transformed = self.transform(image=image, mask=mask_remap)
            image, mask_remap = transformed['image'], transformed['mask']
        return image, mask_remap.long()

# ======================
# Augmentations
# ======================
def get_training_augmentation(img_size=CONFIG['img_size']): 
    return A.Compose([
        A.Resize(img_size,img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2,0.2,p=0.5),
        A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ToTensorV2()
    ])

def get_validation_augmentation(img_size=CONFIG['img_size']): 
    return A.Compose([
        A.Resize(img_size,img_size),
        A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ToTensorV2()
    ])

# ======================
# Loss & metrics
# ======================
def calculate_iou(pred, target, num_classes=11, ignore_index=11):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    valid = target != ignore_index
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred==cls)&valid
        target_mask = (target==cls)&valid
        inter = np.logical_and(pred_mask,target_mask).sum()
        union = np.logical_or(pred_mask,target_mask).sum()
        if union>0: ious.append(inter/union)
    return np.mean(ious) if ious else 0.0

def get_combined_loss(ignore_index):
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
    def combined_loss(y_pred,y_true):
        return 0.5*ce_loss(y_pred,y_true)+0.5*dice_loss(y_pred,y_true)
    return combined_loss

# ======================
# Training
# ======================
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_iou = 0,0
    pbar = tqdm(loader, desc=f"Train E{epoch}", ncols=120)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        preds = outputs.argmax(1)
        iou = calculate_iou(preds,masks)
        total_loss+=loss.item()
        total_iou+=iou
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated()/1024**2
            pbar.set_postfix({'loss':f'{loss.item():.4f}','mIoU':f'{iou:.4f}','GPU':f'{mem:.0f}MB'})
        else:
            pbar.set_postfix({'loss':f'{loss.item():.4f}','mIoU':f'{iou:.4f}'})
    return total_loss/len(loader), total_iou/len(loader)

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss,total_iou=0,0
    pbar = tqdm(loader, desc="Val", ncols=120)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        preds = outputs.argmax(1)
        iou = calculate_iou(preds,masks)
        total_loss+=loss.item()
        total_iou+=iou
        pbar.set_postfix({'loss':f'{loss.item():.4f}','mIoU':f'{iou:.4f}'})
    return total_loss/len(loader), total_iou/len(loader)

# ======================
# Main
# ======================
def main():
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    bdh_config = BDHConfig()
    model = BDH_UNet(bdh_config, num_classes=CONFIG['num_classes']+1, img_size=CONFIG['img_size'], patch_size=CONFIG['patch_size']).to(CONFIG['device'])

    # ----- Loading the best model from previous training -----
    if os.path.exists(CONFIG['pretrained_model']):
        print(f"🔄 Loading pretrained model from {CONFIG['pretrained_model']}")
        checkpoint = torch.load(CONFIG['pretrained_model'], map_location=CONFIG['device'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️ Pretrained model not found. Training from scratch.")

    # Data
    train_dataset = CamVidDataset(CONFIG['data_dir'], split='train', transform=get_training_augmentation())
    val_dataset = CamVidDataset(CONFIG['data_dir'], split='val', transform=get_validation_augmentation(), class_mapping=train_dataset.mapping)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    criterion = get_combined_loss(CONFIG['ignore_index'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_iou = 0
    history = defaultdict(list)

    for epoch in range(1, CONFIG['epochs']+1):
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'], epoch)
        val_loss, val_iou = val_epoch(model, val_loader, criterion, CONFIG['device'])
        scheduler.step()
        print(f"\nEpoch {epoch}/{CONFIG['epochs']} Train L={train_loss:.4f} mIoU={train_iou:.4f} | Val L={val_loss:.4f} mIoU={val_iou:.4f}")

        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_iou)

        if val_iou>best_iou:
            best_iou = val_iou
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'val_miou':val_iou,'config':CONFIG}, f"{CONFIG['checkpoint_dir']}/best_ft_{val_iou:.3f}.pth")
            print(f"✅ Saved best model: {CONFIG['checkpoint_dir']}/best_ft_{val_iou:.3f}.pth")

    with open(f"{CONFIG['checkpoint_dir']}/training_history.json",'w') as f:
        json.dump(history,f,indent=2)
    print(f"\n🏁 Fine-tuning done! Best mIoU: {best_iou:.4f}")

if __name__=="__main__":
    main()
