# Vision-BDH: Adapting the Baby Dragon Hatchling Architecture for Computer Vision

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CIFAR-10](https://img.shields.io/badge/CIFAR--10-80.45%25-success.svg)
![CIFAR-100](https://img.shields.io/badge/CIFAR--100-51.44%25-success.svg)

This project is a PyTorch-based research framework dedicated to adapting and exploring the novel **Baby Dragon Hatchling (BDH)** architecture for computer vision tasks.

The original BDH architecture was proposed for language modeling in:
**"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"**  
*Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz*  
**[arXiv:2509.26507](https://arxiv.org/abs/2509.26507)**

Our goal is to investigate whether the unique, bio-inspired, and efficiency-oriented features of BDH can offer advantages in image analysis.

---

## What is Vision-BDH?

`Vision-BDH` is not just another Vision Transformer (ViT). It's a hybrid architecture combining:

*   **ViT's "body":** Patch-based image processing as a sequence
*   **BDH's "soul":** Unique recurrent computational core with bio-inspired features

### Unique BDH Features Preserved

Our model preserves 4 out of 5 fundamental innovations from the original BDH architecture:

| Key BDH Feature | Preserved? | Description |
|:----------------|:-----------|:------------|
| **Shared Parameters** | ✅ Yes | Single "layer" reused multiple times (recurrent depth) |
| **Sparse Activations (ReLU)** | ✅ Yes | Sparse, non-negative representations mimicking neural activity |
| **Constrained Attention (Q=K)** | ✅ Yes | Simplified attention based on activation similarity |
| **Multiplicative Gating** | ✅ Yes | Gating mechanism instead of standard residuals |
| Byte-Level Processing | ❌ No (Adapted) | Replaced with patch embeddings for visual data |

### Key Modifications for Vision

1. **Bidirectional Attention:** Removed causal masking to analyze all image patches simultaneously
2. **Enhanced v2 Architecture:** Xavier initialization, optional softmax attention, Pre-LayerNorm for better stability

---

## Experimental Results

We conducted controlled experiments on CIFAR-10 and CIFAR-100, training all models from scratch for **50 epochs** under identical conditions.

### CIFAR-10 Benchmark

| Model | Parameters | Test Accuracy |
|:------|:----------:|:-------------:|
| **Vision-BDH v2** | **3.2M** | **80.45%** 🏆 |
| **Vision-BDH v1** | **3.6M** | **80.43%** |
| ViT-Tiny | 5.4M | 76.05% |

**Key Findings:**
- ✅ **+4.4pp advantage** over ViT-Tiny
- ✅ **~40% fewer parameters** (3.2M vs 5.4M)
- ✅ **Architecture robustness:** Both v1 and v2 achieve ~80.4%

### CIFAR-100 Benchmark (Extended Comparison)

| Model | Parameters | Test Accuracy |
|:------|:----------:|:-------------:|
| **Vision-BDH v2** | **3.2M** | **51.44%** 🏆 |
| ViT-Tiny | 5.7M | 46.53% |
| ResNet-20 | 0.27M | 45.62% |
| EfficientNet-B0 | 4.1M | 40.20% |
| DeiT-Tiny | 5.5M | 35.31% |
| MobileNetV2 | 2.3M | 33.83% |

**Key Findings:**
- ✅ **Dominant performance** across all baselines
- ✅ **+4.9pp advantage** over ViT-Tiny (next best)
- ✅ **Efficient learner:** Outperforms "data-hungry" models (EfficientNet, MobileNet) in limited-epoch regime
- ✅ **Growing advantage:** Performance gap increases with task complexity

### Overall Conclusion

**Vision-BDH demonstrates superior accuracy and parameter efficiency compared to standard CNN and Transformer baselines when trained from scratch. The architectural advantages become more pronounced as task complexity increases.**

---

## Architecture Evolution

| Feature | Vision-BDH v1 | Vision-BDH v2 |
|---------|---------------|---------------|
| **Parameters** | 3.6M | **3.2M** ✅ |
| **CIFAR-10 (50ep)** | 80.43% | **80.45%** 🏆 |
| **CIFAR-100 (50ep)** | - | **51.44%** 🏆 |
| Weight Init | Normal | **Xavier uniform** ✅ |
| LayerNorm | Post-encoder | **Pre-encoder (Pre-LN)** ✅ |
| Gradient Flow | Good | **Improved** ✅ |
| **Recommendation** | Historical reference | **Use for all new experiments** ✅ |

---

## Architecture Details

### Vision-BDH v2 (Recommended)

```
Input: 32×32×3 image
↓
Patch Embedding (4×4 patches) → 64 tokens × 192 dims
↓
Positional Embedding (learned)
↓
BDH Core (6 recurrent layers):
  ├─ Pre-LayerNorm (stability)
  ├─ Sparse projection (ReLU activation)
  ├─ Bidirectional attention (Q=K constraint)
  ├─ Gating mechanism (multiplicative)
  └─ Xavier-initialized weights
↓
Global Average Pooling
↓
Classification Head
```

**Specifications:**
- Parameters: 3.2M
- CIFAR-10: 80.45%
- CIFAR-100: 51.44%

### Vision-BDH v1

```
Input: 32×32×3 image
↓
Patch Embedding (4×4 patches) → 64 tokens × 192 dims
↓
Positional Embedding (learned)
↓
BDH Core (6 recurrent layers):
  ├─ Sparse projection (ReLU activation)
  ├─ Bidirectional attention (Q=K constraint)
  ├─ Gating mechanism (multiplicative)
  └─ Normal weight initialization
↓
Global Average Pooling
↓
Classification Head
```

**Specifications:**
- Parameters: 3.6M
- CIFAR-10: 80.43%

### ViT-Tiny Baseline

```
Input: 32×32×3 image
↓
Patch Embedding (4×4 patches) → 64 tokens × 192 dims
↓
Positional Embedding (learned)
↓
12 Independent Transformer Layers:
  ├─ Multi-head attention (3 heads)
  └─ Standard MLP (768 dims, 4× multiplier)
↓
Classification Head
```

**Specifications:**
- Parameters: 5.4-5.7M
- CIFAR-10: 76.05%
- CIFAR-100: 46.53%

---

## Training Configuration

All experiments used identical settings for fair comparison:

| Setting | Value |
|---------|-------|
| **Datasets** | CIFAR-10 / CIFAR-100 |
| **Epochs** | 50 (from scratch, no pre-training) |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Weight Decay** | 0.05 |
| **LR Schedule** | 500-step warmup + cosine decay |
| **Batch Size** | 32 |
| **Hardware** | Single NVIDIA RTX 4060 |

---

## Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
torchvision
matplotlib
pandas
timm (for DeiT baseline)
CUDA-capable GPU (recommended)
```

### Installation

```bash
# Clone repository
git clone https://github.com/takzen/vision-bdh.git
cd vision-bdh

# Create virtual environment (using uv recommended)
uv venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Install dependencies
uv pip install torch torchvision pandas matplotlib timm
```

### Training

**CIFAR-10:**
```bash
# Vision-BDH v2 (recommended)
python train_bdh_v2_cifar10.py

# Vision-BDH v1
python train_bdh_v1_cifar10.py

# ViT-Tiny baseline
python train_vit_tiny_cifar10.py
```

**CIFAR-100:**
```bash
# Vision-BDH v2
python train_bdh_v2_cifar100.py

# ViT-Tiny baseline
python train_vit_tiny_cifar100.py

# Additional baselines
python train_resnet20_cifar100.py
python train_mobilenetv2_cifar100.py
python train_deit_tiny_cifar100.py
python train_efficientnet_cifar100.py
```

All scripts will:
- Auto-download datasets
- Train for specified epochs
- Save checkpoints
- Report final accuracy

### Visualization

```bash
python analysis/analyze.py
```

Generates plots in `analysis_results/`:
- Learning curves
- Accuracy comparisons
- Parameter efficiency
- Training dynamics

---

## Project Structure

```
vision-bdh/
├── models/
│   ├── bdh.py                      # Original BDH implementation
│   ├── vision_bdh.py               # Vision-BDH v1
│   ├── vision_bdh_v2.py            # Vision-BDH v2 (recommended)
│   └── vit.py                      # ViT-Tiny baseline
├── analysis/
│   └── analyze.py                  # Visualization tools
├── analysis_results/               # Generated plots
├── checkpoints_v1_cifar10/         # Vision-BDH v1 checkpoints
├── checkpoints_v2_cifar10/         # Vision-BDH v2 CIFAR-10 checkpoints
├── checkpoints_v2_cifar100/        # Vision-BDH v2 CIFAR-100 checkpoints
├── checkpoints_vit_tiny_cifar10/   # ViT-Tiny CIFAR-10 checkpoints
├── checkpoints_vit_tiny_cifar100/  # ViT-Tiny CIFAR-100 checkpoints
├── checkpoints_resnet20_cifar100/  # ResNet-20 checkpoints
├── checkpoints_mobilenetv2_cifar100/   # MobileNetV2 checkpoints
├── checkpoints_deit_tiny_cifar100/     # DeiT-Tiny checkpoints
├── checkpoints_efficientnet_cifar100/  # EfficientNet-B0 checkpoints
├── data_cifar10/                   # CIFAR-10 (auto-downloaded)
├── data_cifar100/                  # CIFAR-100 (auto-downloaded)
├── train_bdh_v1_cifar10.py
├── train_bdh_v2_cifar10.py
├── train_bdh_v2_cifar100.py
├── train_vit_tiny_cifar10.py
├── train_vit_tiny_cifar100.py
├── train_resnet20_cifar100.py
├── train_mobilenetv2_cifar100.py
├── train_deit_tiny_cifar100.py
└── train_efficientnet_cifar100.py
```

---

## Results Reproduction

### CIFAR-10 (50 epochs)

1. Train Vision-BDH v2:
   ```bash
   python train_bdh_v2_cifar10.py
   ```
   Expected: 80.45% ± 0.2%

2. Train ViT-Tiny baseline:
   ```bash
   python train_vit_tiny_cifar10.py
   ```
   Expected: 76.05% ± 0.3%

3. Generate visualizations:
   ```bash
   python analysis/analyze.py
   ```

### CIFAR-100 (50 epochs)

1. Train Vision-BDH v2:
   ```bash
   python train_bdh_v2_cifar100.py
   ```
   Expected: 51.44% ± 0.5%

2. Train baselines (optional):
   ```bash
   python train_vit_tiny_cifar100.py      # 46.53%
   python train_resnet20_cifar100.py      # 45.62%
   python train_efficientnet_cifar100.py  # 40.20%
   ```

---

## Future Research Directions

### ✅ Completed
- [x] 50-epoch validation on CIFAR-10
- [x] Extended CIFAR-100 benchmark
- [x] Multiple baseline comparisons
- [x] v1 vs v2 architecture comparison

### 🎯 High Priority

**1. Semantic Segmentation (Primary Focus)**
- [ ] Develop BDH-UNet hybrid architecture
- [ ] Vision-BDH encoder + U-Net decoder
- [ ] Test on Pascal VOC, Cityscapes
- **Hypothesis:** Sparse activations + gating → efficient segmentation

**2. Architecture Analysis**
- [ ] Ablation studies (gating, Q=K, sparsity)
- [ ] Visualize attention patterns
- [ ] Analyze activation sparsity statistics
- [ ] Compare feature representations (CKA, SVCCA)

### 🔬 Medium Priority

**3. Scaling Studies**
- [ ] ImageNet-1K pre-training
- [ ] Larger models (ViT-Small/Base equivalent)
- [ ] Multi-scale training
- [ ] Transfer learning evaluation

**4. Efficiency Optimization**
- [ ] Mixed precision (FP16/BF16)
- [ ] Model quantization (INT8)
- [ ] FlashAttention integration
- [ ] Edge deployment optimization

### 💡 Long-term Goals

**5. Advanced Learning**
- [ ] Self-supervised pre-training (MAE)
- [ ] Fine-grained classification
- [ ] Few-shot learning

**6. New Applications**
- [ ] Object detection (DETR-style heads)
- [ ] Video understanding
- [ ] Multi-modal learning

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@software{pika2025visionbdh,
  author = {Krzysztof Pika},
  title = {Vision-BDH: Adapting Baby Dragon Hatchling for Computer Vision},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/takzen/vision-bdh},
  note = {Achieved 80.45\% on CIFAR-10 and 51.44\% on CIFAR-100, 
          outperforming baselines with 40\% fewer parameters}
}
```

Please also cite the original BDH paper:

```bibtex
@article{kosowski2024dragon,
  title={The Dragon Hatchling: The Missing Link between the Transformer 
         and Models of the Brain},
  author={Kosowski, Adrian and Uzna{\'n}ski, Przemys{\l}aw and 
          Chorowski, Jan and Stamirowska, Zuzanna and 
          Bartoszkiewicz, Micha{\l}},
  journal={arXiv preprint arXiv:2509.26507},
  year={2024}
}
```

---

## Acknowledgments

- Original BDH authors for the innovative architecture
- CIFAR-10/100 datasets by Krizhevsky, Nair, and Hinton
- PyTorch team for the deep learning framework
- ML research community for open-source tools

---

## Contributing

We welcome contributions in:
- 🐛 Bug fixes
- 📊 New experimental results
- 🔬 Architecture variants
- 📝 Documentation
- 🎨 Visualization tools
- ⚡ Performance optimizations

Please:
- Open issues for bugs/questions
- Submit pull requests
- Share experimental insights
- Join discussions about sparse transformers

---

## License

MIT License - See `LICENSE` file for details.

---

## Contact

- **Author:** Krzysztof Pika
- **GitHub:** [@takzen](https://github.com/takzen)
- **Project:** [vision-bdh](https://github.com/takzen/vision-bdh)

---

⭐ **Star** if you find this research interesting!  
🔔 **Watch** for updates on BDH for computer vision.  
🔥 **Fork** to experiment with modifications!

---

## Changelog

### v3.0 (Current) - Extended Benchmarks & CIFAR-100
- ✅ **50-epoch training:** Comprehensive validation on CIFAR-10
- ✅ **CIFAR-100 benchmark:** Extended evaluation (51.44%)
- ✅ **Multiple baselines:** Compared against 6 architectures
- ✅ **Architecture robustness:** v1 and v2 both achieve ~80.4% on CIFAR-10
- ✅ **Performance scaling:** Advantages increase with task complexity
- ✅ **Documentation:** Clean, organized project structure

### v2.0 - Enhanced Architecture
- ✅ Vision-BDH v2 with Xavier init and Pre-LayerNorm
- ✅ 30-epoch validation (v1: 79.54%, v2: 78.76%)
- ✅ Architecture comparison and analysis

### v1.0 - Initial Release
- ✅ BDH adapted for vision with bidirectional attention
- ✅ ViT-Tiny baseline comparison
- ✅ Initial CIFAR-10 results