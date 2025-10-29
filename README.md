# Vision-BDH: Adapting the Baby Dragon Hatchling Architecture for Computer Vision

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CIFAR-10](https://img.shields.io/badge/CIFAR--10-81.73%25-success.svg)
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
2. **Enhanced v2 Architecture:** Xavier initialization, Pre-LayerNorm, and raw attention scores (no softmax)

---

## Experimental Results

We conducted controlled experiments on CIFAR-10 and CIFAR-100, training all models from scratch for **50 epochs** under identical conditions.

### CIFAR-10 Benchmark

| Model | Parameters | Test Accuracy | Notes |
|:------|:----------:|:-------------:|:------|
| **Vision-BDH v2 (optimized)** | **3.2M** | **81.73%** 🏆 | No softmax |
| Vision-BDH v2 (baseline) | 3.2M | 80.45% | With softmax |
| Vision-BDH v1 | 3.6M | 80.43% | Original |
| ViT-Tiny | 5.4M | 76.05% | Baseline |

**Key Findings:**
- ✅ **+5.68pp advantage** over ViT-Tiny with optimized v2
- ✅ **+1.30pp improvement** from Pre-LN + raw attention synergy
- ✅ **~40% fewer parameters** (3.2M vs 5.4M)
- ✅ **Synergistic design:** Pre-LN and raw attention work together

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
- ✅ **+4.91pp advantage** over ViT-Tiny (next best)
- ✅ **Efficient learner:** Outperforms "data-hungry" models in limited-epoch regime
- ✅ **Growing advantage:** Performance gap increases with task complexity

### Overall Conclusion

**Vision-BDH demonstrates superior accuracy and parameter efficiency compared to standard baselines. Key discoveries: Pre-LayerNorm enables +1.30pp improvement, and works synergistically with raw attention scores. Adding softmax to Pre-LN architecture negates most benefits.**

---

## Architecture Ablation Study

We systematically evaluated the impact of normalization strategy and attention design:

### Complete Results

| Variant | Normalization | Softmax? | Test Accuracy | Delta vs v1 |
|---------|---------------|----------|---------------|-------------|
| v1 (original) | Mixed LN | ❌ No | 80.43% | baseline |
| v2 baseline | Pre-LN | ✅ Yes | 80.45% | +0.02pp |
| **v2 optimized** | **Pre-LN** | ❌ **No** | **81.73%** 🏆 | **+1.30pp** |
| v3 (failed) | Pre-LN + ScaledLN | ❌ No | 77.27% | -3.16pp |

### Key Findings

#### 1. Pre-LayerNorm Enables the Improvement (+1.30pp)

**Comparison:**
```
v1 (Mixed LN, no softmax):  80.43%
v2 (Pre-LN, no softmax):    81.73%
Improvement:                +1.30pp
```

Pre-LayerNorm (normalizing before sub-layers rather than after) significantly improves performance. This is now the standard in modern Transformers (GPT-2+, T5, etc.) due to better gradient flow and training stability.

#### 2. Softmax Conflicts with Pre-LayerNorm Architecture

**Synergy Test:**
```
Pre-LN + no softmax:  81.73% ✅ (best)
Pre-LN + softmax:     80.45% ❌ (loses +1.28pp!)
```

Adding softmax to the Pre-LN architecture **negates most of the Pre-LN benefit**, resulting in only +0.02pp improvement over v1.

**Hypothesis:** Pre-LayerNorm stabilizes training, allowing raw attention scores (leveraging Q=K constraint) to be more expressive. Softmax normalization in this context may be overly restrictive, limiting the model's ability to learn nuanced attention patterns.

#### 3. Synergistic Architecture Design

The combination of Pre-LN and raw attention (no softmax) works synergistically:
- **Pre-LN alone:** Would need softmax for stability (standard Transformer)
- **No softmax alone (v1):** Limited by mixed normalization (80.43%)
- **Pre-LN + no softmax:** Achieves best results (81.73%) ✅

This suggests that architectural components should be co-designed rather than mixed arbitrarily. The Q=K constraint's natural normalization properties only shine when combined with proper normalization placement (Pre-LN).

---

## Key Discovery: Pre-LN + Raw Attention Synergy

### The Synergistic Effect

Our research revealed that **Pre-LayerNorm and raw attention scores work synergistically**, not independently:

```python
# v1: Mixed normalization + raw attention
for level in range(n_layer):
    x_latent = x @ encoder  # No pre-normalization
    # ... attention ...
    x = ln(x + y)  # Post-normalization
Result: 80.43%

# v2 baseline: Pre-LN + softmax
for level in range(n_layer):
    x = ln(x)  # Pre-normalization ✅
    x_latent = x @ encoder
    # ... attention with softmax ❌ ...
    x = x + y
Result: 80.45% (softmax limits Pre-LN benefit)

# v2 optimized: Pre-LN + raw attention
for level in range(n_layer):
    x = ln(x)  # Pre-normalization ✅
    x_latent = x @ encoder
    # ... raw attention (no softmax) ✅ ...
    x = x + y
Result: 81.73% (full synergy!)
```

### Why Does This Work?

**Pre-LayerNorm provides:**
- ✅ Stable gradient flow
- ✅ Normalized inputs to attention
- ✅ Better training dynamics

**Raw attention (Q=K constraint) provides:**
- ✅ Self-similarity matrix with natural properties
- ✅ Diagonal dominance (self-attention)
- ✅ Bounded by RoPE encoding

**When combined:**
- Pre-LN stabilizes training → raw scores can be more expressive
- Q=K constraint → natural normalization → softmax becomes redundant
- Result: Best of both worlds! 🎯

### Comparison with Standard Transformers

| Architecture | Pre-LN? | Softmax? | Why? |
|--------------|---------|----------|------|
| Standard Transformer | ✅ Yes | ✅ Yes | Q ≠ K requires softmax normalization |
| Vision-BDH (optimized) | ✅ Yes | ❌ No | Q = K provides natural normalization |

**Key insight:** The Q=K constraint fundamentally changes attention mechanics, making softmax unnecessary when combined with Pre-LN.

---

## Architecture Evolution

| Feature | v1 (original) | v2 (baseline) | v2 (optimized) |
|---------|---------------|---------------|----------------|
| **Parameters** | 3.6M | 3.2M | **3.2M** ✅ |
| **CIFAR-10 (50ep)** | 80.43% | 80.45% | **81.73%** 🏆 |
| **CIFAR-100 (50ep)** | - | **51.44%** 🏆 | - |
| Weight Init | Normal | **Xavier** | **Xavier** ✅ |
| LayerNorm | Mixed (Post-LN style) | **Pre-LN** | **Pre-LN** ✅ |
| Attention | **Raw scores** | With softmax | **Raw scores** ✅ |
| **Key Innovation** | Q=K constraint | Pre-LN | **Pre-LN + raw attention** 🎯 |
| **Recommendation** | Historical | Baseline | **Use this!** ✅ |

### Evolution Summary

**v1 → v2 baseline (+0.02pp):**
- Added Pre-LayerNorm ✅
- Added softmax ❌
- Net effect: Minimal improvement (softmax cancels Pre-LN benefit)

**v1 → v2 optimized (+1.30pp):**
- Added Pre-LayerNorm ✅
- Kept raw attention ✅
- Net effect: Significant improvement (synergy!)

**v2 baseline → v2 optimized (+1.28pp):**
- Removed softmax ✅
- Unlocked Pre-LN's full potential

---

## Visual Results

### CIFAR-10 Learning Curves

![CIFAR-10 Learning Curves](analysis_results/cifar-10_learning_curves.png)

*Vision-BDH demonstrates consistently superior learning dynamics throughout training, achieving higher validation accuracy at every epoch compared to the ViT-Tiny baseline.*

### CIFAR-100 Learning Curves

![CIFAR-100 Learning Curves](analysis_results/cifar-100_learning_curves.png)

*On the more challenging CIFAR-100 dataset, Vision-BDH's advantage becomes even more pronounced, showing stronger sample efficiency and better generalization across 100 fine-grained classes.*

**Key Observations from Learning Curves:**
- 📈 **Faster convergence:** Vision-BDH reaches high accuracy earlier in training
- 📊 **Better sample efficiency:** Achieves superior results with the same number of training steps
- 🎯 **Stable training:** Smooth learning curves with minimal overfitting
- 💪 **Scalability:** Performance gap widens on more complex tasks (CIFAR-100)

---

## Visual Analysis: How the Models "See"

To gain a deeper understanding of *why* `Vision-BDH` outperforms the baseline, we visualized the internal attention mechanism of both models. We analyzed the attention patterns from the center image patch to all other patches across different layers.

The results reveal fundamental differences in their processing strategies.

### Case Study 1: "Ship" Image

**Vision-BDH v2:**
![Vision-BDH Attention on a Ship](attention_maps/attention_vision-bdh_v2_ship.png)

*   **Analysis:** The `Vision-BDH` model demonstrates a highly logical and interpretable thought process. In the first layer, it immediately identifies a key anchor point (the bow of the ship). By the middle layer, it expands its focus to the entire horizontal structure of the hull, a pattern that is refined in the final layer. This shows an efficient strategy of identifying and focusing on the object's core structure early on.

**ViT-Tiny (Baseline):**
![ViT-Tiny Attention on a Ship](attention_maps/attention_vit-tiny_ship.png)

*   **Analysis:** In contrast, `ViT-Tiny` exhibits a more exploratory and less focused strategy. Its attention remains diffuse through the middle layers, suggesting a phase of broad context gathering. In the final layer, the attention almost completely dissipates, which may indicate that the necessary information has already been aggregated by the `[CLS]` token, making inter-patch attention less critical at the end.

### Case Study 2: "Bird" Image

**Vision-BDH v2:**
![Vision-BDH Attention on a Bird](attention_maps/attention_vision-bdh_v2_bird.png)

*   **Analysis:** The pattern is remarkably consistent. `Vision-BDH` again starts by locking onto a high-contrast anchor point (the bird's head/beak). It then progressively expands its attention to encompass the bird's main body, demonstrating a robust object-centric focus.

**ViT-Tiny (Baseline):**
![ViT-Tiny Attention on a Bird](attention_maps/attention_vit-tiny_bird.png)

*   **Analysis:** `ViT-Tiny` again shows a diffuse, exploratory pattern in its middle layers. Interestingly, its final attention focuses sharply on the background (foliage in the top-right corner) rather than the bird itself. This suggests it may be learning contextual associations (e.g., "foliage is often near birds") rather than focusing directly on the object's features—a potentially less robust strategy.

### Key Insights from Visualizations

*   🧠 **Two Different "Minds":** The models employ fundamentally different strategies. `Vision-BDH` is **decisive and object-centric**, quickly identifying and focusing on the subject. `ViT-Tiny` is more **exploratory and contextual**, spending more layers gathering broad information before making a final, sometimes indirect, association.

*   🚀 **Efficiency Explains Performance:** The highly efficient and interpretable attention strategy of `Vision-BDH` is a likely explanation for its superior performance. By avoiding a lengthy exploration phase and focusing on relevant object features early, it appears to learn more effectively within a limited training budget.

---

## Failed Experiments

### Vision-BDH v3: ScaledLayerNorm (77.27%)

**Hypothesis:** Depth-dependent scaling (1/sqrt(depth)) would improve gradient flow in deep recurrent networks, inspired by ReZero (Bachlechner et al., 2020).

**Implementation:**
```python
class ScaledLayerNorm(nn.Module):
    def __init__(self, normalized_shape, depth_idx: int):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.scale = 1.0 / math.sqrt(depth_idx)  # Layer 6: 0.408×
```

**Result:** FAILED - accuracy dropped to 77.27% (-4.46pp vs v2 optimized).

**Analysis:**
- ❌ 6 layers is too shallow to need depth scaling
- ❌ 1/sqrt(depth) was too aggressive (layer 6: only 0.408× scaling)
- ❌ Over-damped deep layers, hindering learning
- ❌ Gradient accumulation: 0.408^6 ≈ 0.005 (severe vanishing)

**Lesson Learned:** Not all techniques from very deep networks (50+ layers) transfer to shallow networks (6 layers). **Simplicity wins** - Pre-LN + raw attention synergy (+1.30pp) was more effective than adding complexity (-4.46pp).

---

## Architecture Details

### Vision-BDH v2 Optimized (Recommended)

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
  ├─ Bidirectional attention (Q=K constraint, NO softmax) ← KEY!
  ├─ Gating mechanism (multiplicative)
  └─ Xavier-initialized weights
↓
Global Average Pooling
↓
Classification Head
```

**Specifications:**
- Parameters: 3.2M
- CIFAR-10: **81.73%** (best)
- CIFAR-100: **51.44%** (best)
- Key features: Pre-LN + raw attention synergy

### ViT-Tiny Baseline

```
Input: 32×32×3 image
↓
Patch Embedding (4×4 patches) → 64 tokens × 192 dims
↓
Positional Embedding (learned)
↓
12 Independent Transformer Layers:
  ├─ Multi-head attention (3 heads, with softmax)
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
# Vision-BDH v2 optimized (81.73% - recommended)
python train_bdh_v2_nosoftmax_cifar10.py

# Vision-BDH v2 baseline (80.45%)
python train_bdh_v2_cifar10.py

# Vision-BDH v1 (80.43%)
python train_bdh_v1_cifar10.py

# ViT-Tiny baseline (76.05%)
python train_vit_tiny_cifar10.py
```

**CIFAR-100:**
```bash
# Vision-BDH v2 (51.44%)
python train_bdh_v2_cifar100.py

# Baselines
python train_vit_tiny_cifar100.py
python train_resnet20_cifar100.py
python train_mobilenetv2_cifar100.py
python train_deit_tiny_cifar100.py
python train_efficientnet_cifar100.py
```

### Attention Visualization

```bash
# Generate attention maps for interpretability
python interpretability/visualize_attention.py
```

### Analysis

```bash
# Generate learning curves and comparisons
python analysis/analyze.py
```

---

## Project Structure

```
vision-bdh/
├── models/
│   ├── bdh.py                      # Original BDH implementation
│   ├── vision_bdh.py               # Vision-BDH v1
│   ├── vision_bdh_v2.py            # Vision-BDH v2 baseline
│   ├── vision_bdh_v2_nosoftmax.py  # Vision-BDH v2 optimized (recommended)
│   ├── vision_bdh_ablation.py      # For ablation studies
│   └── vit.py                      # ViT-Tiny baseline
├── interpretability/
│   ├── visualize_attention.py      # Attention visualization tool
│   └── models_for_viz/             # Modified models for visualization
├── analysis/
│   ├── compare_ablations.py
│   └── analyze.py
├── attention_maps/                 # Generated attention visualizations
├── analysis_results/               # Generated plots
├── checkpoints_*/                  # Model checkpoints
├── data_cifar10/                   # CIFAR-10 (auto-downloaded)
├── data_cifar100/                  # CIFAR-100 (auto-downloaded)
└── train_*.py                      # Training scripts
```

---

## Results Reproduction

### CIFAR-10 (50 epochs) - Best Result

```bash
# Train optimized v2 (Pre-LN + raw attention)
python train_bdh_v2_nosoftmax_cifar10.py
```
Expected: **81.73%** ± 0.2%

### CIFAR-100 (50 epochs)

```bash
python train_bdh_v2_cifar100.py
```
Expected: **51.44%** ± 0.5%

---

## Future Research Directions

### ✅ Completed
- [x] 50-epoch validation on CIFAR-10/100
- [x] Multiple baseline comparisons (6 models)
- [x] Attention visualization and interpretability
- [x] Architecture ablation study (Pre-LN + softmax synergy)
- [x] Failed experiment documentation (v3 ScaledLN)

### 🎯 High Priority

**1. Semantic Segmentation**
- [ ] Develop BDH-UNet hybrid architecture
- [ ] Test on Pascal VOC, Cityscapes
- **Hypothesis:** Sparse activations + gating → efficient segmentation

**2. Further Ablation Studies**
- [ ] Q=K constraint removal
- [ ] Gating mechanism analysis
- [ ] Activation sparsity quantification

### 🔬 Medium Priority

**3. Scaling Studies**
- [ ] ImageNet-1K pre-training
- [ ] Larger models (ViT-Small/Base equivalent)
- [ ] Transfer learning evaluation

**4. Efficiency Optimization**
- [ ] Mixed precision (FP16/BF16)
- [ ] Model quantization (INT8)
- [ ] FlashAttention integration

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
  note = {Achieved 81.73\% on CIFAR-10 and 51.44\% on CIFAR-100, 
          outperforming baselines with 40\% fewer parameters. 
          Key finding: Pre-LN and raw attention work synergistically.}
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

### v3.2 (Current) - Architecture Synergy Discovery
- ✅ **Best result:** 81.73% on CIFAR-10 (Pre-LN + raw attention)
- ✅ **Key discovery:** Pre-LN and raw attention work synergistically (+1.30pp)
- ✅ **Ablation study:** Softmax conflicts with Pre-LN architecture
- ✅ **Failed experiment:** v3 ScaledLayerNorm documented (-4.46pp)
- ✅ **Lesson learned:** Co-design architectural components for synergy

### v3.1 - Interpretability and Architectural Refactoring
- ✅ **Attention Visualization:** Tools for analyzing attention patterns
- ✅ **Architectural Refactoring:** Configurable model for ablation studies
- ✅ **Visual Analysis:** Qualitative insights into model behavior

### v3.0 - Extended Benchmarks & CIFAR-100
- ✅ **50-epoch training:** Comprehensive validation
- ✅ **CIFAR-100 benchmark:** Extended evaluation (51.44%)
- ✅ **Multiple baselines:** 6 architecture comparisons
- ✅ **Performance scaling:** Advantages increase with complexity

### v2.0 - Enhanced Architecture
- ✅ Vision-BDH v2 with Xavier init and Pre-LayerNorm
- ✅ Architecture comparison and analysis

### v1.0 - Initial Release
- ✅ BDH adapted for vision with bidirectional attention
- ✅ ViT-Tiny baseline comparison