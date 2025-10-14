# Vision-BDH: Adapting the Baby Dragon Hatchling Architecture for Computer Vision

This project is a PyTorch-based research framework dedicated to adapting and exploring the novel **Baby Dragon Hatchling (BDH)** architecture for computer vision tasks.

The original BDH architecture was proposed for language modeling in the paper:
**"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"**
*Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz*
**[arXiv:2509.26507](https://arxiv.org/abs/2509.26507)**

Our goal is to investigate whether the unique, bio-inspired, and efficiency-oriented features of BDH can offer advantages in the domain of image analysis.

---

## What is Vision-BDH?

`Vision-BDH` is not just another Vision Transformer (ViT). It's a new hybrid architecture that combines the best of both worlds:

*   **It borrows the "body" from ViT:** It adopts the proven method of "seeing" an image by splitting it into patches and processing them as a sequence.
*   **It borrows the "soul" from BDH:** It uses the unique, recurrent computational core from the BDH model to analyze this sequence, preserving its key architectural features.

### Unique BDH Features Preserved in `Vision-BDH`

Our model preserves 4 out of 5 of the fundamental innovations from the original BDH architecture:

| Key BDH Feature               | Preserved in `Vision-BDH`? | Description                                                                                    |
| :---------------------------- | :------------------------- | :--------------------------------------------------------------------------------------------- |
| **Shared Parameters**         | ✅ **Yes**                 | The same single "layer" is reused multiple times, creating a form of "recurrent depth."        |
| **Sparse Activations (ReLU)** | ✅ **Yes**                 | The model's internal representations are sparse and non-negative, mimicking neural activity.     |
| **Constrained Attention (`Q=K`)** | ✅ **Yes**                 | The attention mechanism is simplified, based on activation similarity rather than complex projections. |
| **Multiplicative Gating**     | ✅ **Yes**                 | Instead of standard residual connections (`x + F(x)`), the model uses gating (`x * y`).        |
| Byte-Level Processing         | ❌ **No** (Adapted)        | Replaced with a patch embedding mechanism, which is the appropriate equivalent for visual data. |

### Key Modification: Bidirectional Attention

The original BDH model used causal (unidirectional) attention, which is necessary for text generation. In `Vision-BDH`, **this constraint has been removed**. This allows the attention mechanism to analyze **all parts of the image simultaneously**. Every patch can "attend" to every other patch, which is fundamental for a holistic understanding of a visual scene.

---

## Experimental Results

We conducted a controlled experiment comparing `Vision-BDH` against a standard **ViT-Tiny** baseline on **CIFAR-10**, training both models from scratch under identical conditions.

### Benchmark Comparison

| Model | Parameters | Test Accuracy | Avg Train Loss (Epoch 10) | Configuration |
|-------|------------|---------------|---------------------------|---------------|
| **Vision-BDH** | 6.5M | **72.51%** 🏆 | 0.86 | 6 layers (recurrent), 192 dim, 6 heads |
| **ViT-Tiny** | 5.7M | 65.96% | 0.85 | 12 layers, 192 dim, 3 heads |

### Key Findings

✅ **Vision-BDH achieves +6.55 percentage points higher accuracy** than ViT-Tiny  
✅ **10% relative improvement** in classification performance  
✅ **Sparse activations + gating mechanism** prove effective for vision tasks  
⚠️ **Training time trade-off:** Vision-BDH is ~166x slower per epoch due to large MLP dimensions (24,576 vs 768)

### Training Configuration

Both models were trained with:
- **Dataset:** CIFAR-10 (32×32 RGB images, 10 classes)
- **Training:** 10 epochs from scratch, no pretraining
- **Optimizer:** AdamW (LR: 1e-4, weight decay: 0.05)
- **Schedule:** 500-step warmup + cosine decay
- **Batch size:** 32
- **Augmentation:** Random crop (0.8-1.0 scale) + horizontal flip

### Learning Curves

**Vision-BDH progression:**
- Epoch 1: 35.61% validation accuracy
- Epoch 2: 50.81% validation accuracy (+15.2pp)
- Epoch 10: 72.51% test accuracy

**ViT-Tiny progression:**
- Epoch 10: 65.96% test accuracy

---

## Architecture Details

### Vision-BDH Model

```
Input: 32×32×3 image
↓
Patch Embedding (4×4 patches) → 64 tokens of 192 dims
↓
Positional Embedding (learned)
↓
BDH Core (6 recurrent layers):
  - Sparse projection (ReLU activation)
  - Bidirectional attention (Q=K constraint)
  - Gating mechanism (x * y)
  - Large MLP (24,576 internal dims)
↓
Global Average Pooling
↓
Classification Head → 10 classes
```

**Total parameters:** ~6.5M (including large MLP: 128× embedding dim)

### ViT-Tiny Baseline

```
Input: 32×32×3 image
↓
Patch Embedding (4×4 patches) → 64 tokens of 192 dims
↓
Positional Embedding (learned)
↓
12 Transformer Layers:
  - Multi-head attention (3 heads)
  - Standard MLP (768 internal dims, 4× multiplier)
↓
Classification Head → 10 classes
```

**Total parameters:** ~5.7M

---

## Current Project Status

The project has successfully completed its initial research phase, demonstrating that BDH can be effectively adapted for computer vision tasks with competitive results.

### Completed Milestones

✅ **Architecture adaptation:** Successfully modified BDH for bidirectional vision tasks  
✅ **Baseline training:** Trained Vision-BDH from scratch on CIFAR-10  
✅ **Benchmarking:** Direct comparison with ViT-Tiny under identical conditions  
✅ **Results analysis:** Documented performance gains and trade-offs

### Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/takzen/vision-bdh.git
    cd vision-bdh
    ```

2.  **Install dependencies (using `uv`):**
    ```bash
    # Create and activate a virtual environment
    uv venv
    source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
    
    # Install packages
    uv pip install torch torchvision numpy
    ```

3.  **Train Vision-BDH:**
    ```bash
    python main.py
    ```

4.  **Train ViT-Tiny baseline (for comparison):**
    ```bash
    python train_vit_tiny.py
    ```

Both scripts will automatically download CIFAR-10, train the models, and save checkpoints to separate directories.

---

## Future Research Directions

Based on our initial findings, we identify several promising directions for future work:

### 1. Architecture Optimization
- **MLP size ablation:** Test different `mlp_internal_dim_multiplier` values (16, 32, 64, 128, 256)
- **Layer scaling:** Experiment with deeper models (8, 12, 16 recurrent layers)
- **Attention variants:** Explore alternatives to Q=K constraint for vision

### 2. Training Efficiency
- **Speed optimization:** Investigate techniques to reduce the 166× training time gap
- **Mixed precision training:** Implement FP16/BF16 for faster computation
- **Efficient attention:** Explore FlashAttention-style optimizations for BDH

### 3. Scaling Experiments
- **Larger datasets:** Evaluate on ImageNet, CIFAR-100
- **Larger models:** Scale to ViT-Small/Base sizes
- **Transfer learning:** Pre-train on ImageNet, fine-tune on downstream tasks

### 4. Analysis and Interpretability
- **Attention visualization:** Analyze what patterns BDH learns vs ViT
- **Sparsity analysis:** Quantify activation sparsity and its impact
- **Feature analysis:** Compare learned representations between architectures

---

## Citation

If you use this code or find our work helpful, please consider citing:
```bibtex
@software{vision-bdh-2025,
  author = {Krzysztof Pika},
  title = {Vision-BDH: Adapting Baby Dragon Hatchling for Computer Vision},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/takzen/vision-bdh}
}
```

And the original BDH paper:
```bibtex
@article{kosowski2025dragon,
  title={The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain},
  author={Kosowski, Adrian and Uzna{\'n}ski, Przemys{\l}aw and Chorowski, Jan and Stamirowska, Zuzanna and Bartoszkiewicz, Micha{\l}},
  journal={arXiv preprint arXiv:2509.26507},
  year={2025}
}
```
---

## Contributing

We welcome contributions! Whether it's:
- Bug fixes and code improvements
- New experimental results
- Architecture variants
- Documentation enhancements

Please feel free to open issues or submit pull requests.

---

## License

This project is released under the MIT License. See `LICENSE` file for details.

---

⭐ **Star this repository** if you find this research interesting! Follow for updates as we continue exploring the potential of BDH for computer vision.