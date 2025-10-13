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

## Current Project Status

The project is in an active research phase. We are currently focused on training and evaluating the `Vision-BDH` model from scratch on the standard **CIFAR-10** dataset to validate its ability to learn visual features.

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

3.  **Run the main training script:**
    ```bash
    python main.py
    ```
    The script will automatically download the CIFAR-10 dataset, configure the `Vision-BDH` model, and begin the training process, saving checkpoints to the `checkpoints/` directory.

---

## Next Steps in Research

Upon completion of the current training phase, our next objectives are:

1.  **Analyze the Results:** Perform a thorough evaluation of the final accuracy and learning curves.
2.  **Visualize the Model:** Investigate what the model has learned by visualizing its input filters, positional embeddings, and attention maps.
3.  **Benchmark:** Conduct a fair, controlled experiment by comparing `Vision-BDH` against a **ViT-Tiny** baseline, trained under identical conditions, to assess the relative performance of the two architectures.

⭐ **Star this repository** to follow our research progress on this fascinating new architecture!