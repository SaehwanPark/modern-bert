# ModernBERT PyTorch Reimplementation

A modular, from-scratch PyTorch reimplementation of the **ModernBERT** architecture described in
[*ModernBERT: A Modernized Transformer for Efficient Long-Context Modeling*](https://arxiv.org/abs/2412.13663).

This project targets architectural fidelity, numerical correctness, and code clarity within a modern Python ecosystem (`PyTorch 2.9.1`, `Python 3.13`, and `uv`). The implementation achieves **logit-level parity** with the officially released pretrained weights (absolute error < 1 × 10⁻⁵) by faithfully reproducing ModernBERT’s distinctive architectural choices.

---

## Key Features

* **Alternating Attention**
  Global attention every third layer, with Local (sliding-window) attention in all other layers.

* **Dual RoPE Scheduling**
  Rotary positional embeddings with θ = 160k for Global layers and θ = 10k for Local layers.

* **GeGLU Activation**
  Modern gated MLP blocks using GeGLU for improved expressivity.

* **Skip-First Pre-Norm**
  Normalization flow aligned with official parameterization and checkpoint structure.

* **Bias-Free LayerNorm**
  Mean-subtraction normalization without additive bias for numerical precision.

---

## Architecture Summary

| Component       | Implementation                                        |
| --------------- | ----------------------------------------------------- |
| **Attention**   | Alternating Global and Sliding-Window Local attention |
| **Window Size** | 128 tokens bidirectional (64 left / 64 right)         |
| **Positional**  | Rotary Positional Embeddings (RoPE)                   |
| **Output Head** | Tied-weight decoder with GELU prediction head         |

---

## Relationship to the Original ModernBERT Implementation

This repository is an **independent, clean-room reimplementation** of the ModernBERT architecture.

* All source code in this repository was written **entirely from scratch**
* No files were copied, forked, modified, or adapted from the original implementation
* No training scripts, configuration files, or internal utilities from the original codebase are included

The following original resources were consulted **solely for architectural understanding and validation**:

* Original paper: [https://arxiv.org/abs/2412.13663](https://arxiv.org/abs/2412.13663)
* Official codebase: [https://github.com/AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT)
* Official Hugging Face model: [https://huggingface.co/answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

This project exists to provide:

* A minimal and readable reference implementation
* Explicit modeling of ModernBERT’s architectural decisions
* A reproducible baseline that matches the published pretrained weights at the logit level

---

## Attribution

ModernBERT was originally developed by **Answer.AI** and collaborators.

This repository is **not affiliated with, endorsed by, or maintained by** the original authors.

---

## Getting Started

### Prerequisites

Dependencies are managed using `uv`:

```bash
uv sync
```

### Usage

Run the side-by-side comparison script to verify predictions against the Hugging Face reference model:

```bash
uv run demo.py
```

---

## Testing

Comprehensive unit and integration tests are provided to ensure functional and numerical parity:

```bash
uv run pytest
```

---

## License Note

This repository contains only original source code authored by the maintainers of this project.

While inspired by the ModernBERT architecture (originally released under the Apache License 2.0), no source code from the original implementation is included here.

