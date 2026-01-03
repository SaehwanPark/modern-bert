# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **from-scratch PyTorch reimplementation of ModernBERT** that achieves logit-level parity (absolute error < 1 × 10⁻⁵) with the official HuggingFace pretrained weights. The codebase is a clean-room implementation (no code copied from the original) designed for architectural fidelity, numerical correctness, and code clarity.

## Development Commands

### Setup
```bash
# Install dependencies (uses uv package manager)
uv sync
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test_model.py
uv run pytest test_integration.py

# Run with verbose output
uv run pytest -v
```

### Demo
```bash
# Run side-by-side comparison with HuggingFace model
uv run demo.py
```

## Code Style

**Indentation**: Always use 2 spaces for indentation. Do not use tabs or 4 spaces.

## Critical Architecture Details

### Attention Mechanism: Alternating Global/Local Pattern
The model does NOT use uniform attention across all layers. Instead:
- **Every 3rd layer (0, 3, 6, 9...)**: Global attention (full sequence attention)
- **All other layers**: Sliding Window Local attention with 128-token bidirectional window (64 left/64 right)

This alternating pattern is defined in `src/layers.py:33` via `layer_id % config.global_attn_every_n_layers == 0`.

### RoPE (Rotary Position Embeddings): Dual Scheduling
The model uses TWO different RoPE base frequencies (θ):
- **Global layers**: θ = 160,000 (for long-range dependencies)
- **Local layers**: θ = 10,000 (for local context)

RoPE caches are precomputed in `src/model.py:46-61` and selected per-layer in `src/model.py:71-76`.

### Normalization: Bias-Free LayerNorm (NOT RMSNorm)
All normalization layers use **bias-free LayerNorm** with mean subtraction:
```python
nn.LayerNorm(hidden_size, eps=1e-5, bias=False)
```
This is critical for numerical precision. Do NOT use RMSNorm despite the lack of bias terms in the state_dict.

### Skip-First Pre-Norm Pattern
Layer 0 skips the attention normalization (`attn_norm = None` in `src/model.py:13-17`) because the embedding layer already provides normalized output. All subsequent layers include bias-free LayerNorm before attention.

### GeGLU MLP Activation
The MLP uses a Gated Linear Unit pattern (`src/layers.py:64-67`):
1. Input projection (`Wi`) outputs 2 × intermediate_size
2. Split output in half: `gate` and `val`
3. Return: `Wo(GELU(gate) * val)`

### MLM Head Architecture
The Masked Language Model prediction head follows this exact sequence (`src/model.py:95-96`):
```
Linear (no bias) → GELU → LayerNorm (no bias) → Decoder (with bias)
```
The decoder weights are tied to `tok_embeddings` for parameter efficiency.

## Codebase Structure

```
src/
├── config.py      # ModernBertConfig dataclass with all hyperparameters
├── layers.py      # RoPE, Attention (Global/Local), GeGLU MLP
├── model.py       # ModernBertModel, ModernBertForMaskedLM
└── loader.py      # Weight loading utilities from HuggingFace

test_model.py          # Unit tests: forward pass shapes, skip-first behavior
test_integration.py    # Integration test: logit comparison vs HuggingFace
demo.py               # Interactive comparison script with masked predictions
```

## Key Implementation Notes

### State Dict Compatibility
The implementation is designed for **strict weight loading** from HuggingFace checkpoints:
```python
custom_model.load_state_dict(hf_model.state_dict(), strict=True)
```
All keys and shapes must match exactly. See `test_integration.py:26` for validation.

### Numerical Precision Requirements
Integration tests verify logit differences < 1 × 10⁻⁵. Three architectural details were critical to achieving this tolerance:
1. Mean-subtraction LayerNorm (not RMSNorm)
2. Alternating attention pattern (not uniform)
3. Dual RoPE scheduling (not single θ value)

See `docs/post-moretem.md` for the full debugging journey.

### Functional Implementation
The codebase uses PyTorch's native `F.scaled_dot_product_attention` for performance and compatibility with `torch.compile`. Sliding window masks are created as boolean tensors in `src/layers.py:15-22`.

## Model Configuration (Base)

```python
vocab_size: 50,368
hidden_size: 768
intermediate_size: 1,152
num_hidden_layers: 22
num_attention_heads: 12
max_position_embeddings: 8,192
global_attn_every_n_layers: 3
local_attention: 128  # window size
global_rope_theta: 160,000.0
local_rope_theta: 10,000.0
```

## Common Pitfalls to Avoid

1. **Do not assume uniform attention**: The alternating pattern is core to ModernBERT's architecture.
2. **Do not use RMSNorm**: Despite no bias in state_dict, LayerNorm's mean subtraction is required.
3. **Do not skip the first layer's special handling**: Layer 0 has no `attn_norm`, which affects state_dict key structure.
4. **Do not use a single RoPE theta**: Global and local layers require different base frequencies.
5. **Do not modify weight-tying**: The decoder must share weights with `tok_embeddings`.

## Testing Philosophy

- **Unit tests** (`test_model.py`): Validate shapes and architectural constraints (skip-first pattern).
- **Integration tests** (`test_integration.py`): Ensure numerical parity with official HuggingFace model.
- **Demo script** (`demo.py`): Provide human-readable verification via masked token predictions.

All tests must pass before modifying core architecture. The strict state_dict loading in integration tests serves as a structural invariant.
