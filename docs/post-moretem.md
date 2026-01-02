This post-mortem documents the architectural alignment and debugging process required to successfully replicate the original **ModernBERT** model and achieve logit-level parity (`<1E-5` difference) with the official HuggingFace pretrained weights.

### 1. Root Cause of Initial Discrepancies

The initial integration attempts resulted in a massive logit difference (around 53.4). Three "hidden" architectural features were identified as the primary causes:

* **Normalization Logic (Mean Subtraction):** While the `state_dict` lacked bias terms for normalization, using `RMSNorm` proved incorrect. ModernBERT uses **bias-free LayerNorm**. Unlike RMSNorm, LayerNorm subtracts the mean of the features before scaling. This mean subtraction is critical for numerical alignment across 22 layers.
* **Alternating Attention (Global vs. Local):** The model does not use a uniform attention mechanism. It alternates between **Global** attention (every 3rd layer: 0, 3, 6...) and **Sliding Window (Local)** attention.
* **Rotary Positional Embedding (RoPE) Scheduling:** The model employs two different RoPE base frequencies $\theta$. Global layers use $\theta = 160000$ for long-range dependency, while local layers use $\theta = 10000$.

### 2. Final Architectural Specifications

To pass the integration tests, the following modules were developed to match the original `FlexBert` codebase:

#### Attention & Positional Encoding

* **Mechanism:** Multi-head self-attention utilizing `torch.nn.functional.scaled_dot_product_attention` for performance.
* **Sliding Window:** Local layers restrict attention to a **128-token bidirectional window** (64 tokens left, 64 tokens right).
* **RoPE:** Positional information is injected at every layer by rotating the query and key tensors using precomputed  and  caches.

#### Layer Topology

* **Skip-First Pre-Norm:** To match the `state_dict` keys (e.g., `model.layers.0.attn.Wqkv.weight` without an `attn_norm`), the first layer skips the attention normalization. This is because the embedding layer already provides a normalized output immediately prior to Layer 0.
* **GeGLU MLP:** The MLP (Feed-Forward) layer uses a Gated Linear Unit. The input projection (`Wi`) output is split into two halves; the first half is passed through a `GELU` activation and multiplied by the second half.

#### Prediction Head (Masked LM)

* **MLM Sequence:** The MLM head follows a specific sequence: `Linear` (no bias)  `GELU`  `LayerNorm` (no bias).
* **Weight Tying:** The `decoder` weights are tied to the `tok_embeddings` weights to reduce parameter count and improve regularization.

### 3. Integration Test Results

The final implementation passed the integration suite with the following metrics:

* **State Dict Loading:** `strict=True` confirmed 100% key and shape compatibility.
* **Logit Verification:** The maximum absolute difference between the custom implementation and the HF model was **$8.46 \times 10^{-6}$**, well within the `1E-5` tolerance required for floating-point accumulation variances.
* **Performance:** The functional approach using PyTorch's native SDPA ensures compatibility with `torch.compile` and various acceleration backends (CUDA/Triton/etc).

### 4. Codebase Organization

The project is now structured for long-term maintenance:

* `src/config.py`: Contains `ModernBertConfig` with all 28+ hyper-parameters.
* `src/layers.py`: Contains the functional implementations of RoPE, GeGLU, and Alternating Attention.
* `src/model.py`: Defines the `ModernBertModel` and `ModernBertForMaskedLM` graphs.
* `test_integration.py`: Provides the definitive cross-check against the `transformers` library.