# src/config.py
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModernBertConfig:
  vocab_size: int = 50368
  hidden_size: int = 768
  intermediate_size: int = 1152
  num_hidden_layers: int = 22
  num_attention_heads: int = 12
  norm_eps: float = 1e-5
  initializer_range: float = 0.02
  max_position_embeddings: int = 8192
  dropout_prob: float = 0.0
  attention_probs_dropout_prob: float = 0.0

  # ModernBERT-specific
  # Every 3rd layer is Global, others are Local (Sliding Window)
  global_attn_every_n_layers: int = 3
  # Sliding window size (128 means 64 tokens left/right)
  local_attention: int = 128

  # RoPE Thetas
  global_rope_theta: float = 160000.0
  local_rope_theta: float = 10000.0

  # Token IDs
  cls_token_id: int = 1
  pad_token_id: int = 2
  sep_token_id: int = 3

  @classmethod
  def from_dict(cls, adict: Dict[str, Any]) -> "ModernBertConfig":
    """Maps HF config dictionary to ModernBertConfig fields."""
    # Handle naming differences from HF's custom config if necessary
    mapping = {
      "sliding_window": "local_attention",
      "rotary_emb_base": "local_rope_theta",  # Example: adjust based on actual HF config
    }

    clean_dict = {}
    for k, v in adict.items():
      # Use mapping if exists, otherwise original key
      target_key = mapping.get(k, k)
      if target_key in cls.__dataclass_fields__:
        clean_dict[target_key] = v

    return cls(**clean_dict)
