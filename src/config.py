# src/config.py
from dataclasses import dataclass
from typing import Any, Dict, TypedDict


class HFConfigDict(TypedDict, total=False):
  """TypedDict for HuggingFace ModernBERT config structure."""
  # Core architecture
  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  norm_eps: float
  initializer_range: float
  max_position_embeddings: int
  dropout_prob: float
  attention_probs_dropout_prob: float

  # ModernBERT-specific (our naming)
  global_attn_every_n_layers: int
  local_attention: int
  global_rope_theta: float
  local_rope_theta: float

  # HuggingFace naming variants (may differ)
  sliding_window: int  # Maps to local_attention
  rotary_emb_base: float  # Maps to local_rope_theta

  # Token IDs
  cls_token_id: int
  pad_token_id: int
  sep_token_id: int

  # Additional HF config fields (ignored but may be present)
  model_type: str
  transformers_version: str
  architectures: list[str]


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
  def from_dict(cls, config_dict: HFConfigDict | Dict[str, Any]) -> "ModernBertConfig":
    """Maps HF config dictionary to ModernBertConfig fields.

    Args:
      config_dict: HuggingFace config dictionary with potential naming differences

    Returns:
      ModernBertConfig instance with properly mapped fields
    """
    # Handle naming differences from HF's custom config if necessary
    mapping: Dict[str, str] = {
      "sliding_window": "local_attention",
      "rotary_emb_base": "local_rope_theta",
    }

    clean_dict: Dict[str, Any] = {}
    for k, v in config_dict.items():
      # Use mapping if exists, otherwise original key
      target_key = mapping.get(k, k)
      if target_key in cls.__dataclass_fields__:
        clean_dict[target_key] = v

    return cls(**clean_dict)
