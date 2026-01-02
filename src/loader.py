from transformers import AutoConfig, AutoModel

from .config import ModernBertConfig
from .model import ModernBertModel


def load_modern_bert(model_id: str, device: str = "cpu") -> ModernBertModel:
  """Loads ModernBERT and maps keys from HF format."""
  hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
  config_dict = hf_config.to_dict()

  # If the HF config uses different keys, manually set them or rely on from_dict mapping
  config = ModernBertConfig.from_dict(config_dict)

  model = ModernBertModel(config)
  hf_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

  # The keys provided in your prompt match our structure with a "model." prefix
  # We can load them directly or map them if using a raw state_dict
  state_dict = hf_model.state_dict()

  # Remove "model." prefix if necessary, though our internal structure matches
  new_state_dict = {}
  for k, v in state_dict.items():
    # Example: model.layers.0.attn.Wqkv.weight -> layers.0.attn.Wqkv.weight
    new_key = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_key] = v

  model.load_state_dict(new_state_dict, strict=False)
  return model.to(device).eval()
