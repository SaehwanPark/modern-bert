import torch

from src.config import ModernBertConfig
from src.model import ModernBertModel


def test_modern_bert_forward():
  config = ModernBertConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=4)
  model = ModernBertModel(config).eval()

  input_ids = torch.randint(0, config.vocab_size, (2, 16))
  mask = torch.ones((2, 16))

  with torch.no_grad():
    output = model(input_ids, attention_mask=mask)

  assert output.shape == (2, 16, 128)
  assert not torch.isnan(output).any()


def test_skip_first_norm_exists():
  config = ModernBertConfig(num_hidden_layers=2)
  model = ModernBertModel(config)

  # Layer 0: "Skip-first" behavior - should be None
  assert model.layers[0].attn_norm is None

  # Layer 1: LayerNorm
  assert isinstance(model.layers[1].attn_norm, torch.nn.LayerNorm)
