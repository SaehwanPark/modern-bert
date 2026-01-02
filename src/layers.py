import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
  return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def get_sliding_window_mask(
  q_len: int, window: int, device: torch.device
) -> torch.Tensor:
  """Creates a bidirectional window mask (64 left, 64 right for window=128)."""
  half = window // 2
  idx = torch.arange(q_len, device=device)
  mask = (idx[:, None] - idx[None, :]).abs() <= half
  return mask  # (S, S) bool mask


class ModernBertAttention(nn.Module):
  def __init__(self, config, layer_id: int):
    super().__init__()
    self.num_heads = config.num_attention_heads
    self.head_dim = config.hidden_size // self.num_heads
    self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
    self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    self.is_global = layer_id % config.global_attn_every_n_layers == 0
    self.window_size = config.local_attention if not self.is_global else -1

  def forward(self, x, cos, sin, mask=None):
    bsz, q_len, _ = x.size()
    qkv = (
      self.Wqkv(x).view(bsz, q_len, 3, self.num_heads, self.head_dim).transpose(1, 3)
    )
    q, k, v = qkv.unbind(dim=2)
    q, k = apply_rope(q, k, cos, sin)

    if not self.is_global:
      sw_mask = get_sliding_window_mask(q_len, self.window_size, x.device)
      # SDPA bool mask: True means attend
      mask = (
        (mask & sw_mask[None, None, :, :])
        if mask is not None
        else sw_mask[None, None, :, :]
      )

    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    return self.Wo(attn_out)


class ModernBertMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
    self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

  def forward(self, x):
    # ModernBERT GeGLU: act(first_half) * second_half
    gate, val = self.Wi(x).chunk(2, dim=-1)
    return self.Wo(F.gelu(gate) * val)
