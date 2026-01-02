import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ModernBertAttention, ModernBertMLP


class ModernBertLayer(nn.Module):
  def __init__(self, config, layer_id: int):
    super().__init__()
    # ModernBERT uses bias-free LayerNorm
    # Skip attn_norm for Layer 0 (follows embedding norm)
    self.attn_norm = (
      nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
      if layer_id > 0
      else None
    )
    self.attn = ModernBertAttention(config, layer_id)
    self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
    self.mlp = ModernBertMLP(config)

  def forward(self, x, cos, sin, mask=None):
    res = x
    x = self.attn_norm(x) if self.attn_norm else x
    x = res + self.attn(x, cos, sin, mask)
    res = x
    x = self.mlp_norm(x)
    return res + self.mlp(x)


class ModernBertModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embeddings = nn.ModuleDict(
      {
        "tok_embeddings": nn.Embedding(config.vocab_size, config.hidden_size),
        "norm": nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False),
      }
    )
    self.layers = nn.ModuleList(
      [ModernBertLayer(config, i) for i in range(config.num_hidden_layers)]
    )
    self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
    self._register_rope_caches(config)

  def _register_rope_caches(self, config):
    dim = config.hidden_size // config.num_attention_heads

    def _get_cache(theta):
      inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
      t = torch.arange(config.max_position_embeddings)
      freqs = torch.outer(t, inv_freq)
      emb = torch.cat((freqs, freqs), dim=-1)
      return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    gc, gs = _get_cache(config.global_rope_theta)  # 160,000
    lc, ls = _get_cache(config.local_rope_theta)  # 10,000
    self.register_buffer("rope_cos_g", gc, persistent=False)
    self.register_buffer("rope_sin_g", gs, persistent=False)
    self.register_buffer("rope_cos_l", lc, persistent=False)
    self.register_buffer("rope_sin_l", ls, persistent=False)

  def forward(self, input_ids, attention_mask=None):
    x = self.embeddings.tok_embeddings(input_ids)
    x = self.embeddings.norm(x)
    mask = (
      attention_mask.bool()[:, None, None, :] if attention_mask is not None else None
    )
    q_len = x.size(1)
    for layer in self.layers:
      cos = (self.rope_cos_g if layer.attn.is_global else self.rope_cos_l)[
        :, :, :q_len, :
      ]
      sin = (self.rope_sin_g if layer.attn.is_global else self.rope_sin_l)[
        :, :, :q_len, :
      ]
      x = layer(x, cos, sin, mask)
    return self.final_norm(x)


class ModernBertForMaskedLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.model = ModernBertModel(config)
    self.head = nn.ModuleDict(
      {
        "dense": nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        "norm": nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False),
      }
    )
    self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

  def forward(self, input_ids, attention_mask=None):
    x = self.model(input_ids, attention_mask)
    # MLM Head sequence: Linear -> GELU -> LayerNorm
    x = self.head["norm"](F.gelu(self.head["dense"](x)))
    return self.decoder(x)
