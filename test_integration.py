import torch
from transformers import AutoConfig, AutoModelForMaskedLM

from src.config import ModernBertConfig
from src.model import ModernBertForMaskedLM


def test_compare_with_hf() -> None:
  model_id: str = "answerdotai/ModernBERT-base"
  device: str = "cpu"

  # 1. Load HF Model
  hf_model = (
    AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    .to(device)
    .eval()
  )
  hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

  # 2. Initialize Custom Model
  config = ModernBertConfig.from_dict(hf_config.to_dict())
  custom_model = ModernBertForMaskedLM(config).to(device).eval()

  # 3. Transfer Weights
  # HF stores ModernBERT under 'model' and the head/decoder separately
  custom_model.load_state_dict(hf_model.state_dict(), strict=True)

  # 4. Prepare Input
  input_ids = torch.tensor([[1, 50, 200, 15, 2]], dtype=torch.long)  # [CLS] ... [SEP]

  with torch.no_grad():
    hf_logits = hf_model(input_ids).logits
    custom_logits = custom_model(input_ids)

  # 5. Compare
  # Using atol=1e-5 due to potential floating point accumulation differences in SDPA
  diff = torch.abs(hf_logits - custom_logits).max()
  print(f"Max logit difference: {diff.item()}")

  assert torch.allclose(hf_logits, custom_logits, atol=1e-5), (
    f"Logits mismatch! Max diff: {diff.item()}"
  )


if __name__ == "__main__":
  test_compare_with_hf()
