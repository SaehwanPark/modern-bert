import logging

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from src.config import ModernBertConfig
from src.model import ModernBertForMaskedLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_top_k_predictions(logits, tokenizer, mask_idx, k=5):
  """Extracts top-k tokens and their probabilities for a specific mask."""
  # ModernBERT uses the logits from the decoder head
  probs = F.softmax(logits[0, mask_idx], dim=-1)
  top_probs, top_ids = torch.topk(probs, k)

  results = []
  for i in range(k):
    token = tokenizer.convert_ids_to_tokens([top_ids[i].item()])[0]
    results.append((token, top_probs[i].item()))
  return results


def run_comparison(text, model_id="answerdotai/ModernBERT-base"):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  logger.info(f"Using device: {device}")

  # 1. Prepare Tokenizer and Inputs
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  inputs = tokenizer(text, return_tensors="pt").to(device)
  mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

  # 2. Load HuggingFace Model (Reference)
  logger.info("Loading HuggingFace Reference Model...")
  hf_model = (
    AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    .to(device)
    .eval()
  )

  # 3. Load Custom Replicated Model
  logger.info("Loading Custom Replicated Model...")
  hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
  # Map config to our ModernBertConfig attributes
  config = ModernBertConfig.from_dict(hf_config.to_dict())
  custom_model = ModernBertForMaskedLM(config).to(device).eval()
  # Strict load ensures all architecture details match exactly
  custom_model.load_state_dict(hf_model.state_dict(), strict=True)

  # 4. Inference
  with torch.no_grad():
    hf_logits = hf_model(**inputs).logits
    # Custom model follows the same forward pass logic
    custom_logits = custom_model(
      inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )

  # 5. Display Results Side-by-Side
  print(f"\nInput Text: {text}")
  print("-" * 80)
  print(f"{'Rank':<5} | {'HuggingFace Prediction':<25} | {'Custom Prediction':<25}")
  print("-" * 80)

  for m_idx, token_pos in enumerate(mask_token_index):
    hf_preds = get_top_k_predictions(hf_logits, tokenizer, token_pos)
    custom_preds = get_top_k_predictions(custom_logits, tokenizer, token_pos)

    for i in range(len(hf_preds)):
      hf_str = f"{hf_preds[i][0][1:]} ({hf_preds[i][1]:.4f})"
      custom_str = f"{custom_preds[i][0][1:]} ({custom_preds[i][1]:.4f})"
      print(f"{i + 1:<5} | {hf_str:<25} | {custom_str:<25}")
    print("-" * 80)


if __name__ == "__main__":
  test_sentence = "The capital of France is [MASK]."
  run_comparison(test_sentence)

  # Try an example showcasing alternating attention (Global context)
  long_sentence = "ModernBERT is a new [MASK] model that uses alternating attention for better performance."
  run_comparison(long_sentence)
