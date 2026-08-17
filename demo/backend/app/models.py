"""Module for loading and running inference on summarization models."""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
from app.config import (
    BARTPHO_FFT_PATH, BARTPHO_LORA_PATH, QWEN_LORA_PATH,
    BARTPHO_BASE, QWEN_BASE,
    MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, NUM_BEAMS,
)

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

_models = {}
_available_models = []


def _post_process_qwen(text: str) -> str:
    """Remove prompt remnants and special tokens from Qwen output."""
    text = str(text)
    for marker in ['\n\nTóm tắt:', '\nTóm tắt:', 'Tóm tắt bài báo sau:']:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    for token in ['<|endoftext|>', '<|im_end|>', '<|im_start|>']:
        text = text.replace(token, '')
    return text.strip()


def load_models() -> list:
    """Load all available model checkpoints. Returns list of loaded model names."""
    global _models, _available_models
    _available_models = []

    # 1. BARTpho Full Fine-Tuning
    if BARTPHO_FFT_PATH.exists():
        try:
            logger.info(f"Loading BARTpho FFT from {BARTPHO_FFT_PATH}...")
            tokenizer = AutoTokenizer.from_pretrained(str(BARTPHO_FFT_PATH))
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(BARTPHO_FFT_PATH), torch_dtype=dtype
            ).to(device).eval()
            _models["bartpho_fft"] = {"model": model, "tokenizer": tokenizer, "type": "seq2seq"}
            _available_models.append("bartpho_fft")
            logger.info("BARTpho FFT loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load BARTpho FFT: {e}")

    # 2. BARTpho LoRA
    if BARTPHO_LORA_PATH.exists():
        try:
            logger.info(f"Loading BARTpho LoRA from {BARTPHO_LORA_PATH}...")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(BARTPHO_BASE, torch_dtype=dtype)
            model = PeftModel.from_pretrained(base_model, str(BARTPHO_LORA_PATH)).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(str(BARTPHO_LORA_PATH))
            _models["bartpho_lora"] = {"model": model, "tokenizer": tokenizer, "type": "seq2seq"}
            _available_models.append("bartpho_lora")
            logger.info("BARTpho LoRA loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load BARTpho LoRA: {e}")

    # 3. Qwen2.5 LoRA
    if QWEN_LORA_PATH.exists():
        try:
            logger.info(f"Loading Qwen2.5 LoRA from {QWEN_LORA_PATH}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                QWEN_BASE, torch_dtype=dtype, trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, str(QWEN_LORA_PATH)).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            _models["qwen_lora"] = {"model": model, "tokenizer": tokenizer, "type": "causal"}
            _available_models.append("qwen_lora")
            logger.info("Qwen2.5 LoRA loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Qwen2.5 LoRA: {e}")

    if not _available_models:
        logger.warning("No model checkpoints found. Only pre-computed sample browser will work.")

    return _available_models


def get_available_models() -> list:
    """Return list of successfully loaded model names."""
    return _available_models


def generate_summary(model_name: str, text: str) -> str:
    """Generate summary from a specific model."""
    if model_name not in _models:
        return f"[Model '{model_name}' not available]"

    entry = _models[model_name]
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    model_type = entry["type"]

    with torch.no_grad():
        if model_type == "seq2seq":
            inputs = tokenizer(
                text, max_length=MAX_INPUT_LENGTH,
                truncation=True, return_tensors="pt"
            ).to(device)
            outputs = model.generate(
                **inputs, max_length=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS, early_stopping=True,
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif model_type == "causal":
            prompt = f"Tóm tắt bài báo sau:\n{text}\n\nTóm tắt:\n"
            inputs = tokenizer(
                prompt, max_length=MAX_INPUT_LENGTH,
                truncation=True, return_tensors="pt"
            ).to(device)
            prompt_length = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS, early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            summary_ids = outputs[0][prompt_length:]
            summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
            summary = _post_process_qwen(summary)
        else:
            return "[Unknown model type]"

    return summary.strip()


def generate_all(text: str) -> dict:
    """Run inference on all loaded models."""
    results = {}
    for name in _available_models:
        results[name] = generate_summary(name, text)
    return results
