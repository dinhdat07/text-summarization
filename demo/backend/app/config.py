"""Configuration paths and constants for the backend."""
from pathlib import Path

# Project root (4 levels up: app -> backend -> demo -> text-summarization)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Model checkpoint paths
MODELS_DIR = PROJECT_ROOT / "models"
BARTPHO_FFT_PATH = MODELS_DIR / "bartpho_full_ft_final"
BARTPHO_LORA_PATH = MODELS_DIR / "bartpho_lora_final"
QWEN_LORA_PATH = MODELS_DIR / "qwen-lora-vietnews"

# Base model names (downloaded from HuggingFace if not cached)
BARTPHO_BASE = "vinai/bartpho-syllable"
QWEN_BASE = "Qwen/Qwen2.5-0.5B"

# Pre-computed predictions CSV
PREDICTIONS_CSV = PROJECT_ROOT / "results" / "all_models_predictions_postprocessed.csv"

# Generation config
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 256
NUM_BEAMS = 4
