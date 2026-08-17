import pandas as pd
import random
import os
from app.config import PREDICTIONS_CSV

_df = None

def load_data():
    global _df
    if _df is None:
        if os.path.exists(PREDICTIONS_CSV):
            _df = pd.read_csv(PREDICTIONS_CSV).fillna("")
        else:
            _df = pd.DataFrame()

def get_total_samples() -> int:
    load_data()
    if _df is not None and not _df.empty:
        return len(_df)
    return 0

import re

def _quick_clean(text: str) -> str:
    # Replace underscores with space
    text = text.replace('_', ' ')
    # Fix punctuation spacing
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_sample(index: int) -> dict:
    load_data()
    if _df is None or _df.empty or index < 0 or index >= len(_df):
        return None
        
    row = _df.iloc[index]
    return {
        "index": index,
        "article": _quick_clean(str(row.get("article", ""))),
        "reference": _quick_clean(str(row.get("reference", ""))),
        "predictions": {
            "bartpho_fft": _quick_clean(str(row.get("bartpho_full_beam", ""))),
            "bartpho_lora": _quick_clean(str(row.get("bartpho_lora_beam", ""))),
            "qwen_lora": _quick_clean(str(row.get("qwen_lora_beam", "")))
        }
    }

def get_random_sample() -> dict:
    total = get_total_samples()
    if total == 0:
        return None
    idx = random.randint(0, total - 1)
    return get_sample(idx)
