"""Module for computing evaluation metrics."""
import logging
import numpy as np
from evaluate import load

logger = logging.getLogger(__name__)

# Lazy-load evaluators
_rouge = None
_bleu = None
_bertscore = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = load("rouge")
    return _rouge


def _get_bleu():
    global _bleu
    if _bleu is None:
        _bleu = load("bleu")
    return _bleu


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        _bertscore = load("bertscore")
    return _bertscore


def compute_metrics(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, BLEU, BERTScore for a single pair."""
    if not prediction or not reference:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0, "bleu": 0, "bertscore": 0}

    predictions = [prediction]
    references_rouge = [reference]
    references_bleu = [[reference]]

    # ROUGE
    rouge_result = _get_rouge().compute(predictions=predictions, references=references_rouge)

    # BLEU
    try:
        bleu_result = _get_bleu().compute(predictions=predictions, references=references_bleu)
        bleu_score = bleu_result["bleu"]
    except (ZeroDivisionError, ValueError):
        bleu_score = 0.0

    # BERTScore
    bert_result = _get_bertscore().compute(predictions=predictions, references=references_rouge, lang="vi")

    return {
        "rouge1": round(rouge_result["rouge1"] * 100, 2),
        "rouge2": round(rouge_result["rouge2"] * 100, 2),
        "rougeL": round(rouge_result["rougeL"] * 100, 2),
        "bleu": round(bleu_score * 100, 2),
        "bertscore": round(float(np.mean(bert_result["f1"])) * 100, 2),
    }


import concurrent.futures

def compute_all_metrics(predictions: dict, reference: str) -> dict:
    """Compute metrics for multiple model predictions against one reference concurrently."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(predictions)) as executor:
        future_to_name = {
            executor.submit(compute_metrics, text, reference): name 
            for name, text in predictions.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.error(f"Metrics computation for {name} generated an exception: {exc}")
                results[name] = {"rouge1": 0, "rouge2": 0, "rougeL": 0, "bleu": 0, "bertscore": 0}
    return results
