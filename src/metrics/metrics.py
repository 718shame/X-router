import logging
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-z\s]", "", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_nqc(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 1:
        return 0.0
    if np.isnan(arr).any() or not np.isfinite(arr).all():
        logger.warning("compute_nqc sanitized invalid scores: %s", arr[:5].tolist())
    std = float(arr.std())
    mean = float(np.abs(arr.mean())) + 1e-6
    value = std / mean
    if not math.isfinite(value):
        logger.warning("compute_nqc output non-finite value; std=%s mean=%s scores=%s", std, mean, arr[:5].tolist())
        return 0.0
    return value


def semantic_entropy(similarities: Iterable[float]) -> float:
    sims = np.array(list(similarities), dtype=np.float32)
    if sims.size == 0:
        return 0.0
    probs = np.exp(sims - sims.max())
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
    return entropy


def pairwise_cosine(vectors: List[np.ndarray]) -> List[float]:
    if len(vectors) <= 1:
        return [0.0]
    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normalized = matrix / norms
    sims = normalized @ normalized.T
    triu_indices = np.triu_indices_from(sims, k=1)
    return sims[triu_indices].tolist()


def aggregate_costs(costs: Iterable[Tuple[float, int]]) -> Tuple[float, int]:
    time_sum = 0.0
    token_sum = 0
    for time_cost, token_cost in costs:
        time_sum += float(time_cost)
        token_sum += int(token_cost)
    return time_sum, token_sum
