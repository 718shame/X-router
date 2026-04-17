import os
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import pandas as pd

from src.pipelines.functions import get_last_features

_ARTIFACT_DIR = Path(os.getenv("ROUTER_ARTIFACT_DIR", "artifacts/router"))
_MODEL_CACHE = {}


def set_artifact_dir(path: str) -> None:
    global _ARTIFACT_DIR
    _ARTIFACT_DIR = Path(path)


def _load_model(prefix: str, target: str):
    key = f"{prefix}_{target}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model_path = _ARTIFACT_DIR / f"{key}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    model = joblib.load(model_path)
    _MODEL_CACHE[key] = model
    return model


def _prepare_features(
    rag_metrics: Sequence[Sequence[float]],
    cot_metrics: Sequence[Sequence[float]],
) -> pd.DataFrame:
    feature_vectors = get_last_features()
    rows = []
    for idx, (rag, cot) in enumerate(zip(rag_metrics, cot_metrics)):
        nqc = rag[0] if rag else 0.0
        semantic_entropy = cot[0] if cot else 0.0
        if feature_vectors and idx < len(feature_vectors):
            vector = feature_vectors[idx]
            nqc = vector.get("nqc", nqc)
            ccp = vector.get("ccp",  ccp)
        rows.append({"nqc": nqc, "nll": nll})
    return pd.DataFrame(rows)


def _combine_predictions(rag: Iterable[int], cot: Iterable[int]) -> List[int]:
    mapping = {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (1, 1): 3,
    }
    return [mapping.get((int(r), int(c)), 0) for r, c in zip(rag, cot)]


def router_logit(
    rag_metrics: Sequence[Sequence[float]],
    cot_metrics: Sequence[Sequence[float]],
) -> List[int]:
    features = _prepare_features(rag_metrics, cot_metrics)
    rag_model = _load_model("router_logit", "need_RAG")
    cot_model = _load_model("router_logit", "need_CoT")
    rag_pred = rag_model.predict(features)
    cot_pred = cot_model.predict(features)
    return _combine_predictions(rag_pred, cot_pred)


def router_tree(
    rag_metrics: Sequence[Sequence[float]],
    cot_metrics: Sequence[Sequence[float]],
) -> List[int]:
    features = _prepare_features(rag_metrics, cot_metrics)
    rag_model = _load_model("router_tree", "need_RAG")
    cot_model = _load_model("router_tree", "need_CoT")
    rag_pred = rag_model.predict(features)
    cot_pred = cot_model.predict(features)
    return _combine_predictions(rag_pred, cot_pred)


def router_mlp(
    rag_metrics: Sequence[Sequence[float]],
    cot_metrics: Sequence[Sequence[float]],
) -> List[int]:
    features = _prepare_features(rag_metrics, cot_metrics)
    rag_model = _load_model("router_mlp", "need_RAG")
    cot_model = _load_model("router_mlp", "need_CoT")
    rag_pred = rag_model.predict(features)
    cot_pred = cot_model.predict(features)
    return _combine_predictions(rag_pred, cot_pred)