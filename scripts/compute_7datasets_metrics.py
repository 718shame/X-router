#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Metric spec from metric documentation.pdf
# -----------------------------
DATASET_METRICS: Dict[str, List[str]] = {
    "hotpotqa": ["F1", "EM", "FF", "ROUGE-L"],
    "2wikimultihopqa": ["F1", "EM", "FF", "ROUGE-L"],
    "eli5": ["F1", "EM", "FF", "ROUGE-L"],
    "fiqa": ["F1", "EM", "FF", "ROUGE-L"],
    "medqa": ["AC", "FF"],
    "strategyqa": ["AC", "FF"],
    "gsm8k": ["AC"],
}

# Your 4 pipelines (file names)
PIPELINES: List[Tuple[str, str]] = [
    ("noRAG_noCoT", "noRAG_noCoT.csv"),
    ("noRAG_CoT", "noRAG_CoT.csv"),
    ("RAGnoCoT", "noARAG_noCoT.csv"),  # code noARAG == doc RAG
    ("RAG_CoT", "noARAG_CoT.csv"),
]


# -----------------------------
# Text normalization + F1/EM/AC
# -----------------------------
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    if text is None:
        text = ""
    text = str(text).lower()
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


def accuracy(prediction: str, ground_truth: str) -> float:
    # AC is basically EM for these datasets (choice/yes-no/short answer)
    return exact_match(prediction, ground_truth)


# -----------------------------
# ROUGE-L (LCS-based)
# Return ROUGE-L F1 (common in papers)
# -----------------------------
def _lcs_length(x: List[str], y: List[str]) -> int:
    # DP, O(n*m) but answers are short; OK for your 7 datasets
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(prediction: str, ground_truth: str) -> float:
    # Tokenize by normalized tokens to be consistent with your F1/EM pipeline
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, gold_toks)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred_toks)
    rec = lcs / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# -----------------------------
# FF (Faithfulness)
# According to your current CSVs:
# Prefer f_selfcheck; fallback to 1-ccp
# -----------------------------
def compute_ff(row: pd.Series) -> Optional[float]:
    if "f_selfcheck" in row and pd.notna(row["f_selfcheck"]):
        try:
            return float(row["f_selfcheck"])
        except Exception:
            return None
    if "ccp" in row and pd.notna(row["ccp"]):
        try:
            return 1.0 - float(row["ccp"])
        except Exception:
            return None
    return None


# -----------------------------
# I/O helpers
# -----------------------------
def find_dataset_dir(root: Path, dataset_key: str) -> Path:
    """
    Expected structure:
      outputs/router_vllm/<dataset>/<dataset>/*.csv
    Your MedQA might be 'MedQA/MedQA', others lower-case.
    We'll try case-insensitive matches.
    """
    # direct expected
    cand = root / dataset_key / dataset_key
    if cand.exists():
        return cand

    # try case variants in root
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == dataset_key.lower():
            inner = p / p.name
            if inner.exists():
                return inner
            # sometimes inner folder is lower-case
            inner2 = p / dataset_key
            if inner2.exists():
                return inner2
            # fallback: if only one inner dir
            inners = [x for x in p.iterdir() if x.is_dir()]
            if len(inners) == 1:
                return inners[0]

    raise FileNotFoundError(f"Cannot locate dataset folder for '{dataset_key}' under {root}")


def read_pipeline_csv(dataset_dir: Path, filename: str) -> pd.DataFrame:
    path = dataset_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline CSV: {path}")
    df = pd.read_csv(path)

    # normalize column names you mentioned
    required = {"answer_ground_truth", "answer"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {miss}. columns={list(df.columns)}")

    # Ensure time/token columns exist (rag_time, rag_tokens, cot_time, cot_tokens)
    for col in ["rag_time", "rag_tokens", "cot_time", "cot_tokens"]:
        if col not in df.columns:
            # some runs might not have rag part (noRAG) => can be 0
            df[col] = 0.0

    return df


def compute_summary_for(df: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # always compute time/token
    df = df.copy()
    df["time_ms"] = df["rag_time"].fillna(0).astype(float) + df["cot_time"].fillna(0).astype(float)
    df["tokens"] = df["rag_tokens"].fillna(0).astype(float) + df["cot_tokens"].fillna(0).astype(float)

    out["time"] = float(df["time_ms"].mean())
    out["token"] = float(df["tokens"].mean())

    # per-metric means
    gt = df["answer_ground_truth"].fillna("").astype(str).tolist()
    ans = df["answer"].fillna("").astype(str).tolist()

    if "F1" in metrics:
        out["F1"] = float(sum(f1_score(a, g) for a, g in zip(ans, gt)) / len(df))
    if "EM" in metrics:
        out["EM"] = float(sum(exact_match(a, g) for a, g in zip(ans, gt)) / len(df))
    if "AC" in metrics:
        out["AC"] = float(sum(accuracy(a, g) for a, g in zip(ans, gt)) / len(df))
    if "ROUGE-L" in metrics:
        out["ROUGE-L"] = float(sum(rouge_l_f1(a, g) for a, g in zip(ans, gt)) / len(df))
    if "FF" in metrics:
        ffs = []
        for _, row in df.iterrows():
            v = compute_ff(row)
            if v is not None:
                ffs.append(v)
        out["FF"] = float(sum(ffs) / len(ffs)) if ffs else float("nan")

    out["N"] = float(len(df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/router_vllm", help="root dir containing 7 datasets")
    ap.add_argument("--out_dir", default="outputs/metrics_summary", help="where to write summary csvs")
    ap.add_argument("--fail_on_missing", action="store_true", help="stop immediately if any csv missing")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []

    for dataset_key, needed_metrics in DATASET_METRICS.items():
        dataset_dir = find_dataset_dir(root, dataset_key)

        for method_name, csv_name in PIPELINES:
            try:
                df = read_pipeline_csv(dataset_dir, csv_name)
                summary = compute_summary_for(df, needed_metrics)
            except Exception as e:
                if args.fail_on_missing:
                    raise
                summary = {m: float("nan") for m in needed_metrics}
                summary.update({"time": float("nan"), "token": float("nan"), "N": float("nan")})
                summary["error"] = str(e)

            # write long format rows: each metric becomes a row (Method, Metric, Dataset, Value)
            for metric in needed_metrics + ["time", "token"]:
                long_rows.append({
                    "dataset": dataset_key,
                    "method": method_name,
                    "metric": metric,
                    "value": summary.get(metric, float("nan")),
                    "N": summary.get("N", float("nan")),
                })

    long_df = pd.DataFrame(long_rows)

    # wide format: index=(method, metric), columns=datasets
    wide_df = long_df.pivot_table(
        index=["method", "metric"],
        columns="dataset",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # optional: sort rows in your desired order
    method_order = {m: i for i, (m, _) in enumerate(PIPELINES)}
    metric_order = {"F1": 0, "EM": 1, "AC": 2, "ROUGE-L": 3, "FF": 4, "time": 5, "token": 6}
    wide_df["__m"] = wide_df["method"].map(method_order).fillna(999)
    wide_df["__k"] = wide_df["metric"].map(metric_order).fillna(999)
    wide_df = wide_df.sort_values(["__m", "__k"]).drop(columns=["__m", "__k"])
    wide_df = wide_df.round(2)

    long_path = out_dir / "results_long.csv"
    wide_path = out_dir / "results_wide.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("Saved:")
    print(" -", long_path)
    print(" -", wide_path)
    print("\nPreview (wide):")
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(wide_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()