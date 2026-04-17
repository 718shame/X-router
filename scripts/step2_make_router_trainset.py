import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


# --------- F1 (SQuAD-style) ---------
def _normalize_answer(s: str) -> str:
    if s is None:
        s = ""
    s = str(s).lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # white space fix
    s = " ".join(s.split())
    return s


def _f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def choose_ccp_column(df: pd.DataFrame, suf: str) -> Tuple[str, bool]:
    """
    Return (ccp_column_name, is_fallback)
    Priority: ccpXX > CCPXX > cot_ccpXX > f_ccpXX > (fallback) cot_seXX
    """
    candidates = [
        f"ccp{suf}",
        f"CCP{suf}",
        f"cot_ccp{suf}",
        f"f_ccp{suf}",
        f"ccp_metric{suf}",
    ]
    for c in candidates:
        if c in df.columns:
            return c, False

    # Fallback: use cot_seXX (in your code cot_metrics[0])
    fallback = f"cot_se{suf}"
    if fallback in df.columns:
        return fallback, True

    raise KeyError(
        f"Cannot find CCP column for suffix {suf}. Tried {candidates} and fallback {fallback}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", default="outputs/router_train/raw_hotpotqa.csv")
    ap.add_argument("--out_dir", default="outputs/router_train")
    ap.add_argument("--lambda_tok", type=float, default=3e-6)
    ap.add_argument("--lambda_time", type=float, default=3e-6)
    args = ap.parse_args()

    raw_path = Path(args.raw_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    # Required column check
    for c in ["id", "query", "answer_ground_truth"]:
        if c not in df.columns:
            raise ValueError(f"raw csv missing required column: {c}")

    suffixes = ["00", "01", "10", "11"]
    for suf in suffixes:
        if f"answer{suf}" not in df.columns:
            raise ValueError(f"raw csv missing column: answer{suf}")
        for c in [f"rag_tokens{suf}", f"cot_tokens{suf}", f"rag_time{suf}", f"cot_time{suf}"]:
            if c not in df.columns:
                raise ValueError(f"raw csv missing column: {c}")
        if suf in ["10", "11"] and f"rag_nqc{suf}" not in df.columns:
            raise ValueError(f"raw csv missing column: rag_nqc{suf} (needed for NQC)")

    # Select CCP column (00/10 are sufficient to construct training set, but will also print what 01/11 used)
    ccp_col_00, fb00 = choose_ccp_column(df, "00")
    ccp_col_01, fb01 = choose_ccp_column(df, "01")
    ccp_col_10, fb10 = choose_ccp_column(df, "10")
    ccp_col_11, fb11 = choose_ccp_column(df, "11")

    print("[CCP column selection]")
    print(f"  00 -> {ccp_col_00}" + (" (fallback)" if fb00 else ""))
    print(f"  01 -> {ccp_col_01}" + (" (fallback)" if fb01 else ""))
    print(f"  10 -> {ccp_col_10}" + (" (fallback)" if fb10 else ""))
    print(f"  11 -> {ccp_col_11}" + (" (fallback)" if fb11 else ""))

    # Calculate Q/Tok/Time/U
    for suf in suffixes:
        # Q: F1
        df[f"Q_{suf}"] = [
            _f1_score(p, g)
            for p, g in zip(df[f"answer{suf}"].fillna(""), df["answer_ground_truth"].fillna(""))
        ]
        # Tok: rag_tokens + cot_tokens
        df[f"Tok_{suf}"] = df[f"rag_tokens{suf}"].fillna(0).astype(float) + df[f"cot_tokens{suf}"].fillna(0).astype(float)
        # Time(ms): rag_time + cot_time
        df[f"T_{suf}"] = df[f"rag_time{suf}"].fillna(0).astype(float) + df[f"cot_time{suf}"].fillna(0).astype(float)
        # Utility
        df[f"U_{suf}"] = df[f"Q_{suf}"] - args.lambda_tok * df[f"Tok_{suf}"] - args.lambda_time * df[f"T_{suf}"]

    # Take the pipeline with maximum Utility
    u_cols = [f"U_{s}" for s in suffixes]
    df["best_suffix"] = df[u_cols].idxmax(axis=1).str.replace("U_", "", regex=False)

    # Labeling: need_RAG / need_CoT
    df["need_RAG"] = df["best_suffix"].isin(["10", "11"]).astype(int)  # a=1 in document
    df["need_CoT"] = df["best_suffix"].isin(["01", "11"]).astype(int)  # b=1 in document

    # Construct training features:
    # NQC: Directly take NQC under RAG_noCoT (=10)
    df["NQC"] = df["rag_nqc10"].fillna(0).astype(float)

    # CCP: need_RAG=1 -> take 10; need_RAG=0 -> take 00
    df["CCP"] = df[ccp_col_00].astype(float)
    mask = df["need_RAG"] == 1
    df.loc[mask, "CCP"] = df.loc[mask, ccp_col_10].astype(float)

    # Output intermediate (preserving more debug columns)
    intermediate_path = out_dir / "intermediate_hotpotqa.csv"
    df.to_csv(intermediate_path, index=False)
    print(f"Saved: {intermediate_path}")

    # Output router_train with four columns (+id, for tracking)
    train_df = df[["id", "NQC", "CCP", "need_RAG", "need_CoT"]].copy()
    train_path = out_dir / "router_train_hotpotqa.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved: {train_path}")

    # Print simple distribution statistics
    print("\n[Label distribution]")
    print("need_RAG:", train_df["need_RAG"].value_counts(dropna=False).to_dict())
    print("need_CoT:", train_df["need_CoT"].value_counts(dropna=False).to_dict())
    print("\n[Best suffix distribution]")
    print(df["best_suffix"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()