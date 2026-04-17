import argparse
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd


def _normalize_answer(s: str) -> str:
    if s is None:
        s = ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def f1_score(pred: str, gold: str) -> float:
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
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intermediate_csv", required=True)
    ap.add_argument("--lambda_tok", type=float, required=True)
    ap.add_argument("--lambda_time", type=float, required=True)
    ap.add_argument("--out_router_train_csv", required=True)
    ap.add_argument("--out_intermediate_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.intermediate_csv)

    # Standardize suffix
    suffixes = ["00", "01", "10", "11"]

    # Ensure Tok_xx / T_xx exist; if not, combine rag/cot (compatible with different intermediate table versions)
    for s in suffixes:
        if f"Tok_{s}" not in df.columns:
            rt = df.get(f"rag_tokens{s}", 0)
            ct = df.get(f"cot_tokens{s}", 0)
            df[f"Tok_{s}"] = rt.fillna(0) + ct.fillna(0)
        if f"T_{s}" not in df.columns:
            rt = df.get(f"rag_time{s}", 0)
            ct = df.get(f"cot_time{s}", 0)
            df[f"T_{s}"] = rt.fillna(0) + ct.fillna(0)

    # Ensure Q_xx (quality) exists; if not, calculate F1(answerxx, answer_ground_truth)
    if "answer_ground_truth" not in df.columns:
        # Usually this is the column name in your intermediate; if not, add compatibility here
        raise ValueError("intermediate_csv must contain 'answer_ground_truth' column")

    for s in suffixes:
        qcol = f"Q_{s}"
        acol = f"answer{s}"
        if qcol not in df.columns:
            if acol not in df.columns:
                raise ValueError(f"Missing '{acol}' in intermediate_csv columns")
            df[qcol] = [
                f1_score(a, g) for a, g in zip(df[acol].fillna(""), df["answer_ground_truth"].fillna(""))
            ]

    # Calculate U_xx with current lambda
    lam_tok = args.lambda_tok
    lam_time = args.lambda_time
    for s in suffixes:
        df[f"U_{s}"] = df[f"Q_{s}"] - lam_tok * df[f"Tok_{s}"] - lam_time * df[f"T_{s}"]

    # best_suffix: select suffix with maximum U
    df["best_suffix"] = df[[f"U_{s}" for s in suffixes]].idxmax(axis=1).str.replace("U_", "", regex=False)

    # Labels: need_RAG / need_CoT (aligned with best_suffix)
    df["need_RAG"] = df["best_suffix"].isin(["10", "11"]).astype(int)
    df["need_CoT"] = df["best_suffix"].isin(["01", "11"]).astype(int)

    # Save updated intermediate (containing new U/best/labels)
    out_i = Path(args.out_intermediate_csv)
    out_i.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_i, index=False)

    # Generate router_train table (training only keeps necessary columns to avoid confusion)
    # Features: rag_nqc(for RAG router), ccp(for CoT router) + labels
    need_cols = ["id", "query", "NQC", "CCP", "need_RAG", "need_CoT"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"intermediate_csv missing columns required for training: {missing}")

    train_df = df[need_cols].copy()

    # For consistency with Step3 / Router1 naming (optional but strongly recommended)
    train_df = train_df.rename(columns={
        "NQC": "rag_nqc",
        "CCP": "ccp"
    })

    out_t = Path(args.out_router_train_csv)
    out_t.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_t, index=False)

    print("[Step2-lambda] Saved:")
    print(" - intermediate:", out_i)
    print(" - router_train:", out_t)
    print("[Label distribution]")
    print(" need_RAG:", train_df["need_RAG"].value_counts().to_dict())
    print(" need_CoT:", train_df["need_CoT"].value_counts().to_dict())


if __name__ == "__main__":
    main()