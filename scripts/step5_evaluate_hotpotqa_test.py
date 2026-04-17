import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Optional

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


def add_metrics_cols(df: pd.DataFrame, lambda_tok: float, lambda_time: float) -> pd.DataFrame:
    # required columns
    for c in ["answer_ground_truth", "answer", "rag_tokens", "cot_tokens", "rag_time", "cot_time"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {list(df.columns)}")

    df = df.copy()
    df["F1"] = [f1_score(a, g) for a, g in zip(df["answer"].fillna(""), df["answer_ground_truth"].fillna(""))]
    df["Tokens"] = df["rag_tokens"].fillna(0).astype(float) + df["cot_tokens"].fillna(0).astype(float)
    df["Time_ms"] = df["rag_time"].fillna(0).astype(float) + df["cot_time"].fillna(0).astype(float)
    df["Utility"] = df["F1"] - lambda_tok * df["Tokens"] - lambda_time * df["Time_ms"]
    return df


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "F1": float(df["F1"].mean()),
        "Tokens": float(df["Tokens"].mean()),
        "Time_ms": float(df["Time_ms"].mean()),
        "Utility": float(df["Utility"].mean()),
        "N": float(len(df)),
    }


def try_find_join_key(df: pd.DataFrame) -> str:
    # router1 may have id, or only query
    if "id" in df.columns:
        return "id"
    if "query" in df.columns:
        return "query"
    raise ValueError(f"Router1 file has no join key. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="outputs/router_vllm/hotpotqa/hotpotqa")
    ap.add_argument("--router1_csv", default="outputs/router_test/router1_hotpotqa.csv")
    ap.add_argument("--out_table", default="outputs/router_test/step5_hotpotqa_results.csv")
    ap.add_argument("--out_oracle_csv", default="outputs/router_test/oracle_hotpotqa.csv")
    ap.add_argument("--lambda_tok", type=float, default=3e-4)
    ap.add_argument("--lambda_time", type=float, default=3e-5)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)

    files = {
        "always_00(noRAG,noCoT)": base_dir / "noRAG_noCoT.csv",
        "always_01(noRAG,CoT)": base_dir / "noRAG_CoT.csv",
        "always_10(RAG,noCoT)": base_dir / "noARAG_noCoT.csv",  # noARAG == document RAG
        "always_11(RAG,CoT)": base_dir / "noARAG_CoT.csv",
    }
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing baseline file: {p}")

    # read + compute per-row metrics
    baseline_dfs = {}
    for name, p in files.items():
        df = pd.read_csv(p)
        if "id" not in df.columns:
            raise ValueError(f"Baseline {p} missing 'id' column.")
        baseline_dfs[name] = add_metrics_cols(df, args.lambda_tok, args.lambda_time)

    # ---- ORACLE among 4 baselines (per row) ----
    # merge utilities by id
    ids = baseline_dfs[next(iter(baseline_dfs))]["id"].tolist()
    oracle_rows = []
    for i, _id in enumerate(ids):
        best_name = None
        best_u = -1e9
        best_row = None
        for name, df in baseline_dfs.items():
            row = df.iloc[i]
            # assume same ordering; if not, safer to join by id:
            if row["id"] != _id:
                # fallback: locate by id
                row = df[df["id"] == _id].iloc[0]
            u = float(row["Utility"])
            if u > best_u:
                best_u = u
                best_name = name
                best_row = row
        oracle_rows.append({
            "id": _id,
            "chosen": best_name,
            "answer_ground_truth": best_row["answer_ground_truth"],
            "answer": best_row["answer"],
            "F1": best_row["F1"],
            "Tokens": best_row["Tokens"],
            "Time_ms": best_row["Time_ms"],
            "Utility": best_row["Utility"],
        })
    oracle_df = pd.DataFrame(oracle_rows)
    Path(args.out_oracle_csv).parent.mkdir(parents=True, exist_ok=True)
    oracle_df.to_csv(args.out_oracle_csv, index=False)

    # ---- Router1 (test) ----
    router1_path = Path(args.router1_csv)
    if not router1_path.exists():
        raise FileNotFoundError(f"Missing router1 csv: {router1_path}")

    r1 = pd.read_csv(router1_path)
    join_key = try_find_join_key(r1)

    # router1 must provide gt/answer/token/time; if gt column name is different, make compatibility
    if "gt" in r1.columns and "answer_ground_truth" not in r1.columns:
        r1 = r1.rename(columns={"gt": "answer_ground_truth"})
    if "answer" not in r1.columns:
        raise ValueError("router1 csv missing 'answer' column.")
    for c in ["rag_tokens", "cot_tokens", "rag_time", "cot_time", "answer_ground_truth"]:
        if c not in r1.columns:
            raise ValueError(f"router1 csv missing '{c}' column. columns={list(r1.columns)}")

    r1m = add_metrics_cols(r1, args.lambda_tok, args.lambda_time)

    # ---- summarize table ----
    rows = []
    for name, df in baseline_dfs.items():
        s = summarize(df)
        s["Method"] = name
        rows.append(s)

    s_oracle = summarize(oracle_df)
    s_oracle["Method"] = "oracle(best Utility among 4 baselines)"
    rows.append(s_oracle)

    s_r1 = summarize(r1m)
    s_r1["Method"] = "router1(Our Method)"
    rows.append(s_r1)

    out = pd.DataFrame(rows)[["Method", "F1", "Tokens", "Time_ms", "Utility", "N"]]
    out_path = Path(args.out_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Saved Step5 table to:", out_path)
    print(out.to_string(index=False))
    print("\nSaved oracle per-row choices to:", args.out_oracle_csv)


if __name__ == "__main__":
    main()