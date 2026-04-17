import itertools
import pandas as pd
from pathlib import Path
from collections import Counter
import re
import string

# ================== F1 ==================
def normalize(s):
    if s is None:
        s = ""
    s = str(s).lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def f1(pred, gold):
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

# ================== Metrics ==================
def compute_metrics(df, lam_tok, lam_time):
    df = df.copy()

    # Compatible with gt / answer_ground_truth
    if "answer_ground_truth" not in df.columns and "gt" in df.columns:
        df = df.rename(columns={"gt": "answer_ground_truth"})

    required = ["answer", "answer_ground_truth", "rag_tokens", "cot_tokens", "rag_time", "cot_time"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}', got {list(df.columns)}")

    df["F1"] = [
        f1(a, g)
        for a, g in zip(df["answer"].fillna(""), df["answer_ground_truth"].fillna(""))
    ]
    df["Tokens"] = df["rag_tokens"].fillna(0).astype(float) + df["cot_tokens"].fillna(0).astype(float)
    df["Time"] = df["rag_time"].fillna(0).astype(float) + df["cot_time"].fillna(0).astype(float)
    df["Utility"] = df["F1"] - lam_tok * df["Tokens"] - lam_time * df["Time"]

    return {
        "F1": df["F1"].mean(),
        "Tokens": df["Tokens"].mean(),
        "Time": df["Time"].mean(),
        "Utility": df["Utility"].mean(),
    }

# ================== Main ==================
def main():
    base_dir = Path("outputs/router_vllm/hotpotqa/hotpotqa")
    router1_path = Path("outputs/router_test/router1_hotpotqa.csv")

    baselines = {
        "always_00": base_dir / "noRAG_noCoT.csv",
        "always_01": base_dir / "noRAG_CoT.csv",
        "always_10": base_dir / "noARAG_noCoT.csv",
        "always_11": base_dir / "noARAG_CoT.csv",
    }

    dfs = {k: pd.read_csv(v) for k, v in baselines.items()}
    router1 = pd.read_csv(router1_path)

    lambda_tok_list  = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    lambda_time_list = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]

    records = []

    for lam_tok, lam_time in itertools.product(lambda_tok_list, lambda_time_list):
        row = {
            "lambda_tok": lam_tok,
            "lambda_time": lam_time,
        }

        # baselines
        for name, df in dfs.items():
            m = compute_metrics(df, lam_tok, lam_time)
            for k, v in m.items():
                row[f"{name}_{k}"] = v

        # router1
        m_r1 = compute_metrics(router1, lam_tok, lam_time)
        for k, v in m_r1.items():
            row[f"router1_{k}"] = v

        # oracle (upper bound of baseline)
        row["oracle_Utility"] = max(row[f"{k}_Utility"] for k in baselines)

        records.append(row)

    out = pd.DataFrame(records)
    out_path = Path("outputs/router_test/lambda_grid_results_full.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Saved full lambda grid results to:", out_path)
    print("\nTop settings by Router1 Utility:")
    print(out.sort_values("router1_Utility", ascending=False).head(10))

if __name__ == "__main__":
    main()