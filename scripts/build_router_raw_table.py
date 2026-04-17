# scripts/build_router_raw_table.py
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def suffix_columns(df: pd.DataFrame, suffix: str, keep: Tuple[str, ...]) -> pd.DataFrame:
    """
    Add suffix to all columns except keep.
    keep: merge keys or common fields, e.g. id/query/answer_ground_truth
    """
    rename: Dict[str, str] = {}
    for c in df.columns:
        if c in keep:
            continue
        rename[c] = f"{c}{suffix}"
    return df.rename(columns=rename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge four pipeline CSVs into a raw table for router training (with 00/01/10/11 suffixes).")
    parser.add_argument("--dataset", default="hotpotqa", type=str, help="Corresponds to result file names under outputs/<dataset>/")
    parser.add_argument("--output_root", default="outputs", type=str)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    dataset = args.dataset

    # Important: Using "code naming" here
    # Document mapping: noARAG == RAG
    files = {
        "00": out_root / dataset / "noRAG_noCoT.csv",
        "01": out_root / dataset / "noRAG_CoT.csv",
        "10": out_root / dataset / "noARAG_noCoT.csv",  # == RAG_noCoT in pdf
        "11": out_root / dataset / "noARAG_CoT.csv",    # == RAG_CoT   in pdf
    }

    for suf, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing pipeline csv for suffix {suf}: {path}")

    keep_cols = ("id", "query", "answer_ground_truth")

    df00 = pd.read_csv(files["00"])
    df01 = pd.read_csv(files["01"])
    df10 = pd.read_csv(files["10"])
    df11 = pd.read_csv(files["11"])

    # Ensure keep columns exist (raise error if missing to avoid merge errors)
    for suf, d in [("00", df00), ("01", df01), ("10", df10), ("11", df11)]:
        for k in keep_cols:
            if k not in d.columns:
                raise ValueError(f"CSV {files[suf]} missing required column: {k}")

    df00 = suffix_columns(df00, "00", keep_cols)
    df01 = suffix_columns(df01, "01", keep_cols)
    df10 = suffix_columns(df10, "10", keep_cols)
    df11 = suffix_columns(df11, "11", keep_cols)

    # Sequentially merge (using 00 as base)
    merged = df00.merge(df01, on=list(keep_cols), how="inner")
    merged = merged.merge(df10, on=list(keep_cols), how="inner")
    merged = merged.merge(df11, on=list(keep_cols), how="inner")

    out_dir = out_root / "router_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"raw_{dataset}.csv"
    merged.to_csv(out_path, index=False)

    print(f"Saved merged raw table to: {out_path}")
    print(f"Rows: {len(merged)}, Cols: {len(merged.columns)}")


if __name__ == "__main__":
    main()