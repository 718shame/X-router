import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
FEATURE_COLUMNS = {
    "nqc": ("rag_nqc00", 0.0),
    "ccp": ("cot_se00", 0.0),
    "f_entropy": ("f_entropy00", 0.0),
    "f_top1": ("f_top100", 0.0),
    "f_divergence": ("f_divergence00", 0.0),
    "f_selfcheck": ("f_selfcheck00", 0.0),
}
PIPELINE_BITS: Dict[str, Tuple[int, int]] = {
    "00": (0, 0),
    "01": (0, 1),
    "10": (1, 0),
    "11": (1, 1),
}


def infer_timestamp(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if match:
        return match.group(1)
    return "timestamp"


def load_penalties(lambda_token: float, lambda_time: float):
    config = {}
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
    router_cfg = config.get("router", {}) if config else {}
    return (
        lambda_token if lambda_token is not None else router_cfg.get("lambda_token", 0.01),
        lambda_time if lambda_time is not None else router_cfg.get("lambda_time", 0.01),
    )


def get_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate router training labels from intermediary scores.")
    parser.add_argument("input", help="Path to intermediary_results CSV")
    parser.add_argument("--lambda-token", type=float, default=None, help="Lambda for token penalty")
    parser.add_argument("--lambda-time", type=float, default=None, help="Lambda for time penalty")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write labeled CSV",
    )
    args = parser.parse_args()

    lambda_token, lambda_time = load_penalties(args.lambda_token, args.lambda_time)

    df = pd.read_csv(args.input)

    features = {
        name: df.get(column, pd.Series([default] * len(df)))
        for name, (column, default) in FEATURE_COLUMNS.items()
    }

    utility_columns = {}
    for suffix in PIPELINE_BITS:
        answer = get_column(df, f"answer{suffix}")
        rag_tokens = get_column(df, f"rag_tokens{suffix}")
        cot_tokens = get_column(df, f"cot_tokens{suffix}")
        rag_time = get_column(df, f"rag_time{suffix}")
        cot_time = get_column(df, f"cot_time{suffix}")
        total_tokens = rag_tokens + cot_tokens
        total_time = rag_time + cot_time
        utility_columns[suffix] = answer - lambda_token * total_tokens - lambda_time * total_time

    utility_df = pd.DataFrame(utility_columns, index=df.index)
    best_suffix = utility_df.idxmax(axis=1).fillna("00")
    need_rag = best_suffix.map(lambda suffix: PIPELINE_BITS.get(str(suffix), (0, 0))[0]).astype(int)
    need_cot = best_suffix.map(lambda suffix: PIPELINE_BITS.get(str(suffix), (0, 0))[1]).astype(int)

    labeled_df = pd.DataFrame(
        {
            **{name: column for name, column in features.items()},
            "need_RAG": need_rag,
            "need_CoT": need_cot,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = infer_timestamp(Path(args.input))
    output_file = output_dir / f"labeled_results_to_train_{timestamp}.csv"
    labeled_df.to_csv(output_file, index=False)
    print(f"Saved labeled dataset to {output_file}")


if __name__ == "__main__":
    main()
