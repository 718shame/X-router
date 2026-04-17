import argparse
import re
from pathlib import Path

import pandas as pd

from src.metrics.metrics import f1_score


SUFFIXES = ["NR0", "NR1", "00", "10", "01", "11"]


def infer_timestamp(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if match:
        return match.group(1)
    return "timestamp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw pipeline outputs to F1 score CSV.")
    parser.add_argument("input", help="Path to raw_results timestamped CSV")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write intermediary CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    scores_df = df.copy()

    for suffix in SUFFIXES:
        pred_column = f"answer{suffix}"
        backup_column = f"pred_{pred_column}"
        scores_df[backup_column] = scores_df[pred_column]
        scores_df[pred_column] = scores_df.apply(
            lambda row: f1_score(str(row[pred_column]), str(row.get("answer_ground_truth", ""))), axis=1
        )

    timestamp = infer_timestamp(input_path)
    output_file = output_dir / f"intermediary_results_{timestamp}.csv"
    scores_df.to_csv(output_file, index=False)
    print(f"Saved intermediary scores to {output_file}")


if __name__ == "__main__":
    main()
