import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure src package can be imported when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.functions import (  # noqa: E402
    f_ARAG_CoT,   # actual code doesn't run
    f_ARAG_noCoT, # actual code doesn't run
    f_noRAG_CoT,
    f_noRAG_noCoT,
    f_noARAG_CoT,
    f_noARAG_noCoT,
    get_last_features,
)


logger = logging.getLogger(__name__)


def persist_partial_results(frame: pd.DataFrame, path: Path) -> None:
    """Write current dataframe snapshot to disk for crash-safe progress."""
    if path.exists():
        path.unlink()
    frame.to_csv(path, index=False, mode="w")
    logger.info("Writing CSV snapshot: %s (columns=%d)", path, len(frame.columns))


def load_jsonl(path: Path) -> List[Dict]:
    logger.info("Loading QA: %s", path)
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def load_corpus(path: Path) -> Dict[str, str]:
    logger.info("Loading corpus: %s", path)
    corpus: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            entry = json.loads(line)
            corpus[entry["id"]] = entry.get("text", "")
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run six RAG/CoT pipelines on HotpotQA (dataset_test) and export raw result CSVs."
    )
    parser.add_argument(
        "--qa",
        default="dataset/dataset_test/hotpotqa_qa_unified.json",
        help="HotpotQA QA jsonl path (default: dataset/dataset_test/hotpotqa_qa_unified.json)",
    )
    parser.add_argument(
        "--corpus",
        default="dataset/dataset_test/hotpotqa_corpus_unified.json",
        help="HotpotQA corpus jsonl path (default: dataset/dataset_test/hotpotqa_corpus_unified.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write CSV results (default: outputs)",
    )

    #parser.add: This parser controls which pipeline to call
    #For example, input noRAG_noCoT to run that pipeline



    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of QA items")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Reduce httpx/httpcore noise logs to avoid flooding the console
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    qa_path = Path(args.qa)
    corpus_path = Path(args.corpus)

    dataset_name = qa_path.stem
    if dataset_name.endswith("_qa_unified"):
        dataset_name = dataset_name[: -len("_qa_unified")]

    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_items = load_jsonl(qa_path)
    if args.limit:
        qa_items = qa_items[: args.limit]
    if not qa_items:
        raise ValueError(f"No QA items found in {qa_path}")

    corpus = load_corpus(corpus_path)
    corpus_text = "\n\n".join(corpus.values())
    queries = [entry.get("query", "") for entry in qa_items]
    answers_gt = [entry.get("answer_ground_truth", "") for entry in qa_items]

    logger.info(
        "Preparing to run pipelines: samples=%d (limit=%s), corpus_docs=%d, output=%s",
        len(queries),
        args.limit,
        len(corpus),
        output_dir,
    )

    base_df = pd.DataFrame(
        [
            {
                "id": entry.get("id"),
                "answer_ground_truth": answer_gt,
            }
            for entry, answer_gt in zip(qa_items, answers_gt)
        ]
    )

    pipelines = [
        ("noRAG_noCoT", f_noRAG_noCoT),
        ("noRAG_CoT", f_noRAG_CoT),
        ("noARAG_noCoT", f_noARAG_noCoT),
        ("ARAG_noCoT", f_ARAG_noCoT),
        ("noARAG_CoT", f_noARAG_CoT),
        ("ARAG_CoT", f_ARAG_CoT),
    ]

    total_start = time.perf_counter()
    produced_files = []
    for pipeline_name, func in pipelines:
        logger.info("Starting to execute pipeline %s", pipeline_name)
        t0 = time.perf_counter()
        answers, rag_metrics, cot_metrics = func(queries, corpus_text)
        logger.info("Pipeline %s execution completed, elapsed time: %.2fs", pipeline_name, time.perf_counter() - t0)
        features_list = get_last_features() or []
        feature_keys = set()
        for feature_map in features_list:
            feature_keys.update(feature_map.keys())

        df = base_df.copy()
        new_columns: Dict[str, List[object]] = {}
        for idx in range(len(df)):
            answer = answers[idx] if idx < len(answers) else ""
            rag_metric = rag_metrics[idx] if idx < len(rag_metrics) else (0.0, 0.0, 0)
            cot_metric = cot_metrics[idx] if idx < len(cot_metrics) else (0.0, 0.0, 0)

            new_columns.setdefault("answer", []).append(answer)
            new_columns.setdefault("rag_nqc", []).append(rag_metric[0])
            new_columns.setdefault("rag_time", []).append(rag_metric[1])
            new_columns.setdefault("rag_tokens", []).append(rag_metric[2])
            new_columns.setdefault("cot_se", []).append(cot_metric[0])
            new_columns.setdefault("cot_time", []).append(cot_metric[1])
            new_columns.setdefault("cot_tokens", []).append(cot_metric[2])

            feature_map = features_list[idx] if idx < len(features_list) else {}
            for feature_name in feature_keys:
                new_columns.setdefault(feature_name, []).append(feature_map.get(feature_name, 0.0))

        for col_name, col_data in new_columns.items():
            df[col_name] = col_data

        pipeline_file = output_dir / f"{pipeline_name}.csv"
        logger.info("Pipeline %s writing number of columns=%d", pipeline_name, len(new_columns))
        persist_partial_results(df, pipeline_file)
        produced_files.append(pipeline_file)

    logger.info("All pipelines completed, total elapsed time: %.2fs", time.perf_counter() - total_start)
    logger.info("Generated CSVs in this run: %s", ", ".join(str(p) for p in produced_files))
    print("Saved results to:")
    for path in produced_files:
        print(f" - {path}")


if __name__ == "__main__":
    main()