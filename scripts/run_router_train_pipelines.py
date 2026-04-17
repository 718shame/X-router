# scripts/run_router_train_pipelines.py
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow script to import src/*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing pipeline functions in your project
# Important mapping: code noARAG_* == document RAG_*
from src.pipelines.functions import (  # noqa: E402
    f_noRAG_noCoT,
    f_noRAG_CoT,
    f_noARAG_noCoT,  # == RAG_noCoT in pdf
    f_noARAG_CoT,    # == RAG_CoT   in pdf
    get_last_features,
)

logger = logging.getLogger("router-train-runner")


def load_jsonl(path: Path) -> List[Dict]:
    """One JSON per line (jsonl)."""
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def load_corpus_jsonl(path: Path) -> Dict[str, str]:
    """corpus jsonl: {"id":..., "text":...}"""
    corpus: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            entry = json.loads(line)
            cid = entry.get("id")
            txt = entry.get("text", "")
            if cid is not None:
                corpus[cid] = txt
    return corpus


def persist_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_triplet(metrics, n: int):
    """Ensure metrics length is n, each element is (a,b,c) structure; pad with default if insufficient."""
    default = (0.0, 0.0, 0)
    if metrics is None:
        return [default] * n
    if len(metrics) < n:
        metrics = list(metrics) + [default] * (n - len(metrics))
    return metrics[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run four pipelines on train_router for router training data construction."
    )
    parser.add_argument("--qa", required=True, type=str, help="QA jsonl path")
    parser.add_argument("--corpus", required=True, type=str, help="corpus jsonl path")
    parser.add_argument("--dataset_name", default="hotpotqa", type=str, help="output directory name")
    parser.add_argument("--limit", default=500, type=int, help="maximum number of QAs to run")
    parser.add_argument("--output_root", default="outputs", type=str, help="output root directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    qa_path = Path(args.qa)
    corpus_path = Path(args.corpus)

    qa = load_jsonl(qa_path)[: args.limit]
    corpus = load_corpus_jsonl(corpus_path)

    # Construct corpus_text for pipeline (consistent with scripts/run_pipelines.py)
    corpus_text = "\n\n".join(corpus.values())

    # Construct queries / base_df
    queries = [x.get("query", "") for x in qa]
    ids = [x.get("id", "") for x in qa]
    answers_gt = [x.get("answer_ground_truth", "") for x in qa]
    n = len(queries)

    base_df = pd.DataFrame(
        {
            "id": ids,
            "query": queries,
            "answer_ground_truth": answers_gt,
        }
    )

    output_dir = Path(args.output_root) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Four pipelines (strictly aligned with the "first four" in the pdf)
    # suffix mapping (used by subsequent build_router_raw_table):
    # 00: noRAG_noCoT
    # 01: noRAG_CoT
    # 10: noARAG_noCoT (== document RAG_noCoT)
    # 11: noARAG_CoT   (== document RAG_CoT)
    pipelines = [
        ("noRAG_noCoT", f_noRAG_noCoT),
        ("noRAG_CoT", f_noRAG_CoT),
        ("noARAG_noCoT", f_noARAG_noCoT),
        ("noARAG_CoT", f_noARAG_CoT),
    ]

    start_all = time.perf_counter()
    produced = []

    for name, func in pipelines:
        logger.info("Running pipeline: %s", name)
        t0 = time.perf_counter()

        # Aligned with your repo's scripts/run_pipelines.py: return triplet
        answers, rag_metrics, cot_metrics = func(queries, corpus_text)

        # features taken from global cache (your function is named get_last_features)
        features_list = get_last_features() or []

        # Unify length
        if answers is None:
            answers = [""] * n
        if len(answers) < n:
            answers = list(answers) + [""] * (n - len(answers))
        answers = answers[:n]

        rag_metrics = safe_triplet(rag_metrics, n)
        cot_metrics = safe_triplet(cot_metrics, n)

        # Collect feature keys
        feature_keys = set()
        for fm in features_list:
            if isinstance(fm, dict):
                feature_keys.update(fm.keys())

        # Assemble output df
        df = base_df.copy()
        col_answer = []
        col_rag_nqc, col_rag_time, col_rag_tokens = [], [], []
        col_cot_se, col_cot_time, col_cot_tokens = [], [], []
        feature_cols: Dict[str, List[float]] = {k: [] for k in feature_keys}

        for i in range(n):
            col_answer.append(answers[i])

            rag = rag_metrics[i]
            col_rag_nqc.append(rag[0])
            col_rag_time.append(rag[1])
            col_rag_tokens.append(rag[2])

            cot = cot_metrics[i]
            col_cot_se.append(cot[0])
            col_cot_time.append(cot[1])
            col_cot_tokens.append(cot[2])

            fm = features_list[i] if i < len(features_list) and isinstance(features_list[i], dict) else {}
            for k in feature_keys:
                feature_cols[k].append(fm.get(k, 0.0))

        df["answer"] = col_answer
        df["rag_nqc"] = col_rag_nqc
        df["rag_time"] = col_rag_time
        df["rag_tokens"] = col_rag_tokens
        df["cot_se"] = col_cot_se
        df["cot_time"] = col_cot_time
        df["cot_tokens"] = col_cot_tokens

        for k, v in feature_cols.items():
            df[k] = v

        out_file = output_dir / f"{name}.csv"
        persist_csv(df, out_file)
        produced.append(out_file)

        logger.info("Finished %s in %.2fs, wrote: %s", name, time.perf_counter() - t0, out_file)

    logger.info("All done in %.2fs", time.perf_counter() - start_all)
    print("Saved results to:")
    for p in produced:
        print(f" - {p}")


if __name__ == "__main__":
    main()