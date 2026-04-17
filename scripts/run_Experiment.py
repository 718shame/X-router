"""Utility script to run all dataset_test pipelines sequentially."""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "dataset_test"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "run_pipelines.py"


def discover_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """Return (qa, corpus) pairs inside dataset_dir."""
    qa_files = sorted(dataset_dir.glob("*_qa_unified.json"))
    pairs: List[Tuple[Path, Path]] = []
    for qa_file in qa_files:
        prefix = qa_file.name[: -len("_qa_unified.json")]
        corpus_file = dataset_dir / f"{prefix}_corpus_unified.json"
        if not corpus_file.exists():
            logging.warning("Skip %s because %s is missing", qa_file.name, corpus_file.name)
            continue
        pairs.append((qa_file, corpus_file))
    return pairs


def run_pipeline(command: List[str]) -> None:
    """Run a single pipeline command and stream output."""
    logging.info("Running: %s", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run scripts/run_pipelines.py sequentially for every dataset under dataset/dataset_test",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Directory containing *_qa_unified.json and *_corpus_unified.json files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for run_pipelines.py outputs (defaults to each dataset subdir)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of QA rows passed to run_pipelines.py",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for this orchestrator script",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter used to launch run_pipelines.py (defaults to current interpreter)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    dataset_dir = Path(args.dataset_dir).resolve()
    pairs = discover_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(f"No dataset pairs found under {dataset_dir}")

    logging.info("Discovered %d dataset pairs", len(pairs))

    for qa_file, corpus_file in pairs:
        cmd = [
            args.python_exe,
            str(DEFAULT_RUNNER),
            "--qa",
            str(qa_file),
            "--corpus",
            str(corpus_file),
            "--output-dir",
            args.output_dir,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        run_pipeline(cmd)

    logging.info("All datasets finished successfully")


if __name__ == "__main__":
    main()
