import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_CANDIDATES: List[str] = [
    "nqc",
    "ccp",
    "f_entropy",
    "f_top1",
    "f_divergence",
    "f_selfcheck",
]


def _preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("features", "passthrough", feature_columns),
        ]
    )


def build_models(feature_columns: List[str]) -> Dict[str, Pipeline]:
    if not feature_columns:
        raise ValueError("feature_columns is empty, unable to train router")

    models = {
        "router_logit": Pipeline(
            steps=[
                ("preprocess", _preprocessor(feature_columns)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "router_tree": Pipeline(
            steps=[
                ("preprocess", _preprocessor(feature_columns)),
                ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "router_mlp": Pipeline(
            steps=[
                ("preprocess", _preprocessor(feature_columns)),
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)),
            ]
        ),
    }
    return models


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_and_save(
    models: Dict[str, Pipeline],
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    test_size: float,
    feature_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    X = df[feature_columns]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    scores: Dict[str, Dict[str, float]] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        scores[name] = metrics
        artifact_path = output_dir / f"{name}_{target}.pkl"
        joblib.dump(pipeline, artifact_path)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router classifiers for RAG and CoT decisions.")
    parser.add_argument("input", help="Path to labeled_results_to_train CSV")
    parser.add_argument("--output-dir", default="artifacts/router", help="Directory to store trained models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    args = parser.parse_args()

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required_cols = {"need_RAG", "need_CoT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input dataset: {missing}")

    feature_columns = [col for col in FEATURE_CANDIDATES if col in df.columns]
    if len(feature_columns) < 2:
        raise ValueError(
            f"Insufficient training features, at least two columns required. Detected {feature_columns}, please confirm the labeled CSV contains {FEATURE_CANDIDATES}"
        )

    models = build_models(feature_columns)
    results = {}

    for target in ["need_RAG", "need_CoT"]:
        metrics = train_and_save(models, df, target, output_dir, args.test_size, feature_columns)
        results[target] = metrics

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"training_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Training complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()