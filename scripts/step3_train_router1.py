# scripts/step3_train_router1.py
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_binary_router(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    out_path: Path,
    test_size: float = 0.2,
):
    X = df[[feature_col]]
    y = df[target_col]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    acc = accuracy_score(y_te, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y_te, preds, average="binary", zero_division=0
    )

    joblib.dump(pipe, out_path)
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_csv",
        default="outputs/router_train/router_train_hotpotqa.csv",
        help="CSV with columns: id, NQC, CCP, need_RAG, need_CoT",
    )
    ap.add_argument("--out_dir", default="outputs/router_train")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)

    for c in ["NQC", "CCP", "need_RAG", "need_CoT"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # RAG router: NQC -> need_RAG
    rag_metrics = train_binary_router(
        df,
        feature_col="NQC",
        target_col="need_RAG",
        out_path=out_dir / "router1_rag_logreg.pkl",
        test_size=args.test_size,
    )

    # CoT router: CCP -> need_CoT
    cot_metrics = train_binary_router(
        df,
        feature_col="CCP",
        target_col="need_CoT",
        out_path=out_dir / "router1_cot_logreg.pkl",
        test_size=args.test_size,
    )

    print("[Router1 Training Results]")
    print("RAG router (NQC -> need_RAG):", rag_metrics)
    print("CoT router (CCP -> need_CoT):", cot_metrics)
    print("Saved models to:", out_dir)


if __name__ == "__main__":
    main()
