import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def eval_binary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router_train_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.router_train_csv)

    # Features: 1 dimension is sufficient (consistent with your existing Router1 design)
    X_rag = df[["rag_nqc"]].fillna(0.0).astype(float)
    y_rag = df["need_RAG"].astype(int)

    X_cot = df[["ccp"]].fillna(0.0).astype(float)
    y_cot = df["need_CoT"].astype(int)

    # stratified split
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_rag, y_rag, test_size=0.2, random_state=args.random_state, stratify=y_rag
    )
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_cot, y_cot, test_size=0.2, random_state=args.random_state, stratify=y_cot
    )

    rag_clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    cot_clf = LogisticRegression(class_weight="balanced", max_iter=2000)

    rag_clf.fit(Xr_tr, yr_tr)
    cot_clf.fit(Xc_tr, yc_tr)

    rag_pred = rag_clf.predict(Xr_te)
    cot_pred = cot_clf.predict(Xc_te)

    rag_metrics = eval_binary(yr_te, rag_pred)
    cot_metrics = eval_binary(yc_te, cot_pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rag_path = out_dir / "router1_rag_logreg.pkl"
    cot_path = out_dir / "router1_cot_logreg.pkl"
    joblib.dump(rag_clf, rag_path)
    joblib.dump(cot_clf, cot_path)

    print("[Step3-lambda] Saved models:")
    print(" -", rag_path)
    print(" -", cot_path)
    print("[Router1 Training Metrics]")
    print(" RAG router:", rag_metrics)
    print(" CoT router:", cot_metrics)


if __name__ == "__main__":
    main()