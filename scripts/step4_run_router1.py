# scripts/step4_run_router1.py
import argparse
import json
from pathlib import Path
import time

import joblib
import pandas as pd

from src.pipelines.functions import (
    f_noRAG_noCoT,
    f_noRAG_CoT,
    f_noARAG_noCoT,  # RAG_noCoT
    f_noARAG_CoT,    # RAG_CoT
    get_last_features,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_csv", default="outputs/router_test/router1_results.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rag_model", default="outputs/router_train/router1_rag_logreg.pkl")
    ap.add_argument("--cot_model", default="outputs/router_train/router1_cot_logreg.pkl")
    args = ap.parse_args()

    qa = load_jsonl(Path(args.qa))
    if args.limit:
        qa = qa[: args.limit]

    corpus = load_jsonl(Path(args.corpus))
    corpus_text = "\n\n".join(x["text"] for x in corpus)

    queries = [x["query"] for x in qa]
    ids = [x["id"] for x in qa]
    gts = [x["answer_ground_truth"] for x in qa]

    # load routers
    rag_router = joblib.load(args.rag_model)
    cot_router = joblib.load(args.cot_model)

    records = []

    for i, q in enumerate(queries):
        # -------- Step 1: RAG_noCoT (for signals) --------
        ans10, rag_m10, cot_m10 = f_noARAG_noCoT([q], corpus_text)
        features = get_last_features()[0]
        nqc = rag_m10[0][0]
        ccp_no_cot = features.get("ccp", features.get("cot_se", 0.0))

        # -------- Step 2: RAG router --------
        need_rag = int(rag_router.predict([[nqc]])[0])

        # -------- Step 3: Choose noRAG / RAG's noCoT --------
        if need_rag == 1:
            base_ans = ans10[0]
            base_rag_m = rag_m10[0]
            base_cot_m = cot_m10[0]
        else:
            ans00, rag_m00, cot_m00 = f_noRAG_noCoT([q], corpus_text)
            base_ans = ans00[0]
            base_rag_m = rag_m00[0]
            base_cot_m = cot_m00[0]
            ccp_no_cot = get_last_features()[0].get("ccp", get_last_features()[0].get("cot_se", 0.0))

        # -------- Step 4: CoT router --------
        need_cot = int(cot_router.predict([[ccp_no_cot]])[0])

        # -------- Step 5: Final pipeline --------
        if need_rag == 0 and need_cot == 0:
            final_ans, rag_m, cot_m = f_noRAG_noCoT([q], corpus_text)
            suf = "00"
        elif need_rag == 0 and need_cot == 1:
            final_ans, rag_m, cot_m = f_noRAG_CoT([q], corpus_text)
            suf = "01"
        elif need_rag == 1 and need_cot == 0:
            final_ans, rag_m, cot_m = f_noARAG_noCoT([q], corpus_text)
            suf = "10"
        else:
            final_ans, rag_m, cot_m = f_noARAG_CoT([q], corpus_text)
            suf = "11"

        records.append({
            "id": ids[i],
            "query": q,
            "answer": final_ans[0],
            "gt": gts[i],
            "need_RAG": need_rag,
            "need_CoT": need_cot,
            "suffix": suf,
            "rag_nqc": rag_m[0][0],
            "rag_time": rag_m[0][1],
            "rag_tokens": rag_m[0][2],
            "cot_time": cot_m[0][1],
            "cot_tokens": cot_m[0][2],
        })

    out_df = pd.DataFrame(records)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved router1 results to {args.out_csv}")


if __name__ == "__main__":
    main()