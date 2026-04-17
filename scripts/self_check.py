"""Local self-check script: Run through the main pipeline without calling external APIs."""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from src.data.corpus_manager import CorpusManager
from src.pipelines.rag_pipeline import PipelineConfig, RAGPipeline
from src.rag.generator import GenerationTemplate, LLMGenerator
from src.rag.retriever import HybridRetriever, RetrievalConfig


class DummyClient:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def embedding(self, texts: Iterable[str]) -> Dict[str, Any]:
        vectors = []
        for idx, text in enumerate(texts):
            seed = sum(ord(ch) for ch in text)
            random.seed(seed)
            vec = [math.sin(seed + i) for i in range(self.dim)]
            vectors.append({"index": idx, "embedding": vec})
        return {"data": vectors}

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 128,
        stop: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        answer = "Answer: dummy response"
        return {
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    async def rerank(self, query: str, docs: List[str], top_n: int = 20) -> Dict[str, Any]:
        results = []
        for idx, doc in enumerate(docs[:top_n]):
            score = float(len(set(query.split()) & set(doc.split())))
            results.append({"index": idx, "relevance_score": score})
        return {"results": results}

    async def close(self) -> None:  # pragma: no cover - compatible with real client interface
        return


@dataclass
class DummySiliconConfig:
    base_url: str = "https://dummy"
    timeout: int = 30
    max_retries: int = 1
    backoff_factor: float = 1.0
    concurrency: int = 4


async def run_self_check() -> None:
    client = DummyClient(dim=8)

    corpus_text = """Doc1: Retrieval augmented generation is useful.
Doc2: Chain of thought reasoning helps complex QA.
Doc3: Hybrid search blends sparse and dense signals."""

    corpus_manager = CorpusManager(
        client=client,
        embedding_dir="artifacts/self_check/embeddings",
        index_dir="artifacts/self_check/index",
        chunk_size=40,
        chunk_overlap=10,
        embedding_batch_size=4,
        use_gpu=False,
    )

    retrieval = HybridRetriever(
        client=client,
        retrieval_config=RetrievalConfig(
            dense_top_k=5,
            sparse_top_k=5,
            final_top_k=3,
            mmr_lambda=0.5,
        ),
    )
    generator = LLMGenerator(
        client=client,
        template=GenerationTemplate(
            default="Answer: {query_str}",
            cot="Answer: {query_str}",
        ),
        max_new_tokens=32,
    )

    pipeline = RAGPipeline(
        corpus_manager=corpus_manager,
        retriever=retrieval,
        generator=generator,
        client=client,
        retrieval_config=retrieval._config,
        context_max_tokens=200,
    )

    queries = ["Why combine RAG and CoT", "When is advanced retrieval needed"]

    for cfg_name, cfg in {
        "Naive": PipelineConfig(False, False, False, False, False, 2, 0.2),
        "Advanced": PipelineConfig(True, True, True, True, True, 2, 0.2),
    }.items():
        outputs = await pipeline.run(queries, corpus_text, cfg)
        print(f"[{cfg_name}] answers: {outputs.answers}")
        print(f"[{cfg_name}] RAG metrics: {outputs.rag_metrics}")
        print(f"[{cfg_name}] CoT metrics: {outputs.cot_metrics}")


if __name__ == "__main__":
    asyncio.run(run_self_check())