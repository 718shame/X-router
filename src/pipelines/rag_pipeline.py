#src/pipelines/rag_pipeline.py
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import os

from src.clients.API_BASE import API_BASEClient
from src.data.corpus_manager import CorpusArtifacts, CorpusManager
from src.metrics.metrics import pairwise_cosine
from src.pipelines.semantic_utils import compute_ccp_metric
from Router.cognitive_LLM.src.metrics.nll import NLIPredictor, NLIConfig
from src.rag.generator import LLMGenerator
from src.rag.retriever import HybridRetriever, RetrievalConfig, RetrievalResult

_nli_predictor = None
def get_nli():
    global _nli_predictor
    if _nli_predictor is None:
        _nli_predictor = NLIPredictor(NLIConfig())
    return _nli_predictor

logger = logging.getLogger(__name__)


HYDE_PROMPT = """You are an expert knowledge base.
Given the following question, write a concise hypothetical passage that would help answer it.
Question: {query}
Passage:"""

# Maximum query concurrency, controlled via environment variable, default 16
# export RAG_MAX_QUERY_CONCURRENCY=8 or similar to modify
MAX_QUERY_CONCURRENCY = int(os.getenv("RAG_MAX_QUERY_CONCURRENCY", "3"))


async def _gather_with_concurrency(limit: int, coros: Sequence[asyncio.Future]):
    """
    Execute a group of coroutines with given concurrency limit to avoid creating too many concurrent tasks at once.
    When limit <= 0, it is equivalent to regular asyncio.gather.
    """
    if limit is None or limit <= 0:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(limit)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    wrapped = [sem_coro(c) for c in coros]
    return await asyncio.gather(*wrapped)


@dataclass
class PipelineConfig:
    use_hyde: bool
    use_hybrid: bool
    use_rerank: bool
    use_mmr: bool
    use_cot: bool
    semantic_samples: int
    semantic_temperature: float


@dataclass
class PipelineOutputs:
    answers: List[str]
    rag_metrics: List[Tuple[float, float, int]]
    cot_metrics: List[Tuple[float, float, int]]
    features: List[Dict[str, float]]


class RAGPipeline:
    def __init__(
        self,
        corpus_manager: CorpusManager,
        retriever: HybridRetriever,
        generator: LLMGenerator,
        client: API_BASEClient,
        retrieval_config: RetrievalConfig,
        context_max_tokens: int,
    ) -> None:
        self._corpus_manager = corpus_manager
        self._retriever = retriever
        self._generator = generator
        self._client = client
        self._retrieval_config = retrieval_config
        self._context_max_tokens = context_max_tokens

    async def _hyde(self, queries: Sequence[str]) -> List[str]:
        async def _generate(query: str) -> str:
            prompt = HYDE_PROMPT.format(query=query)
            response = await self._client.generate(
                prompt, model=None, temperature=0.5, max_tokens=1024
            )
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text.strip()

        tasks = [_generate(q) for q in queries]
        # Use gather with concurrency limit to avoid too many HYDEs launching simultaneously
        return await _gather_with_concurrency(MAX_QUERY_CONCURRENCY, tasks)

    def _approx_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _build_context(self, passages: Sequence[str]) -> Tuple[str, int]:
        assembled = []
        token_count = 0
        for idx, passage in enumerate(passages):
            tokens = self._approx_tokens(passage)
            if token_count + tokens > self._context_max_tokens:
                break
            assembled.append(f"[Doc {idx+1}] {passage}")
            token_count += tokens
        return "\n\n".join(assembled), token_count

    async def _ccp_metric(
        self,
        context: str,
        query: str,
        use_cot: bool,
        temperature: float,
    ) -> float:
        nli = get_nli()
        ccp_max_tokens = getattr(self._retrieval_config, "ccp_max_tokens", 256)
        ccp = await compute_ccp_metric(
            client=self._client,
            generator=self._generator,
            context=context,
            query=query,
            use_cot=use_cot,
            temperature=temperature,
            top_logprobs=10,
            max_tokens=ccp_max_tokens,
            nli=nli,
        )
        return ccp


    async def run(
        self,
        queries: Sequence[str],
        corpus: str,
        config: PipelineConfig,
    ) -> PipelineOutputs:
        # Prepare the entire corpus only once


        logger.info(
                f"RAGPipeline.run starting: queries={len(queries)}, "
                f"MAX_QUERY_CONCURRENCY={MAX_QUERY_CONCURRENCY}, "
                f"use_hyde={config.use_hyde}, use_cot={config.use_cot}, "
                f"semantic_samples={config.semantic_samples}"
            )

        logger.info("Start preparing corpus")
        artifacts = await self._corpus_manager.prepare(corpus)
        logger.info("Corpus preparation completed")
        # (Optional) HYDE generation
        hyde_passages: List[str] = []
        hyde_overhead_ms = 0.0

        if config.use_hyde:
            logger.info("Start hyde")
            hyde_start = time.perf_counter()
            hyde_passages = await self._hyde(queries)
            hyde_overhead_ms = round((time.perf_counter() - hyde_start) * 1000.0, 6)
            if queries:
                hyde_overhead_ms /= len(queries)
        else:
            logger.info("Skip hyde")

        if config.use_hyde:
            embed_inputs = [f"{q}\n{h}" for q, h in zip(queries, hyde_passages)]
        else:
            embed_inputs = list(queries)
        logger.info("End hyde")
        # Batch query embedding
        logger.info("Start embedding")
        query_vectors = await self._retriever.query_embedding(embed_inputs)
        logger.info("End embedding")
        
        async def _process(idx: int) -> Tuple[str, Tuple[float, float, int], Tuple[float, float, int], Dict[str, float]]:
            query = queries[idx]
            query_vec = query_vectors[idx]

            retrieval_start = time.perf_counter()
            retrieval_result: RetrievalResult = await self._retriever.retrieve(
                query=query,
                query_vec=query_vec,
                artifacts=artifacts,
                use_rerank=config.use_rerank,
                use_sparse=config.use_hybrid,
                apply_mmr=config.use_mmr,
            )
            retrieval_time = round((time.perf_counter() - retrieval_start) * 1000.0, 6) + hyde_overhead_ms
            logger.info(f"Processing retrieval for index {idx} completed, took {retrieval_time}ms")
            context, context_tokens = self._build_context(retrieval_result.passages)

            gen_start = time.perf_counter()
            answer, total_tokens, completion_tokens = await self._generator.generate(
                context=context,
                query=query,
                use_cot=config.use_cot,
            )
            generation_time = round((time.perf_counter() - gen_start) * 1000.0, 6)
            logger.info(f"Processing generation for index {idx} completed, took {generation_time}ms")

            logger.info(f"Index {idx} starts semantic metric")
            ccp = await self._ccp_metric(
                context=context,
                query=query,
                use_cot=config.use_cot,
                temperature=config.semantic_temperature,
            )

            selfcheck_score = 1.0 - ccp
            semantic_ent = ccp
            logger.info(f"Processing semantic metric for index {idx} completed, CCP={semantic_ent:.4f}, selfcheck={selfcheck_score:.4f}")


            dense_scores = retrieval_result.dense_scores[: self._retrieval_config.dense_top_k]
            dense_scores = np.nan_to_num(
                np.array(dense_scores, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).tolist()

            if dense_scores:
                probs = np.exp(dense_scores - np.max(dense_scores))
                probs = probs / (np.sum(probs) + 1e-9)
                entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
                normalized_entropy = entropy / (np.log(len(probs)) + 1e-9)
            else:
                normalized_entropy = 0.0

            if len(dense_scores) > 1:
                top1_gap = float(dense_scores[0] - np.mean(dense_scores[1:]))
            elif dense_scores:
                top1_gap = float(dense_scores[0])
            else:
                top1_gap = 0.0

            divergence = 0.0
            if len(retrieval_result.passage_ids) > 1:
                candidate_vecs = artifacts.embeddings[retrieval_result.passage_ids]
                sims = pairwise_cosine(list(candidate_vecs))
                if sims:
                    divergence = float(np.mean(1 - np.array(sims)))

            feature_vector = {
                "f_entropy": normalized_entropy,
                "f_top1": top1_gap,
                "f_divergence": divergence,
                "ccp": ccp,
                "f_selfcheck": selfcheck_score,
                "nqc": retrieval_result.nqc,
            }

            rag_metric = (
                retrieval_result.nqc,
                retrieval_time,
                context_tokens,
            )
            cot_metric = (
                ccp,
                generation_time,
                total_tokens,
            )
            return answer, rag_metric, cot_metric, feature_vector

        # Process queries with "concurrency throttling"
        logger.info("Start retrieval and generation")
        coros = [_process(i) for i in range(len(queries))]

        results = await _gather_with_concurrency(MAX_QUERY_CONCURRENCY, coros)
        logger.info("End retrieval and generation")

        answers = [item[0] for item in results]
        rag_metrics = [item[1] for item in results]
        cot_metrics = [item[2] for item in results]
        features = [item[3] for item in results]
        return PipelineOutputs(
            answers=answers,
            rag_metrics=rag_metrics,
            cot_metrics=cot_metrics,
            features=features,
        )