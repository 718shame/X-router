#src/pipelines/functions.py
import asyncio
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import yaml
from dotenv import load_dotenv

from src.clients.API_BASE import API_BASEClient, API_BASEConfig
from src.data.corpus_manager import CorpusManager
from src.pipelines.rag_pipeline import PipelineConfig, RAGPipeline
from src.pipelines.semantic_utils import compute_ccp_metric
from Router.cognitive_LLM.src.metrics.nll import NLIPredictor, NLIConfig
from src.rag.generator import GenerationTemplate, LLMGenerator
from src.rag.retriever import HybridRetriever, RetrievalConfig
from src.utils.env import APIKeyPool

logger = logging.getLogger(__name__)

LAST_FEATURES: List[Dict[str, float]] = []

# Global NLI model to avoid reloading for each sample
_NLI_PREDICTOR: Optional[NLIPredictor] = None


def get_nli_predictor() -> NLIPredictor:
    global _NLI_PREDICTOR
    if _NLI_PREDICTOR is None:
        _NLI_PREDICTOR = NLIPredictor(NLIConfig())
    return _NLI_PREDICTOR

load_dotenv()
# Reduce default concurrency to avoid overwhelming the service with public API 503 errors; can be overridden with environment variable
NO_RAG_MAX_CONCURRENCY = int(os.getenv("NO_RAG_MAX_CONCURRENCY", "4"))

_DEFAULT_TEMPLATE = (
    "\nYou are a helpful assistant.\n"
    "Answer the provided question given the context information and not prior knowledge."
    " Be concise and precise! Provide only the essential information directly answering the question."
    " Avoid unnecessary elaboration or background information.\n"
    "Context information is below. \n---------------------\n{context_str}\n---------------------\n"
    "Question: {query_str}\n\nProvide your answer in the format:\nAnswer: [your answer here]"
)

_COT_TEMPLATE = (
    "\nYou are a helpful assistant.\n"
    "Answer the provided question given the context step-by-step.\n"
    "Context information is below. \n---------------------\n{context_str}\n---------------------\n"
    "Question: {query_str}\n\nPlease think through this step-by-step and then provide your final answer in the format:\nAnswer: [your answer here]"
)


@lru_cache(maxsize=1)
def _load_settings() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _create_client(settings: Dict[str, Any]) -> API_BASEClient:
    silicon_cfg = settings["API_BASE"].copy()
    base_url_override = os.getenv("API_BASE_BASE_URL")
    if base_url_override:
        silicon_cfg["base_url"] = base_url_override
    client_config = API_BASEConfig(
        base_url=silicon_cfg["base_url"],
        timeout=silicon_cfg["timeout"],
        max_retries=silicon_cfg["max_retries"],
        backoff_factor=silicon_cfg["backoff_factor"],
        concurrency=silicon_cfg["concurrency"],
    )
    key_pool = APIKeyPool()
    models = silicon_cfg["models"]
    models["max_model_len"] = 8192
    models["token_safety_margin"] = 64  
    models["embedding_cfg"] = silicon_cfg.get("embedding", {})
    models["rerank_cfg"] = silicon_cfg.get("rerank", {})
    return API_BASEClient(client_config, models, key_pool=key_pool)


async def _run_pipeline(
    queries: Sequence[str],
    corpus: str,
    pipeline_cfg: PipelineConfig,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    settings = _load_settings()
    project_cfg = settings["project"]
    retrieval_cfg = settings["retrieval"]
    metrics_cfg = settings["metrics"]
    client = _create_client(settings)

    retrieval = RetrievalConfig(
        dense_top_k=retrieval_cfg["dense_top_k"],
        sparse_top_k=retrieval_cfg["sparse_top_k"],
        final_top_k=retrieval_cfg["final_top_k"],
        mmr_lambda=retrieval_cfg["mmr_lambda"],
    )

    corpus_manager = CorpusManager(
        client=client,
        embedding_dir=project_cfg["embedding_dir"],
        index_dir=project_cfg["index_dir"],
        chunk_size=retrieval_cfg["chunk_size"],
        chunk_overlap=retrieval_cfg["chunk_overlap"],
        embedding_batch_size=retrieval_cfg["embedding_batch_size"],
        use_gpu=retrieval_cfg.get("use_gpu", True),
    )

    retriever = HybridRetriever(client=client, retrieval_config=retrieval)
    generator = LLMGenerator(
        client=client,
        template=GenerationTemplate(default=_DEFAULT_TEMPLATE, cot=_COT_TEMPLATE),
        max_new_tokens=settings["retrieval"].get("max_new_tokens", 1024),
    )

    pipeline = RAGPipeline(
        corpus_manager=corpus_manager,
        retriever=retriever,
        generator=generator,
        client=client,
        retrieval_config=retrieval,
        context_max_tokens=retrieval_cfg["max_context_tokens"],
    )

    global LAST_FEATURES
    try:
        outputs = await pipeline.run(
            queries=queries,
            corpus=corpus,
            config=PipelineConfig(
                use_hyde=pipeline_cfg.use_hyde,
                use_hybrid=pipeline_cfg.use_hybrid,
                use_rerank=pipeline_cfg.use_rerank,
                use_mmr=pipeline_cfg.use_mmr,
                use_cot=pipeline_cfg.use_cot,
                semantic_samples=metrics_cfg["semantic_entropy_samples"],
                semantic_temperature=metrics_cfg["semantic_entropy_temperature"],
            ),
        )
        LAST_FEATURES = outputs.features
    finally:
        await client.close()

    return outputs.answers, outputs.rag_metrics, outputs.cot_metrics


async def _run_no_rag_pipeline(
    queries: Sequence[str],
    use_cot: bool,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    settings = _load_settings()
    metrics_cfg = settings["metrics"]
    retrieval_cfg = settings["retrieval"]
    client = _create_client(settings)
    generator = LLMGenerator(
        client=client,
        template=GenerationTemplate(default=_DEFAULT_TEMPLATE, cot=_COT_TEMPLATE),
        max_new_tokens=retrieval_cfg.get("max_new_tokens", 1024),
    )

    logger.info(
        "[noRAG pipeline] Starting: queries=%d, use_cot=%s, semantic_samples=%d, max_concurrency=%d",
        len(queries),
        use_cot,
        metrics_cfg["semantic_entropy_samples"],
        NO_RAG_MAX_CONCURRENCY,
    )

    async def _process_one(idx: int, query: str):
        """Process a single query, return (idx, answer, rag_metric, cot_metric, feature_vector)."""
        gen_start = time.perf_counter()
        # First generate an answer normally (for downstream evaluation; CCP will make another call with logprobs)
        answer, total_tokens, _ = await generator.generate(
            context="",
            query=query,
            use_cot=use_cot,
        )
        generation_time = round((time.perf_counter() - gen_start) * 1000.0, 6)

        # Use CCP metric (white-box logits) to replace the original Semantic Entropy
        nli = get_nli_predictor()
        ccp = await compute_ccp_metric(
            client=client,
            generator=generator,
            context="",  # no context for noRAG case
            query=query,
            use_cot=use_cot,
            # Reuse the original temperature configuration; you can also add ccp_temperature separately in default.yaml
            temperature=metrics_cfg["semantic_entropy_temperature"],
            top_logprobs=metrics_cfg.get("ccp_top_logprobs", 10),
            max_tokens=retrieval_cfg.get("ccp_max_tokens", 256),
            nli=nli,
        )

        # The feature_vector is a feature collection for router training
        # Keep original fields and add "ccp" (and use ccp to overwrite "f_semantic_entropy")
        feature_vector = {
            "f_entropy": 0.0,
            "f_top1": 0.0,
            "f_divergence": 0.0,
            # For backward compatibility, still provide f_semantic_entropy but with CCP value
            "f_semantic_entropy": ccp,
            # You can think of selfcheck as "1-CCP", higher values indicate more confidence
            "f_selfcheck": 1.0 - ccp,
            "nqc": 0.0,
            # Explicitly provide CCP feature, router/train.py can use it directly
            "ccp": ccp,
        }

        logger.info(
            "[noRAG pipeline] idx=%d completed, gen_t=%.2fms, CCP=%.4f",
            idx,
            generation_time,
            ccp,
        )

        rag_metric = (0.0, 0.0, 0)
        # Switch the original semantic_entropy position to CCP
        cot_metric = (ccp, generation_time, total_tokens)
        return idx, answer, rag_metric, cot_metric, feature_vector


    # Semaphore controls concurrency
    semaphore = asyncio.Semaphore(NO_RAG_MAX_CONCURRENCY)

    async def _sem_wrapper(idx: int, query: str):
        async with semaphore:
            return await _process_one(idx, query)

    global LAST_FEATURES
    try:
        # Create all tasks
        tasks = [_sem_wrapper(i, q) for i, q in enumerate(queries)]
        logger.info("[noRAG pipeline] Created %d tasks, starting concurrent execution", len(tasks))

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # results are out of order, sort by idx to restore original order
        results.sort(key=lambda x: x[0])

        answers: List[str] = []
        rag_metrics: List[Tuple[float, float, int]] = []
        cot_metrics: List[Tuple[float, float, int]] = []
        features: List[Dict[str, float]] = []

        for idx, answer, rag_metric, cot_metric, feature_vector in results:
            answers.append(answer)
            rag_metrics.append(rag_metric)
            cot_metrics.append(cot_metric)
            features.append(feature_vector)

        LAST_FEATURES = features
        logger.info("[noRAG pipeline] All completed, total samples=%d", len(queries))
    finally:
        await client.close()

    return answers, rag_metrics, cot_metrics



def _run_sync(
    queries: Sequence[str],
    corpus: str,
    cfg: PipelineConfig,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    return asyncio.run(_run_pipeline(queries, corpus, cfg))


def _run_no_rag_sync(
    queries: Sequence[str],
    use_cot: bool,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    return asyncio.run(_run_no_rag_pipeline(queries, use_cot))


def get_last_features() -> List[Dict[str, float]]:
    return LAST_FEATURES


def f_noRAG_noCoT(
    queries: Sequence[str],
    corpus: str,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    del corpus  # Keep signature consistent, corpus is not actually used
    return _run_no_rag_sync(queries, use_cot=False)


def f_noRAG_CoT(
    queries: Sequence[str],
    corpus: str,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    del corpus
    return _run_no_rag_sync(queries, use_cot=True)


def f_noARAG_noCoT(
    queries: Sequence[str],
    corpus: str,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    cfg = PipelineConfig(
        use_hyde=False,
        use_hybrid=False,
        use_rerank=False,
        use_mmr=False,
        use_cot=False,
        semantic_samples=_load_settings()["metrics"]["semantic_entropy_samples"],
        semantic_temperature=_load_settings()["metrics"]["semantic_entropy_temperature"],
    )
    return _run_sync(queries, corpus, cfg)


def f_ARAG_noCoT(
    queries: Sequence[str],
    corpus: str,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    cfg = PipelineConfig(
        use_hyde=True,
        use_hybrid=True,
        use_rerank=True,
        use_mmr=True,
        use_cot=False,
        semantic_samples=_load_settings()["metrics"]["semantic_entropy_samples"],
        semantic_temperature=_load_settings()["metrics"]["semantic_entropy_temperature"],
    )
    return _run_sync(queries, corpus, cfg)


def f_noARAG_CoT(
    queries: Sequence[str],
    corpus: str,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    cfg = PipelineConfig(
        use_hyde=False,
        use_hybrid=False,
        use_rerank=False,
        use_mmr=False,
        use_cot=True,
        semantic_samples=_load_settings()["metrics"]["semantic_entropy_samples"],
        semantic_temperature=_load_settings()["metrics"]["semantic_entropy_temperature"],
    )
    return _run_sync(queries, corpus, cfg)


def f_ARAG_CoT(
    queries: Sequence[str],
    corpus: str,
) -> Tuple[List[str], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    cfg = PipelineConfig(
        use_hyde=True,
        use_hybrid=True,
        use_rerank=True,
        use_mmr=True,
        use_cot=True,
        semantic_samples=_load_settings()["metrics"]["semantic_entropy_samples"],
        semantic_temperature=_load_settings()["metrics"]["semantic_entropy_temperature"],
    )
    return _run_sync(queries, corpus, cfg)