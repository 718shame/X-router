import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from src.clients.API_BASE import API_BASEClient
from src.data.corpus_manager import CorpusArtifacts
from src.metrics.metrics import compute_nqc
from src.utils.text import ensure_list

logger = logging.getLogger(__name__)
SENTINEL_SCORE = -np.finfo(np.float32).max


@dataclass
class RetrievalConfig:
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int
    mmr_lambda: float


@dataclass
class RetrievalResult:
    query: str
    passages: List[str]
    passage_ids: List[int]
    scores: List[float]
    dense_scores: List[float]
    sparse_scores: List[float]
    nqc: float


class HybridRetriever:
    def __init__(
        self,
        client: API_BASEClient,
        retrieval_config: RetrievalConfig,
    ) -> None:
        self._client = client
        self._config = retrieval_config

    async def query_embedding(self, queries: Sequence[str], batch_size: int = 8) -> np.ndarray:
        query_list = list(queries)
        batched = [query_list[i : i + batch_size] for i in range(0, len(query_list), batch_size)]
        responses = await asyncio.gather(*(self._client.embedding(batch) for batch in batched))
        vectors: List[np.ndarray] = []
        for response in responses:
            data = response.get("data", [])
            if not data:
                raise RuntimeError("API_BASE embedding returned empty")
            sorted_data = sorted(data, key=lambda x: x.get("index", 0))
            for item in sorted_data:
                vectors.append(np.array(item.get("embedding", []), dtype=np.float32))
        return np.stack(vectors)

    async def dense_search(
        self,
        query_vector: np.ndarray,
        artifacts: CorpusArtifacts,
    ) -> Tuple[List[int], List[float]]:
        query_vec = query_vector.astype(np.float32)
        distances, indices = artifacts.faiss_index.search(np.expand_dims(query_vec, axis=0), self._config.dense_top_k)
        distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=SENTINEL_SCORE)
        return indices[0].tolist(), distances[0].tolist()

    def sparse_search(
        self,
        query: str,
        artifacts: CorpusArtifacts,
    ) -> Tuple[List[int], List[float]]:
        tokenized = ensure_list(query.lower().split())
        scores = artifacts.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][: self._config.sparse_top_k]
        return top_indices.tolist(), scores[top_indices].tolist()

    def _normalize(self, scores: Sequence[float]) -> List[float]:
        arr = np.array(scores, dtype=np.float32)
        if arr.size == 0:
            return []
        min_score = float(arr.min())
        max_score = float(arr.max())
        if abs(max_score - min_score) < 1e-6:
            return [1.0 for _ in arr]
        normalized = (arr - min_score) / (max_score - min_score)
        return normalized.tolist()

    def _merge_scores(
        self,
        dense: Tuple[List[int], List[float]],
        sparse: Optional[Tuple[List[int], List[float]]],
        alpha: float = 0.6,
    ) -> Dict[int, float]:
        dense_idx, dense_scores = dense
        dense_norm = self._normalize(dense_scores)
        merged: Dict[int, float] = {}
        for idx, score in zip(dense_idx, dense_norm):
            merged[idx] = merged.get(idx, 0.0) + alpha * score
        if sparse is not None:
            sparse_idx, sparse_scores = sparse
            sparse_norm = self._normalize(sparse_scores)
            for idx, score in zip(sparse_idx, sparse_norm):
                merged[idx] = merged.get(idx, 0.0) + (1 - alpha) * score
        return merged

    async def rerank(
        self,
        query: str,
        candidates: List[str],
    ) -> List[int]:
        if not candidates:
            return []
        response = await self._client.rerank(query, candidates, top_n=len(candidates))
        results = response.get("results") or response.get("data") or []
        if not results:
            return list(range(len(candidates)))
        ordered = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return [item.get("index", idx) for idx, item in enumerate(ordered)]

    def _mmr(
        self,
        query_vec: np.ndarray,
        candidate_indices: List[int],
        candidate_vectors: np.ndarray,
        lambda_param: float,
        top_k: int,
    ) -> List[int]:
        selected: List[int] = []
        if not candidate_indices:
            return selected
        candidate_vecs = candidate_vectors[candidate_indices]
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        candidate_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-9)
        similarities = candidate_norms @ query_norm
        remaining = set(range(len(candidate_indices)))
        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = -np.inf
            for idx in remaining:
                candidate_sim = similarities[idx]
                redundancy = 0.0
                if selected:
                    chosen_vecs = candidate_norms[selected]
                    redundancy = float(np.max(chosen_vecs @ candidate_norms[idx]))
                score = lambda_param * candidate_sim - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx
            assert best_idx is not None
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [candidate_indices[i] for i in selected]

    async def retrieve(
        self,
        query: str,
        query_vec: np.ndarray,
        artifacts: CorpusArtifacts,
        use_rerank: bool,
        use_sparse: bool,
        apply_mmr: bool,
    ) -> RetrievalResult:
        dense_indices, dense_scores = await self.dense_search(query_vec, artifacts)
        filtered_dense = [
            (idx, score)
            for idx, score in zip(dense_indices, dense_scores)
            if idx >= 0 and score > SENTINEL_SCORE * 0.99
        ]
        if not filtered_dense:
            logger.warning("Dense search returned no valid candidates | query='%s...'", query[:48])
            dense_indices, dense_scores = [], []
        else:
            dense_indices, dense_scores = zip(*filtered_dense)
            dense_indices = list(dense_indices)
            dense_scores = list(dense_scores)

        if dense_scores and any(score <= SENTINEL_SCORE * 0.9 for score in dense_scores):
            logger.warning(
                "Dense search returned sentinel values | query='%s...' | top_scores=%s | indices=%s",
                query[:48],
                dense_scores[:5],
                dense_indices[:5],
            )
        if np.isnan(dense_scores).any():
            logger.warning(
                "Dense search scores contain NaN | query_head=%s | indices=%s",
                query[:40],
                dense_indices[:5],
            )
        sparse_indices: List[int] = []
        sparse_scores: List[float] = []
        sparse_payload: Optional[Tuple[List[int], List[float]]] = None
        if use_sparse:
            sparse_indices, sparse_scores = self.sparse_search(query, artifacts)
            sparse_payload = (sparse_indices, sparse_scores)
        merged_scores = self._merge_scores((dense_indices, dense_scores), sparse_payload)
        sorted_candidates = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in sorted_candidates[: max(self._config.final_top_k * 2, 20)]]
        candidate_passages = [artifacts.chunks[idx] for idx in candidate_indices]

        order = list(range(len(candidate_indices)))
        if use_rerank:
            rerank_order = await self.rerank(query, candidate_passages)
            if rerank_order:
                order = rerank_order
        ordered_indices = [candidate_indices[i] for i in order[: self._config.final_top_k]]
        if apply_mmr:
            selected_indices = self._mmr(
                query_vec,
                ordered_indices,
                artifacts.embeddings,
                lambda_param=self._config.mmr_lambda,
                top_k=self._config.final_top_k,
            )
        else:
            selected_indices = ordered_indices[: self._config.final_top_k]
        passages = [artifacts.chunks[idx] for idx in selected_indices]
        scores = [merged_scores[idx] for idx in selected_indices]
        nqc = compute_nqc(dense_scores[:10])
        sparse_subset = [merged_scores[idx] for idx in selected_indices]
        return RetrievalResult(
            query=query,
            passages=passages,
            passage_ids=selected_indices,
            scores=scores,
            dense_scores=dense_scores,
            sparse_scores=sparse_subset,
            nqc=nqc,
        )