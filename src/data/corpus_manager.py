import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.clients.API_BASE import API_BASEClient
from src.utils.text import chunk_text, ensure_list, sha1_text

logger = logging.getLogger(__name__)


@dataclass
class CorpusArtifacts:
    corpus_id: str
    chunks: List[str]
    embeddings: np.ndarray
    bm25: BM25Okapi
    faiss_index: faiss.Index


class CorpusManager:
    def __init__(
        self,
        client: API_BASEClient,
        embedding_dir: str,
        index_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_batch_size: int,
        use_gpu: bool = True,
    ) -> None:
        self._client = client
        self._embedding_dir = Path(embedding_dir)
        self._index_dir = Path(index_dir)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = embedding_batch_size
        self._use_gpu = use_gpu
        self._embedding_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_resources = None
        if use_gpu:
            try:
                self._gpu_resources = faiss.StandardGpuResources()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to initialize Faiss GPU, falling back to CPU mode: %s", exc)
                self._use_gpu = False

    async def prepare(self, corpus: str) -> CorpusArtifacts:
        corpus_id = sha1_text(corpus)
        embedding_path = self._embedding_dir / f"{corpus_id}.npy"
        chunks_path = self._embedding_dir / f"{corpus_id}.json"
        index_path = self._index_dir / f"{corpus_id}.faiss"

        if chunks_path.exists() and embedding_path.exists() and index_path.exists():
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            embeddings = np.load(embedding_path)
            index = faiss.read_index(str(index_path))
            if self._use_gpu:
                index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
            bm25 = self._load_bm25(chunks)
            logger.info("Cache hit, reusing vector index and embeddings: %s", corpus_id)
            return CorpusArtifacts(corpus_id=corpus_id, chunks=chunks, embeddings=embeddings, bm25=bm25, faiss_index=index)

        chunks = chunk_text(corpus, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        if not chunks:
            raise ValueError("Corpus chunk result is empty, cannot continue.")

        embeddings = await self._embed_chunks(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        if self._use_gpu:
            index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
        bm25 = self._load_bm25(chunks)

        # Persist
        np.save(embedding_path, embeddings)
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        if self._use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(index, str(index_path))

        return CorpusArtifacts(corpus_id=corpus_id, chunks=chunks, embeddings=embeddings, bm25=bm25, faiss_index=index)

    def _load_bm25(self, chunks: List[str]) -> BM25Okapi:
        tokenized = [ensure_list(chunk.lower().split()) for chunk in chunks]
        return BM25Okapi(tokenized)

    async def _embed_chunks(self, chunks: List[str]) -> np.ndarray:
        batches = [chunks[i : i + self._batch_size] for i in range(0, len(chunks), self._batch_size)]

        async def _embed_batch(batch: List[str]) -> List[List[float]]:
            response = await self._client.embedding(batch)
            data = sorted(response.get("data", []), key=lambda x: x.get("index", 0))
            return [item.get("embedding", []) for item in data]

        tasks = [_embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        flat = [vec for batch in results for vec in batch]
        return np.array(flat, dtype=np.float32)