import hashlib
import math
from typing import Iterable, List

import nltk

nltk.download("punkt", quiet=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        tokens = sentence.split()
        if current_len + len(tokens) > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_tokens = math.floor(chunk_overlap)
            if overlap_tokens > 0:
                current = current[-overlap_tokens:]
                current_len = len(current)
            else:
                current = []
                current_len = 0
        current.extend(tokens)
        current_len += len(tokens)
    if current:
        chunks.append(" ".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def ensure_list(texts: Iterable[str]) -> List[str]:
    return [text for text in texts if text]
