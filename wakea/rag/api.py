from __future__ import annotations

from typing import List

from .ingest import ingest_texts
from .retriever import InMemoryRetriever


def build_index_from_file(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    ingest_texts(texts)
    return len(texts)


def query_index(q: str, k: int = 5) -> List[str]:
    r = InMemoryRetriever.get_global()
    if not r:
        return []
    docs = r.search(q, top_k=k)
    return [d.text for d in docs]

