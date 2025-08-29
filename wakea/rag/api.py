from __future__ import annotations

from typing import List, Optional

from .ingest import ingest_texts
from .retriever import InMemoryRetriever


def build_index_from_file(path: str, out_path: Optional[str] = None) -> int:
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    ingest_texts(texts)
    if out_path:
        r = InMemoryRetriever.get_global()
        if r:
            r.save(out_path)
    return len(texts)


def load_index(path: str) -> int:
    r = InMemoryRetriever.load(path)
    InMemoryRetriever.set_global(r)
    return len(r.index.docs)


def query_index(q: str, k: int = 5) -> List[str]:
    r = InMemoryRetriever.get_global()
    if not r:
        return []
    docs = r.search(q, top_k=k)
    return [d.text for d in docs]
