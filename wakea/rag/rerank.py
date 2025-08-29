from __future__ import annotations

from typing import List

from .retriever import RetrievedDoc


def simple_rerank(query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    # Already cosine-sorted; keep as-is for scaffold
    return docs

