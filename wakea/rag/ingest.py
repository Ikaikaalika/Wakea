from __future__ import annotations

from typing import List

from .retriever import InMemoryRetriever


def ingest_texts(texts: List[str]) -> None:
    r = InMemoryRetriever.get_global() or InMemoryRetriever()
    r.ingest(texts)
    InMemoryRetriever.set_global(r)

