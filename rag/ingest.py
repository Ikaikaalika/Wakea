from __future__ import annotations

from typing import List

from .retriever import Retriever


def ingest_texts(texts: List[str]) -> None:
    r = Retriever.get_global() or Retriever()
    r.ingest(texts)
    Retriever.set_global(r)
