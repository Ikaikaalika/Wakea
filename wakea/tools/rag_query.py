from __future__ import annotations

from typing import List

from ..rag.retriever import InMemoryRetriever
from ..rag.rerank import simple_rerank


def rag_answer(query: str, k: int = 3) -> str:
    retriever = InMemoryRetriever.get_global()
    if retriever is None:
        return "[RAG] No index built yet. Run build_index script."
    docs = retriever.search(query, top_k=k)
    reranked = simple_rerank(query, docs)
    context = "\n".join([d.text for d in reranked])
    return f"[RAG] Context preview:\n{context[:500]}\nAnswer (draft): {query} — see citations above."

