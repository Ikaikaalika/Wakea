from __future__ import annotations

from ..rag.retriever import InMemoryRetriever


def web_search(query: str, k: int = 3) -> str:
    """Local web-search using the in-memory RAG index if available.

    This avoids network calls and uses whatever corpus you indexed via build_index.
    """
    r = InMemoryRetriever.get_global()
    if not r:
        return f"[web_search] No local index loaded. Query='{query}'"
    docs = r.search(query, top_k=k)
    results = "\n".join([f"- {d.text}" for d in docs])
    return f"[web_search] Top results for '{query}':\n{results}"
