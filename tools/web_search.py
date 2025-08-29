from __future__ import annotations

from typing import Optional

from rag.retriever import InMemoryRetriever
from utils.config import load_yaml
from tools.web_api import build_client_from_config


def web_search(query: str, k: int = 3, tools_cfg_path: str = "configs/tools.yaml") -> str:
    """Web search that prefers a real API provider if configured, else falls back to local RAG.

    Set SERPAPI_API_KEY or TAVILY_API_KEY in your environment and configure provider in configs/tools.yaml.
    """
    try:
        cfg = load_yaml(tools_cfg_path)
    except Exception:
        cfg = {}
    web_cfg = cfg if isinstance(cfg, dict) else {}
    client = build_client_from_config(web_cfg)
    if client is not None:
        try:
            topk = int(web_cfg.get("web_config", {}).get("top_k", k))
            results = client.search(query, top_k=topk)
            lines = [f"- {r.get('title')}: {r.get('link')}" for r in results]
            return "[web_search]\n" + "\n".join(lines[:k])
        except Exception as e:  # pragma: no cover
            return f"[web_search:error] {e}"
    # Fallback: local RAG
    r = InMemoryRetriever.get_global()
    if not r:
        return f"[web_search] No API configured and no local index loaded. Query='{query}'"
    docs = r.search(query, top_k=k)
    results = "\n".join([f"- {d.text}" for d in docs])
    return f"[web_search:RAG fallback] Top results for '{query}':\n{results}"
