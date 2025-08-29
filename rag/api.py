from __future__ import annotations

from typing import List, Optional

from .retriever import Retriever
from ..utils.config import load_yaml


def _get_or_create_global(cfg_path: Optional[str] = None) -> Retriever:
    r = Retriever.get_global()
    if r is not None:
        return r
    cfg = load_yaml(cfg_path) if cfg_path else {"index": {"backend": "in_memory", "dim": 256}, "embed": {"model": None}}
    backend = cfg.get("index", {}).get("backend", "in_memory")
    dim = int(cfg.get("index", {}).get("dim", 256))
    embed_model = cfg.get("embed", {}).get("model")
    r = Retriever(backend=backend, dim=dim, embed_model=embed_model)
    Retriever.set_global(r)
    return r


def build_index_from_file(path: str, out_path: Optional[str] = None, cfg_path: Optional[str] = None) -> int:
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    r = _get_or_create_global(cfg_path)
    r.ingest(texts)
    if out_path:
        r.save(out_path)
    return len(texts)


def load_index(path: str, cfg_path: Optional[str] = None) -> int:
    cfg = load_yaml(cfg_path) if cfg_path else {"index": {"backend": "in_memory", "dim": 256}, "embed": {"model": None}}
    backend = cfg.get("index", {}).get("backend", "in_memory")
    dim = int(cfg.get("index", {}).get("dim", 256))
    embed_model = cfg.get("embed", {}).get("model")
    r = Retriever.load(path, backend=backend, dim=dim, embed_model=embed_model)
    Retriever.set_global(r)
    # Return doc count
    try:
        return len(r.index.texts)
    except Exception:
        return len(r.index.docs)


def query_index(q: str, k: int = 5) -> List[str]:
    r = Retriever.get_global()
    if not r:
        return []
    docs = r.search(q, top_k=k)
    return [d.text for d in docs]
