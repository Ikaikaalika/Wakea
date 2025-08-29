from __future__ import annotations

import argparse

from rag.api import build_index_from_file, load_index
from utils.config import load_yaml


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", help="Path to text file (one doc per line)")
    ap.add_argument("--out", help="Path to save built index (pickle)")
    ap.add_argument("--load", help="Load an existing index pickle into memory")
    ap.add_argument("--config", help="RAG config YAML (index backend, embed model)", default="configs/rag.yaml")
    args = ap.parse_args(argv)
    if args.load:
        n = load_index(args.load, cfg_path=args.config)
        print(f"[build_index] Loaded index with {n} docs from {args.load}")
        return
    if not args.docs:
        ap.error("--docs is required when not using --load")
    n = build_index_from_file(args.docs, out_path=args.out, cfg_path=args.config)
    print(f"[build_index] Indexed {n} docs from {args.docs}")


if __name__ == "__main__":  # pragma: no cover
    main()
