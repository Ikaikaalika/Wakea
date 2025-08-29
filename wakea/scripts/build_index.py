from __future__ import annotations

import argparse

from ..rag.api import build_index_from_file


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="Path to text file (one doc per line)")
    args = ap.parse_args(argv)
    n = build_index_from_file(args.docs)
    print(f"[build_index] Indexed {n} docs from {args.docs}")


if __name__ == "__main__":  # pragma: no cover
    main()

