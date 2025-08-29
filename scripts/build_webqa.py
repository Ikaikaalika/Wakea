from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.web_api import build_client_from_config
from utils.config import load_yaml


def main(argv=None):
    ap = argparse.ArgumentParser("build_webqa")
    ap.add_argument("--queries", required=True, help="Path to text file with one query per line")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--tools-cfg", default="configs/tools.yaml", help="Path to tools.yaml")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.tools_cfg)
    client = build_client_from_config(cfg)
    if client is None:
        raise SystemExit("No web API configured or API key missing. Set provider and env key in configs/tools.yaml.")

    queries = [q.strip() for q in Path(args.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            results = client.search(q, top_k=args.top_k)
            item = {
                "id": f"webqa-{i}",
                "prompt": q,
                "tool": "web",
                "context": results,
            }
            f.write(json.dumps(item) + "\n")
    print(f"[build_webqa] Wrote {len(queries)} items to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

