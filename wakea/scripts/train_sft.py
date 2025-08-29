from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser("train_sft (scaffold)")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args(argv)
    print(f"[train_sft] Loading config: {args.config}")
    # Stub: just print a summary; integrate with torch when ready
    cfg = Path(args.config).read_text(encoding="utf-8")
    print(cfg)
    print("[train_sft] Stub run complete. Replace with actual training loop.")


if __name__ == "__main__":  # pragma: no cover
    main()

