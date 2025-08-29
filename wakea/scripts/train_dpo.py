from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser("train_dpo (scaffold)")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args(argv)
    print(f"[train_dpo] Loading config: {args.config}")
    print("[train_dpo] Stub run complete. Replace with actual DPO training loop.")


if __name__ == "__main__":  # pragma: no cover
    main()

