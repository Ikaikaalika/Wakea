from __future__ import annotations

import argparse

from ..training.rfs_bridge import rfs_ppo_compat


def main(argv=None):
    ap = argparse.ArgumentParser("train_rl (scaffold)")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--use_rfs_ppo", action="store_true")
    args = ap.parse_args(argv)
    print(f"[train_rl] Loading config: {args.config}")
    if args.use_rfs_ppo:
        print("[train_rl] Trying rfs PPO...")
        out = rfs_ppo_compat(rollouts=None)
        if out is None:
            print("[train_rl] rfs PPO unavailable; please integrate or disable flag.")
    print("[train_rl] Stub run complete. Replace with PPO/GRPO training loop.")


if __name__ == "__main__":  # pragma: no cover
    main()

