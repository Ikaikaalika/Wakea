from __future__ import annotations

import argparse

from ..training.rfs_bridge import rfs_ppo_compat
from ..utils.logging import setup_logging
from ..utils.config import load_yaml
from ..utils.seed import seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser("train_rl (scaffold)")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--use_rfs_ppo", action="store_true")
    args = ap.parse_args(argv)
    log = setup_logging(name="wakea.train_rl")
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    log.info(f"Loaded config: {args.config}")
    if args.use_rfs_ppo:
        log.info("Trying rfs PPO...")
        out = rfs_ppo_compat(rollouts=None)
        if out is None:
            log.warning("rfs PPO unavailable; integrate a PPO backend (e.g., TRL).")
    log.info("Stub run complete. Replace with PPO/GRPO training loop.")


if __name__ == "__main__":  # pragma: no cover
    main()
