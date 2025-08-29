from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from utils.logging import setup_logging
from utils.config import load_yaml
from utils.seed import seed_everything
from utils.checkpoint import save_checkpoint
from data.datasets import PreferenceDataset, PrefCollator
from modeling.wakea_lm import build_model_from_cfg


def main(argv=None):
    ap = argparse.ArgumentParser("train_dpo")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args(argv)
    log = setup_logging(name="wakea.train_dpo")
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    if torch is None:
        log.error("PyTorch not available. Please install torch to train.")
        return

    model_cfg = load_yaml(cfg["model_cfg"])
    model = build_model_from_cfg(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_cfg = cfg.get("data", {})
    dset = PreferenceDataset(
        path=data_cfg.get("path", "data/schemas/pref_pairs.jsonl"),
        tokenizer_name=data_cfg.get("tokenizer"),
        max_len=int(data_cfg.get("max_len", 256)),
    )
    loader = DataLoader(
        dset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        collate_fn=PrefCollator(dset.pad_id),
    )
    opt = optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-5)))
    beta = float(cfg.get("beta", 0.1))
    max_steps = int(cfg.get("max_steps", 100))
    step = 0
    model.train()

    for batch in loader:
        prompt = batch["prompt"].to(device)
        chosen = batch["chosen"].to(device)
        rejected = batch["rejected"].to(device)

        # Concatenate prompt + response for logprob calc
        def seq_logprob(seq):
            logits, _ = model(seq[:, :-1], labels=None)
            logp = F.log_softmax(logits, dim=-1)
            gather = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
            # Mask padding
            mask = (seq[:, 1:] != dset.pad_id).float()
            return (gather * mask).sum(dim=-1)

        prompt_chosen = torch.cat([prompt, chosen], dim=1)
        prompt_rejected = torch.cat([prompt, rejected], dim=1)
        logp_c = seq_logprob(prompt_chosen)
        logp_r = seq_logprob(prompt_rejected)
        # DPO objective (without reference): maximize sigmoid(beta*(logp_c - logp_r))
        loss = -torch.log(torch.sigmoid(beta * (logp_c - logp_r))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            log.info(f"step={step} loss={loss.item():.4f}")
        step += 1
        if step >= max_steps:
            break

    out_dir = Path(cfg.get("output_dir", "checkpoints/dpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    save_checkpoint({"model": model.state_dict(), "cfg": model_cfg}, str(ckpt_path))
    log.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
