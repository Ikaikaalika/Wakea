from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore

from ..utils.config import load_yaml
from ..utils.logging import setup_logging
from ..utils.seed import seed_everything
from ..utils.checkpoint import save_checkpoint
from ..data.datasets import SFTDataset, SFTCollator
from ..modeling.wakea_lm import build_model_from_cfg


def load_model_cfg(path: str) -> dict:
    return load_yaml(path)


def main(argv=None):
    ap = argparse.ArgumentParser("train_sft")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args(argv)
    log = setup_logging(name="wakea.train_sft")
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    if torch is None:
        log.error("PyTorch not available. Please install torch to train.")
        return

    model_cfg = load_model_cfg(cfg["model_cfg"])
    model = build_model_from_cfg(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_cfg = cfg.get("data", {})
    dset = SFTDataset(
        path=data_cfg.get("path", "wakea/data/schemas/sft_dialogue.jsonl"),
        tokenizer_name=data_cfg.get("tokenizer"),
        max_len=int(data_cfg.get("max_len", 256)),
    )
    loader = DataLoader(
        dset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        collate_fn=SFTCollator(dset.pad_id),
    )

    opt = optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    max_steps = int(cfg.get("max_steps", 100))
    step = 0
    model.train()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        opt.zero_grad()
        _, loss = model(input_ids=input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        opt.step()
        if step % 10 == 0:
            log.info(f"step={step} loss={loss.item():.4f}")
        step += 1
        if step >= max_steps:
            break

    out_dir = Path(cfg.get("output_dir", "checkpoints/sft"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    save_checkpoint({"model": model.state_dict(), "cfg": model_cfg}, str(ckpt_path))
    log.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
