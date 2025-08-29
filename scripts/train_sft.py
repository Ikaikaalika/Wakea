from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore

from utils.config import load_yaml
from utils.logging import setup_logging
from utils.seed import seed_everything
from utils.checkpoint import save_checkpoint
from data.datasets import SFTDataset, SFTCollator, ToolUseDataset, ToolCollator
from modeling.wakea_lm import build_model_from_cfg


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

    # Tools
    tool_cfg_path = cfg.get("tool_cfg", "configs/tools.yaml")
    tools_cfg = load_yaml(tool_cfg_path) if tool_cfg_path else {"tools": []}
    tool_names = list(tools_cfg.get("tools", []))

    model_cfg = load_model_cfg(cfg["model_cfg"]) 
    model = build_model_from_cfg(model_cfg, tool_names=tool_names if tool_names else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_cfg = cfg.get("data", {})
    dset = SFTDataset(
        path=data_cfg.get("path", "data/schemas/sft_dialogue.jsonl"),
        tokenizer_name=data_cfg.get("tokenizer"),
        max_len=int(data_cfg.get("max_len", 256)),
    )
    loader = DataLoader(
        dset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        collate_fn=SFTCollator(dset.pad_id),
    )

    # Optional tool-use dataset
    tool_data_cfg = cfg.get("tool_data", None)
    tool_loader = None
    tool_loss_weight = float(cfg.get("tool_loss_weight", 0.0))
    if tool_data_cfg and tool_names:
        tset = ToolUseDataset(
            path=tool_data_cfg.get("path", "data/schemas/tool_use.jsonl"),
            tool_names=tool_names,
            tokenizer_name=tool_data_cfg.get("tokenizer"),
            max_len=int(tool_data_cfg.get("max_len", 256)),
        )
        tool_loader = DataLoader(
            tset,
            batch_size=int(cfg.get("batch_size", 4)),
            shuffle=True,
            collate_fn=ToolCollator(tset.pad_id),
        )
        import itertools
        tool_iter = itertools.cycle(tool_loader)
    else:
        tool_iter = None

    opt = optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    max_steps = int(cfg.get("max_steps", 100))
    step = 0
    model.train()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        opt.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.get("loss") if isinstance(out, dict) else None
        if loss is None:
            # backward-compatible path
            _, loss = model(input_ids=input_ids, labels=labels)
        # Optional tool training
        if tool_iter is not None and tool_loss_weight > 0.0:
            tb = next(tool_iter)
            t_input = tb["input_ids"].to(device)
            t_labels = tb["tool_labels"].to(device)
            tout = model(input_ids=t_input, tool_labels=t_labels, tool_loss_weight=tool_loss_weight)
            if isinstance(tout, dict) and "loss" in tout:
                loss = (loss if loss is not None else 0.0) + tout["loss"]
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
