from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from rag.api import build_index_from_file, load_index
from .router import ToolRouter
from modeling.tokenizer import get_text_tokenizer
from modeling.wakea_lm import build_model_from_cfg
from utils.checkpoint import load_checkpoint
from utils.config import load_yaml


def parse_args(argv=None):
    ap = argparse.ArgumentParser("wakea-server (CLI scaffold)")
    ap.add_argument("--prompt", type=str, help="User prompt", default=None)
    ap.add_argument("--build-index", type=str, help="Path to txt file to index", default=None)
    ap.add_argument("--save-index", type=str, help="Path to save pickle index", default=None)
    ap.add_argument("--load-index", type=str, help="Load a pickle index", default=None)
    ap.add_argument("--ckpt", type=str, help="Path to model checkpoint .pt", default=None)
    ap.add_argument("--model-cfg", type=str, help="YAML path for model config", default=None)
    ap.add_argument("--tokenizer", type=str, help="HF tokenizer name (optional)", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tools-cfg", type=str, help="Path to tools.yaml", default="configs/tools.yaml")
    ap.add_argument("--rag-cfg", type=str, help="Path to rag.yaml", default="configs/rag.yaml")
    return ap.parse_args(argv)


def handle_prompt(prompt: str, router: ToolRouter, model=None, tok=None, temperature: float = 0.0) -> str:
    # Heuristic tool triggers in scaffold
    p = prompt.strip()
    if p.lower().startswith("rag:"):
        q = p.split(":", 1)[1].strip()
        return router.route("rag", q)
    if "use tool:calculator" in p.lower():
        # crude extract last expression
        import re

        exprs = re.findall(r"([0-9\s\+\-\*\/\(\)\.]+)", p)
        expr = exprs[-1] if exprs else p
        return router.route("calculator", expr)
    if p.lower().startswith("code:"):
        code = p.split(":", 1)[1]
        return router.route("code", code)
    # LM generation fallback
    if model is not None and tok is not None and torch is not None:
        device = next(model.parameters()).device
        ids = tok.encode(p)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=64, temperature=temperature)
        text = tok.decode(out[0].tolist(), skip_special=True)
        return text
    return f"[wakea] Echo: {p}"


def main(argv=None):
    args = parse_args(argv)
    router = ToolRouter.from_config(args.tools_cfg)
    model = None
    tok = None
    if args.load_index:
        n = load_index(args.load_index, cfg_path=args.rag_cfg)
        print(f"[wakea] Loaded {n} docs from {args.load_index}")
    if args.build_index:
        n = build_index_from_file(args.build_index, out_path=args.save_index, cfg_path=args.rag_cfg)
        print(f"[wakea] Indexed {n} lines from {args.build_index}")
    if args.ckpt and args.model_cfg:
        if torch is None:
            print("[wakea] Torch not available; cannot load model.")
        else:
            cfg = load_yaml(args.model_cfg)
            model = build_model_from_cfg(cfg)
            sd = load_checkpoint(args.ckpt)
            model.load_state_dict(sd["model"])  # type: ignore[index]
            model.eval()
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            tok = get_text_tokenizer(args.tokenizer)
            print("[wakea] Loaded model + tokenizer for generation.")
    if args.prompt:
        out = handle_prompt(args.prompt, router, model=model, tok=tok, temperature=args.temperature)
        print(out)
    else:
        print("[wakea] Interactive mode. Type 'exit' to quit.")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if line.strip().lower() in {"exit", "quit"}:
                break
            print(handle_prompt(line, router, model=model, tok=tok, temperature=args.temperature))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
