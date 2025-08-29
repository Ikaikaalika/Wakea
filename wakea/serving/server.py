from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..rag.api import build_index_from_file
from .router import ToolRouter


def parse_args(argv=None):
    ap = argparse.ArgumentParser("wakea-server (CLI scaffold)")
    ap.add_argument("--prompt", type=str, help="User prompt", default=None)
    ap.add_argument("--build-index", type=str, help="Path to txt file to index", default=None)
    return ap.parse_args(argv)


def handle_prompt(prompt: str, router: ToolRouter) -> str:
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
    return f"[wakea] Echo: {p}"


def main(argv=None):
    args = parse_args(argv)
    router = ToolRouter.default()
    if args.build_index:
        n = build_index_from_file(args.build_index)
        print(f"[wakea] Indexed {n} lines from {args.build_index}")
    if args.prompt:
        out = handle_prompt(args.prompt, router)
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
            print(handle_prompt(line, router))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

