from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def save_checkpoint(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if torch is None:
        raise ImportError("Torch required for checkpoint save.")
    torch.save(obj, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("Torch required for checkpoint load.")
    return torch.load(path, map_location="cpu")

