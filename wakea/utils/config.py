from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("pyyaml not installed. Please install to load YAML configs.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def asdict_dc(obj: Any) -> Dict[str, Any]:
    return dataclasses.asdict(obj)

