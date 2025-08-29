from __future__ import annotations

from typing import Dict


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0


def contains(pred: str, ref_substr: str) -> float:
    return 1.0 if ref_substr.lower() in pred.lower() else 0.0


def aggregate(metrics: Dict[str, float]) -> float:
    if not metrics:
        return 0.0
    return sum(metrics.values()) / len(metrics)

