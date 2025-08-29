from __future__ import annotations

from typing import List, Tuple

from .metrics import exact_match, contains, aggregate


def judge_predictions(pairs: List[Tuple[str, str]]) -> float:
    scores = []
    for pred, ref in pairs:
        m = {
            "em": exact_match(pred, ref),
            "contains": contains(pred, ref),
        }
        scores.append(aggregate(m))
    return sum(scores) / len(scores) if scores else 0.0

