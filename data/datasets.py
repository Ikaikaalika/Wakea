from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore

from modeling.tokenizer import get_text_tokenizer


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_chat_turns(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"<|user|> {content}")
        elif role == "assistant":
            parts.append(f"<|assistant|> {content}")
        else:
            parts.append(f"<|{role}|> {content}")
    return "\n".join(parts)


@dataclass
class SFTExample:
    input_ids: List[int]
    labels: List[int]


class SFTDataset(Dataset):  # type: ignore[misc]
    def __init__(self, path: str, tokenizer_name: str | None = None, max_len: int = 1024):
        self.samples: List[SFTExample] = []
        tok = get_text_tokenizer(tokenizer_name)
        for row in _read_jsonl(path):
            msgs: List[Dict[str, str]] = row["messages"]
            text = format_chat_turns(msgs)
            ids = tok.encode(text, add_special=True)
            ids = ids[:max_len]
            # Next-token prediction
            labels = ids[1:] + [tok.eos_id]
            self.samples.append(SFTExample(ids, labels))
        self.pad_id = tok.pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        ex = self.samples[idx]
        return ex.input_ids, ex.labels


def pad_2d(seqs: List[List[int]], pad_id: int) -> "torch.Tensor":
    assert torch is not None, "Torch required for padding"
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class SFTCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Tuple[List[int], List[int]]]):
        inputs, labels = zip(*batch)
        input_ids = pad_2d(list(inputs), self.pad_id)
        label_ids = pad_2d(list(labels), self.pad_id)
        return {"input_ids": input_ids, "labels": label_ids}


@dataclass
class PrefExample:
    prompt: List[int]
    chosen: List[int]
    rejected: List[int]


class PreferenceDataset(Dataset):  # type: ignore[misc]
    def __init__(self, path: str, tokenizer_name: str | None = None, max_len: int = 1024):
        self.samples: List[PrefExample] = []
        tok = get_text_tokenizer(tokenizer_name)
        for row in _read_jsonl(path):
            p = str(row["prompt"])  # prompt text
            c = str(row["chosen"])  # assistant chosen answer
            r = str(row["rejected"])  # rejected answer
            prompt_ids = tok.encode(p)
            chosen_ids = tok.encode(c, add_special=False)
            rejected_ids = tok.encode(r, add_special=False)
            # Truncate
            prompt_ids = prompt_ids[-max_len:]
            chosen_ids = chosen_ids[: max_len - 1]
            rejected_ids = rejected_ids[: max_len - 1]
            self.samples.append(PrefExample(prompt_ids, chosen_ids, rejected_ids))
        self.pad_id = tok.pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PrefExample:
        return self.samples[idx]


class PrefCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[PrefExample]):
        prompt = pad_2d([b.prompt for b in batch], self.pad_id)
        chosen = pad_2d([b.chosen for b in batch], self.pad_id)
        rejected = pad_2d([b.rejected for b in batch], self.pad_id)
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# Tool-use dataset for auxiliary tool selection training
@dataclass
class ToolExample:
    input_ids: List[int]
    label: int


class ToolUseDataset(Dataset):  # type: ignore[misc]
    def __init__(self, path: str, tool_names: List[str], tokenizer_name: str | None = None, max_len: int = 512):
        self.samples: List[ToolExample] = []
        self.tool_to_id = {t: i for i, t in enumerate(tool_names)}
        tok = get_text_tokenizer(tokenizer_name)
        rows = _read_jsonl(path)
        for row in rows:
            prompt = str(row["prompt"])  # user text
            tool = str(row["tool"]).strip()
            if tool not in self.tool_to_id:
                # skip unknown tools to avoid label mismatch
                continue
            ids = tok.encode(prompt, add_special=True)[:max_len]
            label = self.tool_to_id[tool]
            self.samples.append(ToolExample(ids, label))
        self.pad_id = tok.pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        ex = self.samples[idx]
        return ex.input_ids, ex.label


class ToolCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Tuple[List[int], int]]):
        inputs, labels = zip(*batch)
        input_ids = pad_2d(list(inputs), self.pad_id)
        import torch as _torch

        tool_labels = _torch.tensor(labels, dtype=_torch.long)
        return {"input_ids": input_ids, "tool_labels": tool_labels}
