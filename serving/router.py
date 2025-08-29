from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from tools.calculator import calculate
from tools.rag_query import rag_answer
from tools.code_exec import run_code_snippet
from tools.web_search import web_search
from utils.config import load_yaml


ToolFn = Callable[[str], str]


@dataclass
class ToolRouter:
    tools: Dict[str, ToolFn]

    @classmethod
    def default(cls) -> "ToolRouter":
        return cls(
            tools={
                "calculator": calculate,
                "rag": lambda q: rag_answer(q, k=3),
                "code": run_code_snippet,
                "web": lambda q: web_search(q, k=3),
            }
        )

    @classmethod
    def from_config(cls, tools_cfg_path: str) -> "ToolRouter":
        try:
            _ = load_yaml(tools_cfg_path)
        except Exception:
            _ = None
        # Web search tool reads its own config internally
        return cls(
            tools={
                "calculator": calculate,
                "rag": lambda q: rag_answer(q, k=3),
                "code": run_code_snippet,
                "web": lambda q: web_search(q, k=3, tools_cfg_path=tools_cfg_path),
            }
        )

    def route(self, name: str, arg: str) -> str:
        fn = self.tools.get(name)
        if not fn:
            return f"[tool:error] Unknown tool '{name}'"
        return fn(arg)
