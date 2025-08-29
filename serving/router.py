from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from tools.calculator import calculate
from tools.rag_query import rag_answer
from tools.code_exec import run_code_snippet
from tools.web_search import web_search


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
                "web": web_search,
            }
        )

    def route(self, name: str, arg: str) -> str:
        fn = self.tools.get(name)
        if not fn:
            return f"[tool:error] Unknown tool '{name}'"
        return fn(arg)
