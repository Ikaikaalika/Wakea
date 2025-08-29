from __future__ import annotations

import io
import sys
import contextlib
import textwrap


SAFE_GLOBALS = {"__builtins__": {"print": print, "range": range, "len": len}}  # very restricted


def run_code_snippet(code: str, timeout_s: float = 2.0) -> str:
    """Run a tiny Python snippet with severe restrictions. For research only.

    SECURITY NOTE: This is not a real sandbox. Do not expose in production.
    """
    code = textwrap.dedent(code)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<snippet>", "exec"), SAFE_GLOBALS, {})
    except Exception as e:  # pragma: no cover - side effectful
        return f"[code_exec:error] {e}"
    return buf.getvalue().strip()

