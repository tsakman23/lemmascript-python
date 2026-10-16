from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from lemmascript_python.extract import extract_source

REPO = Path(__file__).resolve().parents[2]
LSC = [str(REPO / "node_modules" / ".bin" / "tsx"), str(REPO / "tools" / "src" / "lsc.ts")]


def extract(src: str, name: str = "t.py") -> dict[str, Any]:
    """Raw IR envelope for a Python source string."""
    return extract_source(src, name)


def fn(src: str, name: str | None = None) -> dict[str, Any]:
    """The Raw IR function `name` (default: the only one) from a source string."""
    fns = extract(src)["module"]["functions"]
    if name is None:
        assert len(fns) == 1, [f["name"] for f in fns]
        return fns[0]  # type: ignore[no-any-return]
    return next(f for f in fns if f["name"] == name)


def run_lsc(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    if not Path(LSC[0]).exists():
        pytest.skip("lsc not installed (run npm ci at the repository root)")
    return subprocess.run(
        [*LSC, *args], cwd=cwd, capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "LSC_RUN_LOG": "false"},
    )


def has_dafny() -> bool:
    return shutil.which("dafny") is not None


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
