"""Differential soundness harness: run a module's Python functions against
their compiled Dafny model on generated inputs (``python -m
lemmascript_python.difftest foo.py``)."""
from __future__ import annotations

from lemmascript_python.difftest.runner import Config, Verdict, check_file

__all__ = ["Config", "Verdict", "check_file"]
