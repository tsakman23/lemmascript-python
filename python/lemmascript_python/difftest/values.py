"""Parameter kinds and the values they describe.

A kind is ``"int"``, ``"bool"`` or ``"list[<kind>]"`` (nested freely). This
module knows how to compare values of a kind strictly (a ``bool`` is never an
``int``, a tuple is never a list), make them hashable for de-duplication, and
carry them across the boundary to Dafny's Python runtime, where a ``seq`` is a
``_dafny.Seq``.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

SCALAR_KINDS = frozenset({"int", "bool"})


def list_elem(kind: str) -> str | None:
    """``"list[int]"`` → ``"int"``; None when ``kind`` is not a list kind."""
    if kind.startswith("list[") and kind.endswith("]"):
        return kind[5:-1]
    return None


def supported(kind: str) -> bool:
    if kind in SCALAR_KINDS:
        return True
    elem = list_elem(kind)
    return elem is not None and supported(elem)


def same(kind: str, a: Any, b: Any) -> bool:
    """Structural equality that also requires both sides to have the kind's type."""
    if kind == "bool":
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if kind == "int":
        return (isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int)
                and not isinstance(b, bool) and a == b)
    elem = list_elem(kind)
    if elem is not None:
        return (type(a) is list and type(b) is list and len(a) == len(b)
                and all(same(elem, x, y) for x, y in zip(a, b)))
    return type(a) is type(b) and bool(a == b)


def freeze(v: Any) -> Any:
    """A hashable stand-in for an input value (lists become tagged tuples)."""
    if isinstance(v, list):
        return ("list", tuple(freeze(x) for x in v))
    if isinstance(v, tuple):
        return ("tuple", tuple(freeze(x) for x in v))
    return v


def _seq_class() -> Any:
    # The Dafny Python runtime (pip: dafnyruntimepython), which translated
    # programs import as `_dafny`.
    return importlib.import_module("_dafny").Seq


def to_dafny(kind: str, v: Any) -> Any:
    elem = list_elem(kind)
    if elem is None:
        return v
    return _seq_class()([to_dafny(elem, x) for x in v])


def from_dafny(kind: str, v: Any) -> Any:
    """The Python value of a model result; a value of the wrong shape is left
    as is, so that ``same`` rejects it."""
    elem = list_elem(kind)
    if elem is None or not hasattr(v, "Elements"):
        return v
    return [from_dafny(elem, x) for x in v.Elements]


def adapt(fn: Callable[..., Any], param_kinds: Sequence[str], returns: str) -> Callable[..., Any]:
    """Wrap a translated Dafny function to take and return Python values."""
    kinds = list(param_kinds)
    if not any(list_elem(k) for k in [*kinds, returns]):
        return fn

    def call(*xs: Any) -> Any:
        return from_dafny(returns, fn(*(to_dafny(k, x) for k, x in zip(kinds, xs))))

    call.__name__ = getattr(fn, "__name__", "model")
    return call
