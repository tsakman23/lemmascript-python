"""Input generation from parameter kinds.

Ints mix a fixed set of edge values (sign changes, word-size boundaries, and
values past 64 bits, where a fixed-width reading of Python's unbounded ints
would diverge) with Hypothesis' own integer strategy. Bools are just booleans.
Lists (``list[<kind>]``, nested freely) hold at most eight elements of their
element kind, and are often sorted, since many contracts require sortedness;
int lists are also drawn from small naturals (zeros, duplicates, no negatives).

When a function takes both an int and a list, its int arguments are often
replaced by the index-like values ``-len-1, -len, -1, 0, len-1, len`` of the
first list argument: the boundaries where indexing and slicing go wrong.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

from hypothesis import strategies as st

from lemmascript_python.difftest.values import SCALAR_KINDS, list_elem, supported

__all__ = ["SUPPORTED_KINDS", "supported", "for_kind", "for_params", "pinned", "finite_space",
           "index_values", "pin_values"]

#: The scalar kinds; list kinds over them are supported too (see `supported`).
SUPPORTED_KINDS = SCALAR_KINDS

_POSITIVE_EDGES = (1, 2, 3, 7, 10, 2**31, 2**63, 2**64 + 1)
EDGES: tuple[int, ...] = (0, *_POSITIVE_EDGES, *(-e for e in _POSITIVE_EDGES), 10**30)

#: Largest all-bool input space enumerated exhaustively instead of sampled.
EXHAUSTIVE_LIMIT = 4096
MAX_LIST = 8


def for_kind(kind: str) -> st.SearchStrategy[Any]:
    if kind == "int":
        # integers() first: Hypothesis shrinks toward the first branch, and only
        # that branch can shrink to the least failing value (101, not 2**31).
        # The small-range branch keeps narrow preconditions (0 <= n <= 100)
        # from rejecting nearly every input.
        return st.one_of(st.integers(), st.integers(-128, 128), st.sampled_from(EDGES))
    if kind == "bool":
        return st.booleans()
    elem = list_elem(kind)
    if elem is not None and supported(elem):
        lists = st.lists(for_kind(elem), max_size=MAX_LIST)
        if elem != "int":
            return st.one_of(lists, lists.map(sorted))
        # Small naturals: zeros, duplicates, and all-non-negative lists, which
        # contracts often require and wide ints almost never produce.
        small = st.lists(st.integers(0, 3), max_size=MAX_LIST)
        return st.one_of(lists, lists.map(sorted), small)
    raise ValueError(f"no input strategy for parameter kind {kind!r}")


def index_values(n: int) -> list[int]:
    """Index-like ints for a list of length ``n``, deduplicated, in order."""
    return list(dict.fromkeys((-n - 1, -n, -1, 0, n - 1, n)))


def _index_like(kinds: Sequence[str]) -> tuple[int, list[int]] | None:
    """(first list position, int positions) when indices make sense."""
    lists = [i for i, k in enumerate(kinds) if list_elem(k) is not None]
    ints = [i for i, k in enumerate(kinds) if k == "int"]
    return (lists[0], ints) if lists and ints else None


def for_params(kinds: Sequence[str]) -> st.SearchStrategy[tuple[Any, ...]]:
    base = st.tuples(*(for_kind(k) for k in kinds))
    shape = _index_like(kinds)
    if shape is None:
        return base
    li, ints = shape

    @st.composite
    def indexed(draw: st.DrawFn) -> tuple[Any, ...]:
        xs = list(draw(base))
        idx = index_values(len(xs[li]))
        for i in ints:
            # 0 (what Hypothesis shrinks to) keeps the plain draw.
            if draw(st.integers(0, 3)):
                xs[i] = draw(st.sampled_from(idx))
        return tuple(xs)

    return indexed()


def pin_values(kind: str) -> list[Any]:
    """Edge values of one kind, the most basic first (deduplicated)."""
    if kind == "int":
        return [0, -1, 1, -7, 3, 2**63, -(2**64 + 1)]
    if kind == "bool":
        return [False, True]
    elem = list_elem(kind)
    if elem is None:
        raise ValueError(f"no edge values for parameter kind {kind!r}")
    e = pin_values(elem)  # always at least two values
    cands: list[list[Any]] = [
        [], [e[0]], [e[1]], [e[min(2, len(e) - 1)]] * 2,  # singletons, a duplicate
        sorted(e[:5]), sorted(e[:5], reverse=True), [e[-1]],
    ]
    if elem == "int":
        cands += [[0, 2**63], [2**63]]
    out: list[Any] = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def _pin(kind: str, value: int) -> Any:
    return value if kind == "int" else value < 0


def pinned(kinds: Sequence[str]) -> list[tuple[Any, ...]]:
    """Explicit edge tuples tried after the search (deduplicated)."""
    n = len(kinds)
    rows: list[list[int]] = [
        [0] * n,
        [-1] * n,
        [1] * n,
        [1 if i % 2 == 0 else -1 for i in range(n)],
        [-1 if i % 2 == 0 else 1 for i in range(n)],
        [-7 if i % 2 == 0 else 3 for i in range(n)],
        [2**63 if i % 2 == 0 else -(2**64 + 1) for i in range(n)],
    ]
    list_pins = {k: pin_values(k) for k in kinds if list_elem(k) is not None}
    longest = max((len(p) for p in list_pins.values()), default=0)
    out: list[tuple[Any, ...]] = []

    def add(t: tuple[Any, ...]) -> None:
        if t not in out:
            out.append(t)

    for r in range(max(len(rows), longest)):
        row = rows[r % len(rows)]
        add(tuple(list_pins[k][r % len(list_pins[k])] if k in list_pins else _pin(k, row[i])
                  for i, k in enumerate(kinds)))
    shape = _index_like(kinds)
    if shape is not None:
        li, ints = shape
        for lst in list_pins[kinds[li]]:
            for v in index_values(len(lst)):
                add(tuple(lst if i == li else v if i in ints else
                          list_pins[k][0] if k in list_pins else _pin(k, 0)
                          for i, k in enumerate(kinds)))
    return out


def finite_space(kinds: Sequence[str]) -> Iterator[tuple[object, ...]] | None:
    """Every input, when the space is small enough to enumerate (all-bool)."""
    if any(k != "bool" for k in kinds) or 2 ** len(kinds) > EXHAUSTIVE_LIMIT:
        return None
    return itertools.product((False, True), repeat=len(kinds))
