"""Input generation from parameter kinds.

Ints mix a fixed set of edge values (sign changes, word-size boundaries, and
values past 64 bits, where a fixed-width reading of Python's unbounded ints
would diverge) with Hypothesis' own integer strategy. Bools are just booleans.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

from hypothesis import strategies as st

SUPPORTED_KINDS = frozenset({"int", "bool"})

_POSITIVE_EDGES = (1, 2, 3, 7, 10, 2**31, 2**63, 2**64 + 1)
EDGES: tuple[int, ...] = (0, *_POSITIVE_EDGES, *(-e for e in _POSITIVE_EDGES), 10**30)

#: Largest all-bool input space enumerated exhaustively instead of sampled.
EXHAUSTIVE_LIMIT = 4096


def for_kind(kind: str) -> st.SearchStrategy[object]:
    if kind == "int":
        # integers() first: Hypothesis shrinks toward the first branch, and only
        # that branch can shrink to the least failing value (101, not 2**31).
        # The small-range branch keeps narrow preconditions (0 <= n <= 100)
        # from rejecting nearly every input.
        return st.one_of(st.integers(), st.integers(-128, 128), st.sampled_from(EDGES))
    if kind == "bool":
        return st.booleans()
    raise ValueError(f"no input strategy for parameter kind {kind!r}")


def for_params(kinds: Sequence[str]) -> st.SearchStrategy[tuple[object, ...]]:
    return st.tuples(*(for_kind(k) for k in kinds))


def _pin(kind: str, value: int) -> object:
    return value if kind == "int" else value < 0


def pinned(kinds: Sequence[str]) -> list[tuple[object, ...]]:
    """Explicit edge tuples tried before any generated input (deduplicated)."""
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
    out: list[tuple[object, ...]] = []
    for row in rows:
        t = tuple(_pin(k, v) for k, v in zip(kinds, row))
        if t not in out:
            out.append(t)
    return out


def finite_space(kinds: Sequence[str]) -> Iterator[tuple[object, ...]] | None:
    """Every input, when the space is small enough to enumerate (all-bool)."""
    if any(k != "bool" for k in kinds) or 2 ** len(kinds) > EXHAUSTIVE_LIMIT:
        return None
    return itertools.product((False, True), repeat=len(kinds))
