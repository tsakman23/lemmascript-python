"""``#@`` annotation comments.

Python's ``ast`` drops comments, so annotations are collected with
``tokenize`` (which also keeps ``#@`` inside string literals from counting)
and claimed by the construct they annotate. Any annotation left unclaimed is
an error: a misplaced contract must never be silently ignored.
"""
from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass

#: Annotations at the head of a function body.
FUNCTION_KEYWORDS = frozenset({"requires", "ensures", "decreases", "verify", "pure", "type", "contract"})
#: Annotations at the head of a loop body (M2).
LOOP_KEYWORDS = frozenset({"invariant", "decreases", "done_with"})
#: Annotations at statement positions (M2).
STATEMENT_KEYWORDS = frozenset({"assert", "ghost"})
#: File-level directives, before the first statement.
FILE_KEYWORDS = frozenset({"backend", "option"})
KNOWN_KEYWORDS = FUNCTION_KEYWORDS | LOOP_KEYWORDS | STATEMENT_KEYWORDS | FILE_KEYWORDS


@dataclass
class Annotation:
    line: int
    col: int
    keyword: str
    text: str  # everything after the keyword, stripped
    claimed: bool = False


def collect(source: str) -> list[Annotation]:
    out: list[Annotation] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT or not tok.string.startswith("#@"):
            continue
        body = tok.string[2:].strip()
        keyword, _, rest = body.partition(" ")
        out.append(Annotation(tok.start[0], tok.start[1], keyword, rest.strip()))
    return out
