"""Spec expressions: Python expression syntax plus ``==>``, ``<==>`` and typed
quantifier binders.

A spec is rewritten into plain Python source and parsed with ``ast``, so the
body lowering (``lower.py``) handles it with Python semantics — ``%`` in a
contract means exactly what ``%`` means in the code. The extensions become
marker calls the lowering recognises:

* ``A ==> B``  → ``__lsc_implies__(A, B)`` (right-associative, looser than ``or``)
* ``A <==> B`` → ``__lsc_iff__(A, B)``     (loosest, right-associative)
* ``forall(k: T, P)`` → ``forall(k, P, __lsc_type__='T')`` (likewise ``exists``)
"""
from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass

from lemmascript_python.errors import UnsupportedError

IMPLIES = "__lsc_implies__"
IFF = "__lsc_iff__"
QUANT_TYPE = "__lsc_type__"

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = set(_OPEN.values())


@dataclass
class _Tok:
    type: int
    string: str
    start: int
    end: int


def _tokens(text: str, line: int) -> list[_Tok]:
    out: list[_Tok] = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
                continue
            if t.type == tokenize.COMMENT:
                if t.string.startswith("#@"):
                    raise UnsupportedError("one annotation per line: a second '#@' would be read as a comment",
                                           line)
                continue  # a plain trailing comment
            if t.type in (tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END):
                raise UnsupportedError("f-strings are not supported in spec expressions", line)
            if t.start[0] != 1:
                raise UnsupportedError("a spec expression must fit on one line", line)
            out.append(_Tok(t.type, t.string, t.start[1], t.end[1]))
    except (tokenize.TokenError, IndentationError) as e:
        raise UnsupportedError(f"invalid spec expression '{text}': {e}", line) from None
    return out


def _op_at(toks: list[_Tok], i: int) -> str | None:
    """``==>`` or ``<==>`` starting at token ``i`` (adjacent tokens only)."""
    def adj(*want: str) -> bool:
        if i + len(want) > len(toks):
            return False
        for k, w in enumerate(want):
            t = toks[i + k]
            if t.type != tokenize.OP or t.string != w:
                return False
            if k > 0 and toks[i + k - 1].end != t.start:
                return False
        return True
    if adj("<=", "=", ">"):
        return "<==>"
    if adj("==", ">"):
        return "==>"
    return None


def _width(op: str) -> int:
    return 3 if op == "<==>" else 2


def _groups(toks: list[_Tok], line: int) -> list[tuple[int, int]]:
    """Index pairs (open, close) of depth-0 bracket groups."""
    out: list[tuple[int, int]] = []
    depth, start = 0, -1
    for i, t in enumerate(toks):
        if t.type == tokenize.OP and t.string in _OPEN:
            if depth == 0:
                start = i
            depth += 1
        elif t.type == tokenize.OP and t.string in _CLOSE:
            depth -= 1
            if depth < 0:
                raise UnsupportedError(f"unbalanced '{t.string}' in spec expression", line)
            if depth == 0:
                out.append((start, i))
    if depth != 0:
        raise UnsupportedError("unbalanced brackets in spec expression", line)
    return out


def _depth0(toks: list[_Tok]) -> list[bool]:
    flags, depth = [], 0
    for t in toks:
        if t.type == tokenize.OP and t.string in _CLOSE:
            depth -= 1
        flags.append(depth == 0)
        if t.type == tokenize.OP and t.string in _OPEN:
            depth += 1
    return flags


def _split(toks: list[_Tok], sep: str) -> list[list[_Tok]]:
    parts: list[list[_Tok]] = [[]]
    for t, top in zip(toks, _depth0(toks)):
        if top and t.type == tokenize.OP and t.string == sep:
            parts.append([])
        else:
            parts[-1].append(t)
    return parts


def _region(toks: list[_Tok], line: int) -> str:
    """Comma-separated items (call args, tuple/list elements, the whole spec)."""
    items = _split(toks, ",")
    return ", ".join(_item(it, line) for it in items)


def _item(toks: list[_Tok], line: int) -> str:
    top = _depth0(toks)
    for i, t in enumerate(toks):
        if top[i] and t.type == tokenize.NAME and t.string in ("for", "async"):
            # Generator/comprehension element: the extension operators bind
            # within the element, never across the `for` clause.
            return _expr(toks[:i], line) + " " + _plain(toks[i:], line)
    return _expr(toks, line)


def _expr(toks: list[_Tok], line: int) -> str:
    if not toks:
        return ""
    top = _depth0(toks)
    for want in ("<==>", "==>"):
        for i in range(len(toks)):
            if top[i] and _op_at(toks, i) == want:
                left, right = toks[:i], toks[i + _width(want):]
                if not left or not right:
                    raise UnsupportedError(f"'{want}' needs an expression on both sides", line)
                marker = IFF if want == "<==>" else IMPLIES
                return f"{marker}(({_expr(left, line)}), ({_expr(right, line)}))"
    return _plain(toks, line)


def _plain(toks: list[_Tok], line: int) -> str:
    parts: list[str] = []
    groups = {o: c for o, c in _groups(toks, line)}
    i = 0
    while i < len(toks):
        t = toks[i]
        if i in groups:
            close = groups[i]
            inner = toks[i + 1:close]
            prev = toks[i - 1] if i > 0 else None
            if (t.string == "(" and prev is not None and prev.type == tokenize.NAME
                    and prev.string in ("forall", "exists")):
                parts.append("(" + _quantifier(inner, line) + ")")
            else:
                parts.append(t.string + _region(inner, line) + toks[close].string)
            i = close + 1
            continue
        if _op_at(toks, i):
            # Only reachable when an extension operator is nested where it
            # cannot be delimited (e.g. inside a subscript slice).
            raise UnsupportedError("parenthesize the operand of '==>' / '<==>'", line)
        parts.append(t.string)
        i += 1
    return " ".join(parts)


def _quantifier(inner: list[_Tok], line: int) -> str:
    """``k: T, P`` or ``k, P`` → ``k, P'[, __lsc_type__='T']``."""
    items = _split(inner, ",")
    if len(items) != 2 or not items[0] or items[0][0].type != tokenize.NAME:
        raise UnsupportedError("a quantifier is written forall(k, P) or forall(k: T, P)", line)
    binder = items[0]
    body = _item(items[1], line)
    if len(binder) == 1:
        return f"{binder[0].string}, {body}"
    if binder[1].string != ":" or len(binder) < 3:
        raise UnsupportedError("a quantifier is written forall(k, P) or forall(k: T, P)", line)
    type_text = " ".join(t.string for t in binder[2:])
    return f"{binder[0].string}, {body}, {QUANT_TYPE}={type_text!r}"


def parse(text: str, line: int) -> ast.expr:
    """Parse a spec expression into a Python AST with marker calls."""
    if not text.strip():
        raise UnsupportedError("empty spec expression", line)
    if "__lsc_" in text:
        raise UnsupportedError("names starting with '__lsc_' are reserved", line)
    source = _region(_tokens(text, line), line)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        raise UnsupportedError(f"invalid spec expression '{text}'", line) from None
    # Report locations at the annotation's line.
    for node in ast.walk(tree):
        if isinstance(node, (ast.expr, ast.stmt)):
            node.lineno = line
            node.end_lineno = line
    return tree.body


def _as_bool(e: ast.expr) -> ast.expr:
    return ast.Call(ast.Name("bool", ast.Load()), [e], [])


class _ToPython(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == IMPLIES:
            a, b = node.args
            return ast.BoolOp(ast.Or(), [ast.UnaryOp(ast.Not(), a), _as_bool(b)])
        if isinstance(node.func, ast.Name) and node.func.id == IFF:
            a, b = node.args
            return ast.Compare(_as_bool(a), [ast.Eq()], [_as_bool(b)])
        return node


def to_python(tree: ast.expr) -> str:
    """Executable Python for a parsed spec (``==>`` → ``not A or B``).

    Used by the differential harness to evaluate contracts on real Python
    values. Quantifiers are left as calls; the harness rejects unbounded ones.
    """
    import copy
    out: ast.AST = _ToPython().visit(copy.deepcopy(tree))
    return ast.unparse(ast.fix_missing_locations(out))
