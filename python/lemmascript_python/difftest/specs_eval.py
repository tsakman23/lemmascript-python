"""Evaluate parsed spec expressions on real Python values.

A spec (``specs.parse``) is ordinary Python plus marker calls: ``A ==> B`` is
``__lsc_implies__(A, B)``, ``A <==> B`` is ``__lsc_iff__(A, B)``, and
``forall(k, P)`` / ``exists(k, P)`` are calls (a typed binder adds the keyword
``__lsc_type__``). The tree is rewritten into executable Python and evaluated
with the module's globals, the parameters and ``result`` in scope:

* implication and equivalence short-circuit exactly like ``not A or B`` and
  ``bool(A) == bool(B)``;
* a quantifier is evaluated by **enumeration** over a finite domain derived
  from the body. The conjuncts of its guard — the premises of a (nested)
  ``==>`` chain for ``forall``, the top-level ``and`` of the body for
  ``exists`` — are searched for bounds on the bound variable ``k``:
  comparisons ``a <= k``, ``a < k``, ``k < b``, ``k <= b`` (either side, or
  inside a chain such as ``0 <= k < len(xs)``, followed transitively), ``k ==
  e``, and memberships ``k in range(...)`` / ``k in xs``. Directly nested
  quantifiers of the same kind (``exists(i, exists(j, 0 <= i < j < n and
  ...))``) are looked through for the guard. At evaluation time every candidate domain
  (the interval between the tightest lower and upper bound, each container)
  is computed and the smallest is enumerated. Each candidate is a superset of
  the values satisfying the guard, and the whole body is evaluated for every
  value, so enumeration decides the quantifier exactly.

A quantifier with no such bounds raises ``Unbounded`` when the spec is
compiled; a domain larger than ``CAP`` values raises it when evaluated.
"""
from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import CodeType
from typing import Any

from lemmascript_python import specs

CAP = 10_000
_QUANTS = ("forall", "exists")
_RT = "__lsc_rt__"  # the runtime helpers, injected into the evaluation env


class Unbounded(Exception):
    """A quantifier whose domain cannot be enumerated (statically or for these values)."""


def _name(node: ast.AST, ident: str) -> bool:
    return isinstance(node, ast.Name) and node.id == ident


def _mentions(node: ast.AST, idents: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in idents for n in ast.walk(node))


def _marker(node: ast.AST, ident: str) -> bool:
    return isinstance(node, ast.Call) and _name(node.func, ident)


def _conjuncts(e: ast.expr) -> list[ast.expr]:
    if isinstance(e, ast.BoolOp) and isinstance(e.op, ast.And):
        return [c for v in e.values for c in _conjuncts(v)]
    return [e]


def _guard(k: str, quant: str, body: ast.expr) -> tuple[list[ast.expr], set[str]]:
    """The conjuncts every satisfying value must meet, and the binders of the
    same-kind quantifiers looked through to find them.

    ``forall(i, forall(j, G ==> P))`` is true at an ``i`` no ``j`` lets meet
    ``G``, and ``exists(i, exists(j, G and P))`` false there, so the inner
    guard bounds ``i`` too — through bound expressions free of ``j``.
    """
    inner: set[str] = set()
    while (_marker(body, quant) and isinstance(body, ast.Call) and len(body.args) == 2
           and isinstance(body.args[0], ast.Name) and body.args[0].id not in {k, *inner}):
        inner.add(body.args[0].id)
        body = body.args[1]
    if quant == "exists":
        return _conjuncts(body), inner
    out: list[ast.expr] = []
    while _marker(body, specs.IMPLIES):
        assert isinstance(body, ast.Call)
        out += _conjuncts(body.args[0])
        body = body.args[1]
    return out, inner


def _plus_one(e: ast.expr) -> ast.expr:
    return ast.BinOp(copy.deepcopy(e), ast.Add(), ast.Constant(1))


_FLIP: dict[type[ast.cmpop], type[ast.cmpop]] = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE,
                                                 ast.GtE: ast.LtE, ast.Eq: ast.Eq}
_UP = (ast.Lt, ast.LtE)
_DOWN = (ast.Gt, ast.GtE)


def _compose(r: type[ast.cmpop], op: type[ast.cmpop]) -> type[ast.cmpop] | None:
    """The relation ``k r a`` and ``a op b`` give between ``k`` and ``b``."""
    if r is ast.Eq or op is ast.Eq:
        return op if r is ast.Eq else r
    for same, strict in ((_UP, ast.Lt), (_DOWN, ast.Gt)):
        if r in same and op in same:
            return strict if strict in (r, op) else r
    return None


@dataclass
class _Bounds:
    lower: list[ast.expr]  # inclusive
    upper: list[ast.expr]  # exclusive
    members: list[ast.expr]  # containers the variable must be in


def _bounds(k: str, guard: list[ast.expr], inner: set[str]) -> _Bounds:
    """Bounds on ``k`` implied by the guard, over expressions free of ``k`` and
    of the inner binders. Chains are followed transitively (in ``0 <= i < j <
    len(xs)``, ``i < len(xs)``)."""
    b = _Bounds([], [], [])
    banned = {k, *inner}
    for c in guard:
        if not isinstance(c, ast.Compare):
            continue
        operands = [c.left, *c.comparators]
        ops = [type(op) for op in c.ops]
        for p, operand in enumerate(operands):
            if not _name(operand, k):
                continue
            if p + 1 < len(operands) and ops[p] is ast.In and not _mentions(operands[p + 1], banned):
                b.members.append(operands[p + 1])
            # Rightwards `k op a op b ...`, then leftwards `... b op a op k` (flipped).
            for steps in ([(ops[q - 1], operands[q]) for q in range(p + 1, len(operands))],
                          [(_FLIP.get(ops[q], ast.In), operands[q]) for q in range(p - 1, -1, -1)]):
                rel: type[ast.cmpop] | None = ast.Eq
                for op, e in steps:
                    rel = _compose(rel, op) if rel is not None and op in _FLIP else None
                    if rel is None:
                        break
                    if _mentions(e, banned):
                        continue
                    if rel in (ast.Eq, ast.GtE):
                        b.lower.append(e)
                    if rel in (ast.Eq, ast.LtE):
                        b.upper.append(_plus_one(e))
                    if rel is ast.Lt:
                        b.upper.append(e)
                    if rel is ast.Gt:
                        b.lower.append(_plus_one(e))
    return b


def _thunks(es: list[ast.expr]) -> ast.expr:
    return ast.List([ast.Lambda(_args([]), e) for e in es], ast.Load())


def _args(names: list[str]) -> ast.arguments:
    return ast.arguments(posonlyargs=[], args=[ast.arg(n) for n in names], kwonlyargs=[], kw_defaults=[],
                         defaults=[])


def _rt(attr: str, args: list[ast.expr]) -> ast.expr:
    return ast.Call(ast.Attribute(ast.Name(_RT, ast.Load()), attr, ast.Load()), args, [])


class _Rewrite(ast.NodeTransformer):
    def __init__(self, text: str, line: int) -> None:
        self.text, self.line = text, line

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if _marker(node, specs.IMPLIES):
            a, b = (self.visit(x) for x in node.args)
            return ast.BoolOp(ast.Or(), [ast.UnaryOp(ast.Not(), a), b])
        if _marker(node, specs.IFF):
            a, b = (self.visit(x) for x in node.args)
            return ast.Compare(_rt("truth", [a]), [ast.Eq()], [_rt("truth", [b])])
        if isinstance(node.func, ast.Name) and node.func.id in _QUANTS:
            return self.quantifier(node.func.id, node)
        self.generic_visit(node)
        return node

    def quantifier(self, quant: str, node: ast.Call) -> ast.expr:
        where = f"`{self.text}` (line {self.line})"
        types = [kw.value.value for kw in node.keywords
                 if kw.arg == specs.QUANT_TYPE and isinstance(kw.value, ast.Constant)]
        if (len(node.args) != 2 or not isinstance(node.args[0], ast.Name) or len(types) != len(node.keywords)
                or len(types) > 1 or not all(isinstance(t, str) for t in types)):
            raise Unbounded(f"unbounded quantifier: malformed {quant}(...) in {where}")
        k = node.args[0].id
        typ = str(types[0]) if types else "int"
        body = node.args[1]
        guard, inner = _guard(k, quant, body)
        b = _bounds(k, guard, inner)
        if typ == "nat":
            b.lower.append(ast.Constant(0))
        interval = bool(b.lower and b.upper) and typ in ("int", "nat")
        if typ == "bool":
            b.members.append(ast.Tuple([ast.Constant(False), ast.Constant(True)], ast.Load()))
        if not interval and not b.members:
            raise Unbounded(f"unbounded quantifier: no finite range for '{k}' in {quant}(...) of {where}")
        if not interval:
            b.lower, b.upper = [], []
        # Copies: the bound expressions are also part of the body rewritten below.
        lower, upper, members = ([self.visit(copy.deepcopy(e)) for e in es] for es in (b.lower, b.upper, b.members))
        domain = _rt("domain", [ast.Constant(f"'{k}' in {where}"), _thunks(lower), _thunks(upper),
                                _thunks(members)])
        pred = ast.Lambda(_args([k]), self.visit(body))
        return _rt(quant, [pred, domain])


class _Runtime:
    @staticmethod
    def truth(v: Any) -> bool:
        return bool(v)

    @staticmethod
    def forall(pred: Callable[[Any], Any], domain: Iterable[Any]) -> bool:
        return all(bool(pred(v)) for v in domain)

    @staticmethod
    def exists(pred: Callable[[Any], Any], domain: Iterable[Any]) -> bool:
        return any(bool(pred(v)) for v in domain)

    @staticmethod
    def domain(where: str, lower: list[Callable[[], Any]], upper: list[Callable[[], Any]],
               members: list[Callable[[], Any]]) -> Iterable[Any]:
        options: list[tuple[int, Iterable[Any]]] = []
        if lower and upper:
            lo, hi = max(f() for f in lower), min(f() for f in upper)
            if isinstance(lo, int) and isinstance(hi, int):
                options.append((max(0, hi - lo), range(lo, hi)))
        for f in members:
            c = f()
            # Iterating a str yields characters, not the substrings `in` tests.
            if isinstance(c, (list, tuple, range, set, frozenset, dict)):
                options.append((len(c), list(c)))
        if not options:
            raise Unbounded(f"unbounded quantifier: no enumerable domain for {where}")
        size, dom = min(options, key=lambda o: o[0])
        if size > CAP:
            raise Unbounded(f"quantifier domain of {size} values exceeds {CAP} for {where}")
        return dom


@dataclass(frozen=True)
class Spec:
    text: str
    line: int
    code: CodeType

    def evaluate(self, env: dict[str, Any]) -> Any:
        """Evaluate with ``env`` as globals (it gains the runtime helpers)."""
        env[_RT] = _Runtime
        return eval(self.code, env)  # noqa: S307 - the module's own contract


def compile_spec(text: str, line: int) -> Spec:
    """Parse and compile a spec; raises ``Unbounded`` for an unboundable quantifier."""
    tree = _Rewrite(text, line).visit(specs.parse(text, line))
    expr = ast.fix_missing_locations(ast.Expression(tree))
    return Spec(text, line, compile(expr, f"<spec line {line}>", "eval"))


def has_quantifier(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _QUANTS
               for n in ast.walk(tree))
