"""Typed lowering of Python function bodies and specs to Raw IR.

Every expression is typed from annotations and a small builtins table, and
lowered to a Raw IR construct whose *engine* semantics equal the Python
semantics. Where the engine's (JavaScript-flavoured) default differs, the
lowering is explicit: truthiness becomes a comparison, ``//`` and ``%``
become the floored ``floordiv``/``floormod`` ops, comparison chains become
conjunctions. Anything else raises ``UnsupportedError``.

The M1 subset has no side effects (no mutation, no I/O), so re-evaluating a
pure operand (``a ** 2`` → ``a * a``) is semantically safe.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from lemmascript_python import specs
from lemmascript_python.errors import UnsupportedError
from lemmascript_python.pytypes import ANY, BOOL, INT, NONE, PyType

Raw = dict[str, Any]

MAX_EXACT_NUM = 2 ** 53


def _err(msg: str, node: ast.AST) -> UnsupportedError:
    return UnsupportedError(msg, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def var(name: str) -> Raw:
    return {"kind": "var", "name": name}


def num(value: int) -> Raw:
    if abs(value) <= MAX_EXACT_NUM:
        return {"kind": "num", "value": value}
    return {"kind": "bigint", "value": str(value)}


def binop(op: str, left: Raw, right: Raw) -> Raw:
    return {"kind": "binop", "op": op, "left": left, "right": right}


def unop(op: str, expr: Raw) -> Raw:
    return {"kind": "unop", "op": op, "expr": expr}


def math_call(name: str, *args: Raw) -> Raw:
    return {"kind": "call", "fn": {"kind": "field", "obj": var("Math"), "field": name}, "args": list(args)}


@dataclass
class Signature:
    name: str
    params: list[tuple[str, PyType]]
    returns: PyType
    verified: bool = True


_ARITH = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "floordiv", ast.Mod: "floormod"}
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "===", ast.NotEq: "!=="}
_CMP_SRC = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}


@dataclass
class _Frame:
    names: dict[str, PyType] = field(default_factory=dict)


class FunctionLowerer:
    """Lowers one function: its body and its spec expressions."""

    def __init__(self, sig: Signature, functions: dict[str, Signature], verify_mode: bool,
                 binding_counts: Counter[str]) -> None:
        self.sig = sig
        self.functions = functions
        self.verify_mode = verify_mode
        self.binding_counts = binding_counts
        self.frames: list[_Frame] = [_Frame(dict(sig.params))]
        self.params = {n for n, _ in sig.params}
        # Names whose first binding was in a block that has ended.
        self.expired: dict[str, int] = {}
        self.in_spec = False
        self.allow_result = False

    # ── scopes ───────────────────────────────────────────────

    def lookup(self, name: str) -> PyType | None:
        for f in reversed(self.frames):
            if name in f.names:
                return f.names[name]
        return None

    def push(self) -> None:
        self.frames.append(_Frame())

    def pop(self, line: int) -> None:
        for name in self.frames.pop().names:
            self.expired.setdefault(name, line)

    # ── statements ───────────────────────────────────────────

    def block(self, stmts: list[ast.stmt]) -> list[Raw]:
        out: list[Raw] = []
        for s in stmts:
            out.extend(self.stmt(s))
        return out

    def nested(self, stmts: list[ast.stmt], line: int) -> list[Raw]:
        self.push()
        try:
            return self.block(stmts)
        finally:
            self.pop(line)

    def stmt(self, s: ast.stmt) -> list[Raw]:
        line = s.lineno
        if isinstance(s, ast.Pass):
            return []
        if isinstance(s, ast.Return):
            return [self.return_(s)]
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                raise _err("only assignment to a single name is supported yet", s)
            return [self.bind(s.targets[0].id, None, s.value, s)]
        if isinstance(s, ast.AnnAssign):
            if not isinstance(s.target, ast.Name):
                raise _err("only assignment to a single name is supported yet", s)
            if s.value is None:
                raise _err(f"'{s.target.id}' needs an initial value", s)
            from lemmascript_python.pytypes import from_annotation
            return [self.bind(s.target.id, from_annotation(s.annotation), s.value, s)]
        if isinstance(s, ast.AugAssign):
            if not isinstance(s.target, ast.Name):
                raise _err("only assignment to a single name is supported yet", s)
            name = s.target.id
            ty = self.lookup(name)
            if ty is None:
                raise _err(f"'{name}' is assigned with '{ast.unparse(s)}' before it is bound", s)
            value, vty = self.arith(s.op, (var(name), ty), self.expr(s.value), s)
            return [{"kind": "assign", "target": name, "value": value, "line": line}]
        if isinstance(s, ast.If):
            cond = self.cond(s.test)
            then = self.nested(s.body, s.end_lineno or line)
            else_ = self.nested(s.orelse, s.end_lineno or line)
            return [{"kind": "if", "cond": cond, "then": then, "else": else_, "line": line}]
        if isinstance(s, ast.Raise):
            return [{"kind": "throw", "line": line}]
        if isinstance(s, ast.Assert):
            return [{"kind": "if", "cond": unop("!", self.cond(s.test)), "then": [{"kind": "throw", "line": line}],
                     "else": [], "line": line}]
        if isinstance(s, ast.Expr):
            raise _err("expression statements are not supported (verified code has no side effects)", s)
        if isinstance(s, (ast.For, ast.AsyncFor)):
            raise _err("'for' loops are not supported yet", s)
        if isinstance(s, ast.While):
            raise _err("'while' loops are not supported yet", s)
        if isinstance(s, (ast.Try, ast.TryStar)):
            raise _err("'try' is not supported (exceptions are modelled only as unreachable 'raise')", s)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            raise _err("'with' is not supported", s)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            raise _err("nested functions are not supported", s)
        if isinstance(s, ast.ClassDef):
            raise _err("nested classes are not supported", s)
        if isinstance(s, (ast.Global, ast.Nonlocal)):
            raise _err(f"'{type(s).__name__.lower()}' is not supported", s)
        if isinstance(s, ast.Match):
            raise _err("'match' is not supported yet", s)
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            raise _err("imports inside functions are not supported", s)
        raise _err(f"'{type(s).__name__}' statements are not supported", s)

    def return_(self, s: ast.Return) -> Raw:
        want = self.sig.returns
        if s.value is None or (isinstance(s.value, ast.Constant) and s.value.value is None):
            if want != NONE:
                raise _err(f"'{self.sig.name}' must return '{want}'", s)
            raise _err("functions returning None are not supported yet", s)
        value, ty = self.expr(s.value)
        self.expect(ty, want, s.value, f"'{self.sig.name}' returns")
        return {"kind": "return", "value": value, "line": s.lineno}

    def bind(self, name: str, annotated: PyType | None, value_node: ast.expr, s: ast.stmt) -> Raw:
        value, ty = self.expr(value_node)
        existing = self.lookup(name)
        if existing is not None:
            if annotated is not None:
                raise _err(f"'{name}' is already bound; annotate only its first assignment", s)
            self.expect(ty, existing, value_node, f"'{name}' has type '{existing}' and is assigned")
            return {"kind": "assign", "target": name, "value": value, "line": s.lineno}
        if annotated is not None:
            self.expect(ty, annotated, value_node, f"'{name}' is annotated '{annotated}' and is assigned")
            ty = annotated
        if ty == ANY:
            raise _err(f"cannot infer a type for '{name}'; annotate it", s)
        self.frames[-1].names[name] = ty
        return {"kind": "let", "name": name, "mutable": self.binding_counts[name] > 1, "tsType": ty.raw,
                "init": value, "line": s.lineno}

    # ── expressions ──────────────────────────────────────────

    def expect(self, got: PyType, want: PyType, node: ast.AST, what: str) -> None:
        if got != want and ANY not in (got, want):
            raise _err(f"{what} a value of type '{got}', expected '{want}'", node)

    def cond(self, node: ast.expr) -> Raw:
        """A Python condition, as an explicit Raw IR bool (Python truthiness)."""
        if isinstance(node, ast.BoolOp):
            op = "&&" if isinstance(node.op, ast.And) else "||"
            parts = [self.cond(v) for v in node.values]
            out = parts[0]
            for p in parts[1:]:
                out = binop(op, out, p)
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.negated(node.operand)
        return self.truthy(self.expr(node), node)

    def truthy(self, e: tuple[Raw, PyType], node: ast.AST) -> Raw:
        raw, ty = e
        if ty in (BOOL, ANY):
            return raw
        if ty == INT:
            return binop("!==", raw, num(0))
        raise _err(f"truthiness of '{ty}' is not supported", node)

    def negated(self, node: ast.expr) -> Raw:
        """``not node`` — Python's negated truthiness."""
        if not isinstance(node, (ast.BoolOp, ast.UnaryOp)):
            raw, ty = self.expr(node)
            if ty == INT:
                return binop("===", raw, num(0))
            if ty in (BOOL, ANY):
                return unop("!", raw)
            raise _err(f"truthiness of '{ty}' is not supported", node)
        return unop("!", self.cond(node))

    def expr(self, node: ast.expr) -> tuple[Raw, PyType]:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return {"kind": "bool", "value": v}, BOOL
            if isinstance(v, int):
                return num(v), INT
            if isinstance(v, float):
                raise _err("float literals are not supported: Python floats round", node)
            if v is None:
                raise _err("'None' is not supported yet", node)
            raise _err(f"{type(v).__name__} literals are not supported yet", node)
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.BinOp):
            return self.binop(node)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return self.negated(node.operand), BOOL
            raw, ty = self.expr(node.operand)
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                self.need_int(ty, node, "unary '-'" if isinstance(node.op, ast.USub) else "unary '+'")
                return (unop("-", raw) if isinstance(node.op, ast.USub) else raw), INT
            raise _err("'~' is not supported yet", node)
        if isinstance(node, ast.BoolOp):
            parts = [self.expr(v) for v in node.values]
            word = "and" if isinstance(node.op, ast.And) else "or"
            if any(t not in (BOOL, ANY) for _, t in parts):
                raise _err(f"'{word}' on non-bool operands in a value position returns an operand, "
                           f"not a bool; compare explicitly (e.g. 'x != 0')", node)
            out = parts[0][0]
            for p, _ in parts[1:]:
                out = binop("&&" if word == "and" else "||", out, p)
            return out, BOOL
        if isinstance(node, ast.Compare):
            return self.compare(node), BOOL
        if isinstance(node, ast.IfExp):
            cond = self.cond(node.test)
            (a, ta), (b, tb) = self.expr(node.body), self.expr(node.orelse)
            if ta != tb and ANY not in (ta, tb):
                raise _err(f"conditional expression branches have different types '{ta}' and '{tb}'", node)
            return {"kind": "conditional", "cond": cond, "then": a, "else": b}, (ta if ta != ANY else tb)
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp, ast.Subscript, ast.Attribute, ast.JoinedStr, ast.Lambda)):
            raise _err(f"'{ast.unparse(node)}' is not supported yet", node)
        raise _err(f"'{ast.unparse(node)}' is not supported", node)

    def name(self, node: ast.Name) -> tuple[Raw, PyType]:
        n = node.id
        ty = self.lookup(n)
        if ty is not None:
            return var(n), ty
        if self.in_spec and n == "result":
            if not self.allow_result:
                raise _err("'result' is only valid in '#@ ensures'", node)
            return {"kind": "result"}, self.sig.returns
        if n in self.expired:
            raise _err(f"'{n}' is used outside the block that first assigns it; "
                       f"bind it (e.g. '{n}: int = 0') before that block", node)
        if n in ("True", "False", "None"):
            raise _err(f"'{n}' is not supported here", node)
        raise _err(f"unknown name '{n}'", node)

    def need_int(self, ty: PyType, node: ast.AST, what: str) -> None:
        if ty == BOOL:
            raise _err(f"arithmetic on 'bool' is not supported ({what})", node)
        if ty not in (INT, ANY):
            raise _err(f"{what} needs 'int', got '{ty}'", node)

    def arith(self, op: ast.operator, left: tuple[Raw, PyType], right: tuple[Raw, PyType],
              node: ast.AST) -> tuple[Raw, PyType]:
        if isinstance(op, ast.Div):
            raise _err("'/' produces a float; use '//' for integer division", node)
        sym = _ARITH.get(type(op))
        if sym is None:
            raise _err(f"operator '{type(op).__name__}' is not supported yet", node)
        self.need_int(left[1], node, f"'{sym}'")
        self.need_int(right[1], node, f"'{sym}'")
        return binop(sym, left[0], right[0]), INT

    def binop(self, node: ast.BinOp) -> tuple[Raw, PyType]:
        if isinstance(node.op, ast.Pow):
            base, bty = self.expr(node.left)
            self.need_int(bty, node, "'**'")
            k = node.right
            if not (isinstance(k, ast.Constant) and type(k.value) is int and 0 <= k.value <= 4):
                raise _err("'**' needs a literal exponent between 0 and 4", node)
            if k.value == 0:
                return num(1), INT
            out = base
            for _ in range(k.value - 1):
                out = binop("*", out, base)
            return out, INT
        if isinstance(node.op, ast.Div):
            raise _err("'/' produces a float; use '//' for integer division", node)
        if type(node.op) not in _ARITH:
            raise _err(f"operator in '{ast.unparse(node)}' is not supported yet", node)
        return self.arith(node.op, self.expr(node.left), self.expr(node.right), node)

    def compare(self, node: ast.Compare) -> Raw:
        operands = [node.left, *node.comparators]
        for mid in operands[1:-1]:
            if not self.is_simple(mid):
                raise _err("the middle of a chained comparison must be a name or literal "
                           "(it is evaluated once in Python)", node)
        lowered = [self.expr(o) for o in operands]
        out: Raw | None = None
        for i, op in enumerate(node.ops):
            (l, lt), (r, rt) = lowered[i], lowered[i + 1]
            if isinstance(op, (ast.Is, ast.IsNot)):
                raise _err("'is' is only supported against None", node)
            if isinstance(op, (ast.In, ast.NotIn)):
                raise _err("'in' is not supported yet", node)
            src = _CMP_SRC[type(op)]
            if lt != rt and ANY not in (lt, rt):
                raise _err(f"'{src}' between '{lt}' and '{rt}' is not supported "
                           f"(bool is an int subclass in Python; compare like types)", node)
            if src in ("<", "<=", ">", ">=") and BOOL in (lt, rt):
                raise _err(f"'{src}' on 'bool' is not supported", node)
            c = binop(_CMP[type(op)], l, r)
            out = c if out is None else binop("&&", out, c)
        assert out is not None
        return out

    @staticmethod
    def is_simple(node: ast.expr) -> bool:
        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)

    def call(self, node: ast.Call) -> tuple[Raw, PyType]:
        if node.keywords:
            raise _err("keyword arguments are not supported", node)
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise _err("'*' arguments are not supported", node)
        if not isinstance(node.func, ast.Name):
            raise _err(f"call '{ast.unparse(node)}' is not supported yet", node)
        name = node.func.id
        if self.lookup(name) is not None:
            raise _err(f"'{name}' is not a function", node)
        if name in self.functions:
            return self.user_call(self.functions[name], node)
        if name in ("forall", "exists") and self.in_spec:
            raise _err("quantifiers are not supported yet", node)
        if name in (specs.IMPLIES, specs.IFF) and self.in_spec:
            a, b = (self.cond(x) for x in node.args)
            return binop("==>" if name == specs.IMPLIES else "<==>", a, b), BOOL
        args = [self.expr(a) for a in node.args]
        if name == "abs" and len(args) == 1:
            self.need_int(args[0][1], node, "abs()")
            return math_call("abs", args[0][0]), INT
        if name in ("min", "max") and len(args) >= 2:
            for _, t in args:
                self.need_int(t, node, f"{name}()")
            out = args[0][0]
            for a, _ in args[1:]:
                out = math_call(name, out, a)
            return out, INT
        if name == "bool" and len(args) == 1:
            return self.truthy(args[0], node), BOOL
        if name == "int" and len(args) == 1 and args[0][1] == INT:
            return args[0][0], INT
        if self.in_spec:
            # Spec-only pass-through: the engine resolves the name (e.g. a ghost
            # function defined in the .dfy proof file), as upstream allows.
            return {"kind": "call", "fn": var(name), "args": [a for a, _ in args]}, ANY
        raise _err(f"unknown function '{name}'", node)

    def user_call(self, sig: Signature, node: ast.Call) -> tuple[Raw, PyType]:
        if self.verify_mode and not sig.verified and not self.in_spec:
            raise _err(f"'{self.sig.name}' calls '{sig.name}', which is not marked '#@ verify'", node)
        if len(node.args) != len(sig.params):
            raise _err(f"'{sig.name}' takes {len(sig.params)} argument(s), got {len(node.args)}", node)
        args: list[Raw] = []
        for a, (pname, pty) in zip(node.args, sig.params):
            raw, ty = self.expr(a)
            self.expect(ty, pty, a, f"argument '{pname}' of '{sig.name}' gets")
            args.append(raw)
        return {"kind": "call", "fn": var(sig.name), "args": args}, sig.returns

    # ── specs ────────────────────────────────────────────────

    def spec_value(self, text: str, line: int) -> Raw:
        """A non-boolean spec expression (e.g. a ``decreases`` measure)."""
        tree = specs.parse(text, line)
        self.in_spec = True
        try:
            return self.expr(tree)[0]
        finally:
            self.in_spec = False

    def spec(self, text: str, line: int, *, allow_result: bool = False) -> Raw:
        tree = specs.parse(text, line)
        self.in_spec, self.allow_result = True, allow_result
        try:
            return self.cond(tree)
        finally:
            self.in_spec, self.allow_result = False, False


def always_exits(stmts: list[ast.stmt]) -> bool:
    """Every path through ``stmts`` ends in ``return`` or ``raise``."""
    for s in stmts:
        if isinstance(s, (ast.Return, ast.Raise)):
            return True
        if isinstance(s, ast.Assert) and isinstance(s.test, ast.Constant) and s.test.value is False:
            return True
        if isinstance(s, ast.If) and always_exits(s.body) and always_exits(s.orelse):
            return True
    return False


def binding_counts(fn: ast.FunctionDef) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign,)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    counts[t.id] += 1
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            counts[node.target.id] += 1
    return counts
