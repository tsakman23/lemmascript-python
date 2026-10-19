"""Typed lowering of Python function bodies and specs to Raw IR.

Every expression is typed from annotations and a small builtins table, and
lowered to a Raw IR construct whose *engine* semantics equal the Python
semantics. Where the engine's (JavaScript-flavoured) default differs, the
lowering is explicit: truthiness becomes a comparison, ``//`` and ``%``
become the floored ``floordiv``/``floormod`` ops, comparison chains become
conjunctions, slices clamp (the file always enables ``safe-slice``).
Anything else raises ``UnsupportedError``.

Lists are values in the model but shared references in Python. The
*fresh-local rule* (see ``_check_aliasing``) makes the difference
unobservable: only locals bound to fresh lists may be changed in place, and
such locals are never shared. Without mutation of shared values the
subset has no side effects, so re-evaluating a pure operand (``a ** 2`` →
``a * a``) is semantically safe.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from lemmascript_python import specs
from lemmascript_python.annotations import LOOP_KEYWORDS, STATEMENT_KEYWORDS, Annotation
from lemmascript_python.errors import UnsupportedError
from lemmascript_python.pytypes import ANY, BOOL, INT, NONE, PyType, from_annotation, from_type_text, list_of

Raw = dict[str, Any]
Typed = tuple[Raw, PyType]

MAX_EXACT_NUM = 2 ** 53

#: List methods that change the list in place, and what the model does instead.
MUTATING_METHODS = {"append": "push", "extend": None}
UNSUPPORTED_LIST_METHODS = {"pop", "insert", "remove", "sort", "reverse", "clear", "index", "count", "copy"}


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


def field_(obj: Raw, name: str) -> Raw:
    return {"kind": "field", "obj": obj, "field": name}


def method(obj: Raw, name: str, *args: Raw) -> Raw:
    return {"kind": "call", "fn": field_(obj, name), "args": list(args)}


def math_call(name: str, *args: Raw) -> Raw:
    return {"kind": "call", "fn": field_(var("Math"), name), "args": list(args)}


def length(obj: Raw) -> Raw:
    return field_(obj, "length")


def conj(parts: list[Raw]) -> Raw:
    out = parts[0]
    for p in parts[1:]:
        out = binop("&&", out, p)
    return out


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
    #: Names visible to Python code in this block.
    names: dict[str, PyType] = field(default_factory=dict)
    #: Names with a Raw IR `let` in this block (a loop variable stays declared after its loop).
    declared: dict[str, PyType] = field(default_factory=dict)
    ghosts: set[str] = field(default_factory=set)


@dataclass
class _Loop:
    #: For a `for` desugared to a counted `while`: the counter increment that
    #: must run before each `continue`, as Python's iterator would.
    increment: Raw | None


@dataclass
class _Binding:
    fresh: bool
    node: ast.AST


@dataclass
class _Use:
    shared: bool
    node: ast.AST


class FunctionLowerer:
    """Lowers one function: its body and its spec expressions."""

    def __init__(self, sig: Signature, functions: dict[str, Signature], verify_mode: bool,
                 binding_counts: Counter[str], annotations: list[Annotation] | None = None,
                 taken_names: set[str] | None = None) -> None:
        self.sig = sig
        self.functions = functions
        self.verify_mode = verify_mode
        self.binding_counts = binding_counts
        self.annotations = annotations or []
        self.taken = set(taken_names or ()) | {n for n, _ in sig.params}
        self.frames: list[_Frame] = [_Frame(dict(sig.params), dict(sig.params))]
        self.params = {n for n, _ in sig.params}
        # Names whose first binding was in a block that has ended.
        self.expired: dict[str, int] = {}
        # Loop variables after their loop has ended.
        self.loop_expired: dict[str, int] = {}
        self.loop_vars: set[str] = set()
        self.loops: list[_Loop] = []
        self.iterating: list[str] = []
        self.in_spec = False
        self.allow_result = False
        # Aliasing bookkeeping (lists only), checked once the body is lowered.
        self.bindings: dict[str, list[_Binding]] = {}
        self.uses: dict[str, list[_Use]] = {}
        self.mutations: dict[str, list[ast.AST]] = {}
        for p, t in sig.params:
            if t.is_list:
                self.bindings[p] = [_Binding(False, ast.Name(p, lineno=0, col_offset=0))]

    # ── names and scopes ─────────────────────────────────────

    def fresh(self, base: str) -> str:
        name, k = base, 1
        while name in self.taken:
            k += 1
            name = f"{base}{k}"
        self.taken.add(name)
        return name

    def lookup(self, name: str) -> PyType | None:
        for f in reversed(self.frames):
            if name in f.names:
                return f.names[name]
        return None

    def declared(self, name: str) -> PyType | None:
        for f in reversed(self.frames):
            if name in f.declared:
                return f.declared[name]
        return None

    def is_ghost(self, name: str) -> bool:
        for f in reversed(self.frames):
            if name in f.names:
                return name in f.ghosts
        return False

    def push(self) -> None:
        self.frames.append(_Frame())

    def pop(self, line: int) -> None:
        for name in self.frames.pop().names:
            self.expired.setdefault(name, line)

    def hide_loop_var(self, name: str, line: int) -> None:
        for f in reversed(self.frames):
            if name in f.names:
                del f.names[name]
                break
        self.loop_vars.discard(name)
        self.loop_expired[name] = line

    # ── annotations inside the body ──────────────────────────

    def claim(self, keywords: frozenset[str], lo: int, hi: int, col: int | None = None,
              min_col: int | None = None) -> list[Annotation]:
        out = []
        for a in self.annotations:
            if a.claimed or a.keyword not in keywords or not (lo < a.line < hi):
                continue
            if col is not None and a.col != col:
                continue
            if min_col is not None and a.col <= min_col:
                continue
            a.claimed = True
            out.append(a)
        return out

    def statement_annotation(self, a: Annotation) -> Raw:
        if a.keyword == "assert":
            return {"kind": "assert", "expr": self.spec(a.text, a.line), "line": a.line}
        # ghost: `#@ ghost x: T = E` declares, `#@ ghost x = E` assigns.
        text = a.text
        target, eq, init = text.partition("=")
        if not eq or init.startswith("="):
            raise UnsupportedError("'#@ ghost' is written '#@ ghost x: T = E' or '#@ ghost x = E'", a.line, a.col)
        name, colon, ty_text = target.partition(":")
        name = name.strip()
        if not name.isidentifier():
            raise UnsupportedError(f"invalid ghost variable name '{name}'", a.line, a.col)
        value, vty = self.spec_typed(init.strip(), a.line)
        if colon:
            ty = from_type_text(ty_text.strip(), a.line)
            self.check_type(vty, ty, a, f"ghost '{name}' is declared '{ty}' and is assigned")
            self.frames[-1].names[name] = ty
            self.frames[-1].declared[name] = ty
            self.frames[-1].ghosts.add(name)
            self.taken.add(name)
            return {"kind": "ghostLet", "name": name, "tsType": ty.raw, "init": value, "line": a.line}
        existing = self.lookup(name)
        if existing is None or not self.is_ghost(name):
            raise UnsupportedError(f"'#@ ghost {name} = ...' assigns an undeclared ghost variable; "
                                   f"declare it with '#@ ghost {name}: T = ...'", a.line, a.col)
        self.check_type(vty, existing, a, f"ghost '{name}' has type '{existing}' and is assigned")
        return {"kind": "ghostAssign", "target": name, "value": value, "line": a.line}

    # ── statements ───────────────────────────────────────────

    def body(self, stmts: list[ast.stmt], start: int, boundary: int) -> list[Raw]:
        """Lower a block, interleaving the `#@ assert` / `#@ ghost` lines placed between its statements.

        ``start`` is the line the block's header is on; ``boundary`` the first line after the block.
        """
        out: list[Raw] = []
        if not stmts:
            return out  # an absent `else`: it owns no annotations
        indent = stmts[0].col_offset
        prev_end = start
        for i, s in enumerate(stmts):
            for a in self.claim(STATEMENT_KEYWORDS, prev_end, s.lineno, col=indent):
                out.append(self.statement_annotation(a))
            nxt = stmts[i + 1].lineno if i + 1 < len(stmts) else boundary
            out.extend(self.stmt(s, nxt))
            prev_end = s.end_lineno or s.lineno
        for a in self.claim(STATEMENT_KEYWORDS, prev_end, boundary, col=indent):
            out.append(self.statement_annotation(a))
        return out

    def nested(self, stmts: list[ast.stmt], start: int, boundary: int, line: int) -> list[Raw]:
        self.push()
        try:
            return self.body(stmts, start, boundary)
        finally:
            self.pop(line)

    def stmt(self, s: ast.stmt, boundary: int) -> list[Raw]:
        line = s.lineno
        if isinstance(s, ast.Pass):
            return []
        if isinstance(s, ast.Return):
            return [self.return_(s)]
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1:
                raise _err("chained assignment is not supported", s)
            t = s.targets[0]
            if isinstance(t, ast.Subscript):
                return [self.set_item(t, s.value, s)]
            if not isinstance(t, ast.Name):
                raise _err("only assignment to a single name is supported yet", s)
            return [self.bind(t.id, None, s.value, s)]
        if isinstance(s, ast.AnnAssign):
            if not isinstance(s.target, ast.Name):
                raise _err("only assignment to a single name is supported yet", s)
            if s.value is None:
                raise _err(f"'{s.target.id}' needs an initial value", s)
            return [self.bind(s.target.id, from_annotation(s.annotation), s.value, s)]
        if isinstance(s, ast.AugAssign):
            return [self.aug_assign(s)]
        if isinstance(s, ast.If):
            cond = self.cond(s.test)
            then_end = s.orelse[0].lineno if s.orelse else boundary
            then = self.nested(s.body, line, then_end, s.end_lineno or line)
            else_ = self.nested(s.orelse, then_end - 1 if s.orelse else line, boundary, s.end_lineno or line)
            return [{"kind": "if", "cond": cond, "then": then, "else": else_, "line": line}]
        if isinstance(s, ast.Raise):
            return [{"kind": "throw", "line": line}]
        if isinstance(s, ast.Assert):
            return [{"kind": "if", "cond": unop("!", self.cond(s.test)), "then": [{"kind": "throw", "line": line}],
                     "else": [], "line": line}]
        if isinstance(s, ast.Expr):
            return [self.expr_stmt(s)]
        if isinstance(s, ast.While):
            return [self.while_(s, boundary)]
        if isinstance(s, ast.For):
            return self.for_(s, boundary)
        if isinstance(s, ast.Break):
            return [{"kind": "break", "line": line}]
        if isinstance(s, ast.Continue):
            loop = self.loops[-1]
            inc = [dict(loop.increment, line=loop.increment["line"])] if loop.increment else []
            return [*inc, {"kind": "continue", "line": line}]
        if isinstance(s, ast.AsyncFor):
            raise _err("'async for' is not supported", s)
        if isinstance(s, (ast.Try, ast.TryStar)):
            raise _err("'try' is not supported (exceptions are modelled only as unreachable 'raise')", s)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            raise _err("'with' is not supported", s)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise _err("nested functions are not supported", s)
        if isinstance(s, ast.ClassDef):
            raise _err("nested classes are not supported", s)
        if isinstance(s, (ast.Global, ast.Nonlocal)):
            raise _err(f"'{type(s).__name__.lower()}' is not supported", s)
        if isinstance(s, ast.Match):
            raise _err("'match' is not supported yet", s)
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            raise _err("imports inside functions are not supported", s)
        if isinstance(s, ast.Delete):
            raise _err("'del' is not supported yet", s)
        raise _err(f"'{type(s).__name__}' statements are not supported", s)

    def return_(self, s: ast.Return) -> Raw:
        want = self.sig.returns
        if s.value is None or (isinstance(s.value, ast.Constant) and s.value.value is None):
            if want != NONE:
                raise _err(f"'{self.sig.name}' must return '{want}'", s)
            raise _err("functions returning None are not supported yet", s)
        value, ty = self.expr(s.value, want, share=False)
        self.expect(ty, want, s.value, f"'{self.sig.name}' returns")
        return {"kind": "return", "value": value, "line": s.lineno}

    def bind(self, name: str, annotated: PyType | None, value_node: ast.expr, s: ast.stmt) -> Raw:
        if name in self.loop_vars:
            raise _err(f"cannot assign the loop variable '{name}' inside its loop "
                       "(Python would not change the iteration)", s)
        if self.is_ghost(name):
            raise _err(f"ghost variable '{name}' cannot be assigned in code; use '#@ ghost {name} = ...'", s)
        existing = self.lookup(name)
        value, ty = self.expr(value_node, annotated or existing)
        if ty.is_list:
            self.bindings.setdefault(name, []).append(_Binding(_is_fresh(value_node), s))
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
        prior = self.declared(name)
        self.frames[-1].names[name] = ty
        self.loop_expired.pop(name, None)
        if prior is not None:
            # Already declared in Raw IR (a loop variable after its loop): reuse it.
            self.expect(ty, prior, value_node, f"'{name}' was a '{prior}' loop variable and is assigned")
            return {"kind": "assign", "target": name, "value": value, "line": s.lineno}
        self.frames[-1].declared[name] = ty
        self.taken.add(name)
        return {"kind": "let", "name": name, "mutable": self.binding_counts[name] > 1, "tsType": ty.raw,
                "init": value, "line": s.lineno}

    def aug_assign(self, s: ast.AugAssign) -> Raw:
        if isinstance(s.target, ast.Subscript):
            cur = ast.Subscript(s.target.value, s.target.slice, ast.Load(), lineno=s.lineno, col_offset=s.col_offset)
            new = ast.BinOp(cur, s.op, s.value, lineno=s.lineno, col_offset=s.col_offset)
            return self.set_item(s.target, new, s)
        if not isinstance(s.target, ast.Name):
            raise _err("only assignment to a single name is supported yet", s)
        name = s.target.id
        if name in self.loop_vars:
            raise _err(f"cannot assign the loop variable '{name}' inside its loop "
                       "(Python would not change the iteration)", s)
        ty = self.lookup(name)
        if ty is None:
            raise _err(f"'{name}' is assigned with '{ast.unparse(s)}' before it is bound", s)
        if ty.is_list:
            if not isinstance(s.op, ast.Add):
                raise _err(f"'{ast.unparse(s)}' is not supported on lists", s)
            # `xs += ys` extends xs in place (unlike `xs = xs + ys`).
            self.mutate(name, s)
            other, oty = self.expr(s.value, ty, share=False)
            self.expect(oty, ty, s.value, f"'{name} +=' gets")
            return {"kind": "assign", "target": name, "value": binop("arrayConcat", var(name), other), "line": s.lineno}
        value, _ = self.arith(s.op, (var(name), ty), self.expr(s.value), s)
        return {"kind": "assign", "target": name, "value": value, "line": s.lineno}

    def set_item(self, target: ast.Subscript, value_node: ast.expr, s: ast.stmt) -> Raw:
        if not isinstance(target.value, ast.Name):
            raise _err("nested in-place modification (e.g. 'a[i][j] = v') is not supported", s)
        name = target.value.id
        ty = self.lookup(name)
        if ty is None or not ty.is_list:
            raise _err(f"item assignment needs a list, got '{name}'", s)
        assert ty.elem is not None
        if isinstance(target.slice, ast.Slice):
            raise _err("slice assignment is not supported", s)
        self.mutate(name, s)
        idx = self.index_expr(var(name), target.slice)
        value, vty = self.expr(value_node, ty.elem)
        self.expect(vty, ty.elem, value_node, f"'{name}[...]' holds '{ty.elem}' and is assigned")
        return {"kind": "assign", "target": name, "value": method(var(name), "with", idx, value), "line": s.lineno}

    def expr_stmt(self, s: ast.Expr) -> Raw:
        c = s.value
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute):
            recv, meth = c.func.value, c.func.attr
            rty = self.static_type(recv)
            if rty is not None and rty.is_list:
                if not isinstance(recv, ast.Name):
                    raise _err(f"nested in-place modification (e.g. 'a[i].{meth}(...)') is not supported", s)
                if meth in UNSUPPORTED_LIST_METHODS:
                    raise _err(f"'{meth}' is not supported yet", s)
                if meth in MUTATING_METHODS:
                    return self.list_mutation(recv.id, rty, meth, c, s)
        raise _err("expression statements are not supported (verified code has no side effects)", s)

    def list_mutation(self, name: str, ty: PyType, meth: str, c: ast.Call, s: ast.stmt) -> Raw:
        assert ty.elem is not None
        if c.keywords or len(c.args) != 1:
            raise _err(f"'{meth}' takes one positional argument", s)
        self.mutate(name, s)
        if meth == "append":
            arg, aty = self.expr(c.args[0], ty.elem)
            self.expect(aty, ty.elem, c.args[0], f"'{name}.append' gets")
            return {"kind": "expr", "expr": method(var(name), "push", arg), "line": s.lineno}
        arg, aty = self.expr(c.args[0], ty, share=False)
        self.expect(aty, ty, c.args[0], f"'{name}.extend' gets")
        return {"kind": "assign", "target": name, "value": binop("arrayConcat", var(name), arg), "line": s.lineno}

    def mutate(self, name: str, node: ast.AST) -> None:
        if name in self.iterating:
            raise _err(f"'{name}' is modified inside a loop over it", node)
        self.mutations.setdefault(name, []).append(node)

    # ── loops ────────────────────────────────────────────────

    def loop_specs(self, loop: ast.While | ast.For) -> tuple[list[Raw], Raw | None, Raw | None]:
        first = loop.body[0].lineno
        invariants: list[Raw] = []
        decreases: Raw | None = None
        done_with: Raw | None = None
        for a in self.claim(LOOP_KEYWORDS, loop.lineno, first, min_col=loop.col_offset):
            if a.keyword == "invariant":
                invariants.append(self.spec(a.text, a.line))
            elif a.keyword == "decreases":
                decreases = self.spec_value(a.text, a.line)
            else:
                done_with = self.spec(a.text, a.line)
        return invariants, decreases, done_with

    def while_(self, s: ast.While, boundary: int) -> Raw:
        if s.orelse:
            raise _err("'while ... else' is not supported", s)
        cond = self.cond(s.test)
        invariants, decreases, done_with = self.loop_specs(s)
        self.loops.append(_Loop(None))
        try:
            body = self.nested(s.body, s.lineno, boundary, s.end_lineno or s.lineno)
        finally:
            self.loops.pop()
        return {"kind": "while", "cond": cond, "invariants": invariants, "decreases": decreases,
                "doneWith": done_with, "body": body, "line": s.lineno}

    def loop_counter(self, name: str, init: Raw, line: int) -> Raw:
        """Bind a desugared loop's counter (re-using a previous Raw IR declaration of it)."""
        prior = self.declared(name)
        visible = self.lookup(name)
        if visible is not None and visible != INT:
            raise UnsupportedError(f"loop variable '{name}' was a '{visible}'", line)
        self.frames[-1].names[name] = INT
        self.loop_expired.pop(name, None)
        if prior is not None or visible is not None:
            return {"kind": "assign", "target": name, "value": init, "line": line}
        self.frames[-1].declared[name] = INT
        self.taken.add(name)
        return {"kind": "let", "name": name, "mutable": True, "tsType": "int", "init": init, "line": line}

    def stable(self, node: ast.expr, body: list[ast.stmt]) -> bool:
        """``node`` evaluates to the same value throughout ``body`` (a literal or an unassigned name)."""
        if isinstance(node, ast.Constant) or (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)):
            return True
        if isinstance(node, ast.Name):
            return node.id not in _assigned_names(body)
        return False

    def bound_once(self, base: str, node: ast.expr, body: list[ast.stmt], line: int, out: list[Raw]) -> Raw:
        """Python evaluates `range(...)` bounds once; keep that by binding non-stable ones to a fresh local."""
        raw, ty = self.expr(node)
        self.need_int(ty, node, "range()")
        if self.stable(node, body):
            return raw
        name = self.fresh(base)
        self.frames[-1].declared[name] = INT
        out.append({"kind": "let", "name": name, "mutable": False, "tsType": "int", "init": raw, "line": line})
        return var(name)

    def for_(self, s: ast.For, boundary: int) -> list[Raw]:
        if s.orelse:
            raise _err("'for ... else' is not supported", s)
        it, tgt, line = s.iter, s.target, s.lineno
        out: list[Raw] = []
        callee = it.func.id if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and not it.keywords else None
        args = it.args if isinstance(it, ast.Call) else []
        if callee == "range":
            if not isinstance(tgt, ast.Name):
                raise _err("a range loop needs a single loop variable", s)
            if len(args) not in (1, 2):
                raise _err("only range(stop) and range(start, stop) are supported yet", s)
            i = tgt.id
            start_node = args[0] if len(args) == 2 else ast.Constant(0, lineno=line, col_offset=0)
            stop_node = args[-1]
            start = self.bound_once(f"{i}_start", start_node, s.body, line, out)
            stop = self.bound_once(f"{i}_stop", stop_node, s.body, line, out)
            out.append(self.loop_counter(i, start, line))
            auto = binop("&&", binop("<=", start, var(i)), binop("||", binop("<=", var(i), stop), binop("===", var(i), start)))
            inc = {"kind": "assign", "target": i, "value": binop("+", var(i), num(1)), "line": line}
            out.append(self.counted_loop(s, i, binop("<", var(i), stop), auto, inc, None, boundary))
            return out
        if callee == "enumerate":
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts)):
                raise _err("enumerate loops are written 'for i, x in enumerate(xs)'", s)
            if len(args) != 1 or not isinstance(args[0], ast.Name):
                raise _err("enumerate(...) is supported over a list variable only", s)
            seq, sty = self.expr(args[0], share=False)
            if not sty.is_list:
                raise _err(f"enumerate needs a list, got '{sty}'", s)
            assert sty.elem is not None
            i, x = (e.id for e in tgt.elts if isinstance(e, ast.Name))
            out.append(self.loop_counter(i, num(0), line))
            auto = binop("&&", binop("<=", num(0), var(i)), binop("<=", var(i), length(seq)))
            inc = {"kind": "assign", "target": i, "value": binop("+", var(i), num(1)), "line": line}
            elem = (x, sty.elem, {"kind": "index", "obj": seq, "idx": var(i)})
            self.iterating.append(args[0].id)
            try:
                out.append(self.counted_loop(s, i, binop("<", var(i), length(seq)), auto, inc, elem, boundary))
            finally:
                self.iterating.pop()
            return out
        if isinstance(tgt, ast.Tuple):
            raise _err("tuple loop targets are only supported with enumerate(...)", s)
        if not isinstance(tgt, ast.Name):
            raise _err("a for loop needs a single loop variable", s)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("reversed", "sorted", "zip"):
            raise _err(f"'{it.func.id}(...)' is not supported yet", s)
        seq, sty = self.expr(it, share=False)
        if not sty.is_list:
            raise _err(f"cannot iterate over '{sty}'", s)
        assert sty.elem is not None
        x = tgt.id
        if self.lookup(x) is not None or self.declared(x) is not None:
            raise _err(f"the loop variable '{x}' is already bound; use a new name", s)
        invariants, decreases, done_with = self.loop_specs(s)
        if decreases is not None:
            raise _err("a 'for x in xs' loop terminates on its own; '#@ decreases' is not used", s)
        self.push()
        self.frames[-1].names[x] = sty.elem
        self.frames[-1].declared[x] = sty.elem
        self.taken.add(x)
        self.loop_vars.add(x)
        self.loops.append(_Loop(None))
        if isinstance(it, ast.Name):
            self.iterating.append(it.id)
        try:
            body = self.body(s.body, line, boundary)
        finally:
            if isinstance(it, ast.Name):
                self.iterating.pop()
            self.loops.pop()
            self.loop_vars.discard(x)
            self.pop(s.end_lineno or line)
        self.expired.pop(x, None)
        self.loop_expired[x] = s.end_lineno or line
        return [{"kind": "forof", "names": [x], "iterable": seq, "invariants": invariants, "doneWith": done_with,
                 "body": body, "line": line}]

    def counted_loop(self, s: ast.For, i: str, cond: Raw, auto: Raw, inc: Raw,
                     elem: tuple[str, PyType, Raw] | None, boundary: int) -> Raw:
        invariants, decreases, done_with = self.loop_specs(s)
        self.loop_vars.add(i)
        self.loops.append(_Loop(inc))
        self.push()
        pre: list[Raw] = []
        if elem is not None:
            x, xty, init = elem
            if self.lookup(x) is not None or self.declared(x) is not None:
                raise _err(f"the loop variable '{x}' is already bound; use a new name", s)
            self.frames[-1].names[x] = xty
            self.frames[-1].declared[x] = xty
            self.taken.add(x)
            self.loop_vars.add(x)
            pre.append({"kind": "let", "name": x, "mutable": False, "tsType": xty.raw, "init": init, "line": s.lineno})
        try:
            body = self.body(s.body, s.lineno, boundary)
        finally:
            self.loops.pop()
            if elem is not None:
                self.loop_vars.discard(elem[0])
            self.pop(s.end_lineno or s.lineno)
        end = s.end_lineno or s.lineno
        if elem is not None:
            self.expired.pop(elem[0], None)
            self.loop_expired[elem[0]] = end
        self.hide_loop_var(i, end)
        if decreases is None:
            # A counted loop's measure is implied by its bounds; Dafny infers it
            # from `i < stop` with `i` increasing.
            pass
        return {"kind": "while", "cond": cond, "invariants": [auto, *invariants], "decreases": decreases,
                "doneWith": done_with, "body": [*pre, *body, inc], "line": s.lineno}

    # ── expressions ──────────────────────────────────────────

    def expect(self, got: PyType, want: PyType | None, node: ast.AST, what: str) -> None:
        if want is not None and got != want and ANY not in (got, want):
            raise _err(f"{what} a value of type '{got}', expected '{want}'", node)

    def check_type(self, got: PyType, want: PyType, a: Annotation, what: str) -> None:
        if got != want and ANY not in (got, want):
            raise UnsupportedError(f"{what} a value of type '{got}', expected '{want}'", a.line, a.col)

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
        return self.truthy(self.expr(node, share=False), node)

    def truthy(self, e: Typed, node: ast.AST) -> Raw:
        raw, ty = e
        if ty in (BOOL, ANY):
            return raw
        if ty == INT:
            return binop("!==", raw, num(0))
        if ty.is_list:
            return binop(">", length(raw), num(0))
        raise _err(f"truthiness of '{ty}' is not supported", node)

    def negated(self, node: ast.expr) -> Raw:
        """``not node`` — Python's negated truthiness."""
        if not isinstance(node, (ast.BoolOp, ast.UnaryOp)):
            raw, ty = self.expr(node, share=False)
            if ty == INT:
                return binop("===", raw, num(0))
            if ty.is_list:
                return binop("===", length(raw), num(0))
            if ty in (BOOL, ANY):
                return unop("!", raw)
            raise _err(f"truthiness of '{ty}' is not supported", node)
        return unop("!", self.cond(node))

    def static_type(self, node: ast.expr) -> PyType | None:
        """The type of a simple expression without lowering it (or None)."""
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            base = self.static_type(node.value)
            return base.elem if base is not None and base.is_list else None
        return None

    def expr(self, node: ast.expr, expected: PyType | None = None, share: bool = True) -> Typed:
        """Lower and type ``node``. ``share`` is False where a list value is only read
        (its identity cannot escape), which the aliasing check needs to know."""
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
            return self.name(node, share)
        if isinstance(node, ast.BinOp):
            return self.binop(node, expected)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return self.negated(node.operand), BOOL
            raw, ty = self.expr(node.operand)
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                self.need_int(ty, node, "unary '-'" if isinstance(node.op, ast.USub) else "unary '+'")
                return (unop("-", raw) if isinstance(node.op, ast.USub) else raw), INT
            raise _err("'~' is not supported yet", node)
        if isinstance(node, ast.BoolOp):
            parts = [self.expr(v, share=False) for v in node.values]
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
            a, ta = self.expr(node.body, expected, share)
            b, tb = self.expr(node.orelse, expected if ta == ANY else ta, share)
            if ta != tb and ANY not in (ta, tb):
                raise _err(f"conditional expression branches have different types '{ta}' and '{tb}'", node)
            return {"kind": "conditional", "cond": cond, "then": a, "else": b}, (ta if ta != ANY else tb)
        if isinstance(node, ast.Call):
            return self.call(node, expected, share)
        if isinstance(node, ast.List):
            return self.list_literal(node, expected)
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.ListComp):
            return self.list_comp(node)
        if isinstance(node, (ast.Tuple, ast.Dict, ast.Set, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                             ast.Attribute, ast.JoinedStr, ast.Lambda, ast.Starred)):
            raise _err(f"'{ast.unparse(node)}' is not supported yet", node)
        raise _err(f"'{ast.unparse(node)}' is not supported", node)

    def name(self, node: ast.Name, share: bool = True) -> Typed:
        n = node.id
        ty = self.lookup(n)
        if ty is not None:
            if self.is_ghost(n) and not self.in_spec:
                raise _err(f"ghost variable '{n}' cannot be used in code", node)
            if ty.is_list and not self.in_spec:
                self.uses.setdefault(n, []).append(_Use(share, node))
            return var(n), ty
        if self.in_spec and n == "result":
            if not self.allow_result:
                raise _err("'result' is only valid in '#@ ensures'", node)
            return {"kind": "result"}, self.sig.returns
        if n in self.loop_expired:
            raise _err(f"'{n}' is not available after its loop (Python leaves it at the last element, "
                       f"or unbound if the loop never ran); keep the value you need in another variable", node)
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

    def arith(self, op: ast.operator, left: Typed, right: Typed, node: ast.AST) -> Typed:
        if isinstance(op, ast.Div):
            raise _err("'/' produces a float; use '//' for integer division", node)
        sym = _ARITH.get(type(op))
        if sym is None:
            raise _err(f"operator '{type(op).__name__}' is not supported yet", node)
        self.need_int(left[1], node, f"'{sym}'")
        self.need_int(right[1], node, f"'{sym}'")
        return binop(sym, left[0], right[0]), INT

    def binop(self, node: ast.BinOp, expected: PyType | None = None) -> Typed:
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
        want = expected if expected is not None and expected.is_list else None
        left = self.expr(node.left, want, share=False)
        right = self.expr(node.right, want or (left[1] if left[1].is_list else None), share=False)
        if left[1].is_list or right[1].is_list:
            if not isinstance(node.op, ast.Add):
                raise _err(f"'{ast.unparse(node)}' is not supported on lists", node)
            if left[1] != right[1]:
                raise _err(f"cannot concatenate '{left[1]}' and '{right[1]}'", node)
            return binop("arrayConcat", left[0], right[0]), left[1]
        return self.arith(node.op, left, right, node)

    def compare(self, node: ast.Compare) -> Raw:
        operands = [node.left, *node.comparators]
        for mid in operands[1:-1]:
            if not self.is_simple(mid):
                raise _err("the middle of a chained comparison cannot contain a call "
                           "(Python evaluates it once); bind it to a variable first", node)
        out: Raw | None = None
        lowered: list[Typed | None] = [None] * len(operands)
        for i, op in enumerate(node.ops):
            if isinstance(op, (ast.Is, ast.IsNot)):
                raise _err("'is' is only supported against None", node)
            if lowered[i] is None:
                lowered[i] = self.expr(operands[i], share=False)
            l, lt = lowered[i]  # type: ignore[misc]
            if isinstance(op, (ast.In, ast.NotIn)):
                r, rt = self.expr(operands[i + 1], list_of(lt) if lt != ANY else None, share=False)
                lowered[i + 1] = (r, rt)
                if not rt.is_list and rt != ANY:
                    raise _err(f"'in' needs a list on the right, got '{rt}'", node)
                if rt.is_list and lt != rt.elem and lt != ANY:
                    raise _err(f"'in' between '{lt}' and '{rt}'", node)
                c = method(r, "includes", l)
                c = unop("!", c) if isinstance(op, ast.NotIn) else c
            else:
                if lowered[i + 1] is None:
                    lowered[i + 1] = self.expr(operands[i + 1], lt if lt.is_list else None, share=False)
                r, rt = lowered[i + 1]  # type: ignore[misc]
                src = _CMP_SRC[type(op)]
                if lt.is_list and rt.is_list and lt != rt:
                    # An empty literal adapts to the other side's element type.
                    raise _err(f"'{src}' between '{lt}' and '{rt}' is not supported", node)
                if lt != rt and ANY not in (lt, rt):
                    raise _err(f"'{src}' between '{lt}' and '{rt}' is not supported "
                               f"(bool is an int subclass in Python; compare like types)", node)
                if src in ("<", "<=", ">", ">="):
                    if BOOL in (lt, rt):
                        raise _err(f"'{src}' on 'bool' is not supported", node)
                    if lt.is_list:
                        raise _err(f"'{src}' on '{lt}' is not supported", node)
                c = binop(_CMP[type(op)], l, r)
            out = c if out is None else binop("&&", out, c)
        assert out is not None
        return out

    @staticmethod
    def is_simple(node: ast.expr) -> bool:
        """Safe to evaluate twice: subset expressions have no side effects, so only
        calls (which the engine may lift into separate statements) are excluded."""
        return not any(isinstance(n, ast.Call) for n in ast.walk(node))

    # ── lists ────────────────────────────────────────────────

    def list_literal(self, node: ast.List, expected: PyType | None) -> Typed:
        want = expected.elem if expected is not None and expected.is_list else None
        elems: list[Raw] = []
        ety = want
        for e in node.elts:
            if isinstance(e, ast.Starred):
                raise _err("'*' unpacking in list literals is not supported yet", e)
            raw, t = self.expr(e, ety)
            if ety is not None and t != ety and ANY not in (t, ety):
                raise _err(f"list elements of types '{ety}' and '{t}' (lists hold one type)", e)
            ety = ety if ety is not None and ety != ANY else t
            elems.append(raw)
        if ety is None:
            raise _err("cannot infer the element type of '[]'; annotate the variable (e.g. 'ys: list[int] = []')", node)
        return {"kind": "arrayLiteral", "elems": elems}, list_of(ety)

    def index_expr(self, seq: Raw, idx_node: ast.expr) -> Raw:
        """A list index. A negative literal counts from the end, as in Python; any other
        index must be proven in range by the verifier (Dafny's bounds check)."""
        if (isinstance(idx_node, ast.UnaryOp) and isinstance(idx_node.op, ast.USub)
                and isinstance(idx_node.operand, ast.Constant) and type(idx_node.operand.value) is int):
            return binop("-", length(seq), num(idx_node.operand.value))
        raw, ty = self.expr(idx_node)
        self.need_int(ty, idx_node, "a list index")
        return raw

    def subscript(self, node: ast.Subscript) -> Typed:
        seq, sty = self.expr(node.value, share=False)
        if not sty.is_list:
            raise _err(f"indexing '{sty}' is not supported yet", node)
        assert sty.elem is not None
        if isinstance(node.slice, ast.Slice):
            sl = node.slice
            if sl.step is not None:
                raise _err("slices with a step are not supported", node)
            lo = self.slice_bound(sl.lower, num(0))
            hi = self.slice_bound(sl.upper, length(seq))
            # The file enables `safe-slice`: Python's clamping, negative-from-the-end slices.
            return method(seq, "slice", lo, hi), sty
        return {"kind": "index", "obj": seq, "idx": self.index_expr(seq, node.slice)}, sty.elem

    def slice_bound(self, node: ast.expr | None, default: Raw) -> Raw:
        if node is None:
            return default
        raw, ty = self.expr(node)
        self.need_int(ty, node, "a slice bound")
        return raw

    def comprehension(self, node: ast.ListComp | ast.GeneratorExp) -> tuple[ast.comprehension, str]:
        if len(node.generators) != 1:
            raise _err("comprehensions with one 'for' clause only are supported", node)
        gen = node.generators[0]
        if gen.is_async:
            raise _err("async comprehensions are not supported", node)
        if not isinstance(gen.target, ast.Name):
            raise _err("a comprehension needs a single loop variable", node)
        x = gen.target.id
        if self.lookup(x) is not None:
            raise _err(f"comprehension variable '{x}' shadows an existing name; use a new name", node)
        return gen, x

    def lambda_(self, x: str, xty: PyType, body: ast.expr, as_cond: bool) -> Typed:
        self.push()
        self.frames[-1].names[x] = xty
        self.frames[-1].declared[x] = xty
        try:
            if as_cond:
                raw, ty = self.cond(body), BOOL
            else:
                raw, ty = self.expr(body)
        finally:
            self.frames.pop()
        return {"kind": "lambda", "params": [{"name": x, "tsType": xty.raw}], "body": raw,
                "returnTsType": ty.raw}, ty

    def iter_source(self, gen: ast.comprehension, node: ast.AST) -> Typed:
        it = gen.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            raise _err("'range' outside a 'for' loop is only supported in specs", node)
        seq, sty = self.expr(it, share=False)
        if not sty.is_list:
            raise _err(f"cannot iterate over '{sty}'", node)
        return seq, sty

    def guard(self, gen: ast.comprehension) -> ast.expr | None:
        if not gen.ifs:
            return None
        if len(gen.ifs) == 1:
            return gen.ifs[0]
        return ast.BoolOp(ast.And(), list(gen.ifs), lineno=gen.ifs[0].lineno, col_offset=gen.ifs[0].col_offset)

    def list_comp(self, node: ast.ListComp) -> Typed:
        gen, x = self.comprehension(node)
        seq, sty = self.iter_source(gen, node)
        assert sty.elem is not None
        cond = self.guard(gen)
        out = seq
        if cond is not None:
            lam, _ = self.lambda_(x, sty.elem, cond, as_cond=True)
            out = method(out, "filter", lam)
        if isinstance(node.elt, ast.Name) and node.elt.id == x:
            return out, sty
        lam, ety = self.lambda_(x, sty.elem, node.elt, as_cond=False)
        return method(out, "map", lam), list_of(ety)

    def all_any(self, name: str, node: ast.Call) -> Typed:
        if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
            raise _err(f"{name}() is supported over a generator, e.g. {name}(p(x) for x in xs)", node)
        g = node.args[0]
        gen, x = self.comprehension(g)
        cond = self.guard(gen)
        it = gen.iter
        is_range = isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
        if self.in_spec:
            # Specs: a bounded quantifier, the form the prover reasons with best.
            self.push()
            try:
                if is_range:
                    assert isinstance(it, ast.Call)
                    if it.keywords or len(it.args) not in (1, 2):
                        raise _err("only range(stop) and range(start, stop) are supported yet", node)
                    lo = self.expr(it.args[0])[0] if len(it.args) == 2 else num(0)
                    hi = self.expr(it.args[-1])[0]
                    xty, vt = INT, "int"
                    domain = [binop("<=", lo, var(x)), binop("<", var(x), hi)]
                else:
                    seq, sty = self.expr(it, share=False)
                    if not sty.is_list:
                        raise _err(f"cannot iterate over '{sty}'", node)
                    assert sty.elem is not None
                    xty, vt = sty.elem, sty.elem.raw
                    domain = [method(seq, "includes", var(x))]
                self.frames[-1].names[x] = xty
                if cond is not None:
                    domain.append(self.cond(cond))
                body = self.cond(g.elt)
            finally:
                self.frames.pop()
            if name == "all":
                return {"kind": "forall", "var": x, "varType": vt, "body": binop("==>", conj(domain), body)}, BOOL
            return {"kind": "exists", "var": x, "varType": vt, "body": conj([*domain, body])}, BOOL
        seq, sty = self.iter_source(gen, node)
        assert sty.elem is not None
        pred: ast.expr
        if cond is not None:
            pred = (ast.BoolOp(ast.Or(), [ast.UnaryOp(ast.Not(), cond), g.elt]) if name == "all"
                    else ast.BoolOp(ast.And(), [cond, g.elt]))
            ast.copy_location(pred, g.elt)
            ast.fix_missing_locations(pred)
        else:
            pred = g.elt
        lam, _ = self.lambda_(x, sty.elem, pred, as_cond=True)
        return method(seq, "every" if name == "all" else "some", lam), BOOL

    # ── calls ────────────────────────────────────────────────

    def call(self, node: ast.Call, expected: PyType | None = None, share: bool = True) -> Typed:
        if self.in_spec and isinstance(node.func, ast.Name) and node.func.id in ("forall", "exists"):
            return self.quantifier(node.func.id, node)
        if node.keywords:
            raise _err("keyword arguments are not supported", node)
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise _err("'*' arguments are not supported", node)
        if isinstance(node.func, ast.Attribute):
            rty = self.static_type(node.func.value)
            meth = node.func.attr
            if rty is not None and rty.is_list and (meth in UNSUPPORTED_LIST_METHODS or meth in MUTATING_METHODS):
                if meth in MUTATING_METHODS:
                    raise _err(f"'{meth}' changes the list and returns None; call it as a statement", node)
                raise _err(f"'{meth}' is not supported yet", node)
            raise _err(f"method call '{ast.unparse(node)}' is not supported here", node)
        if not isinstance(node.func, ast.Name):
            raise _err(f"call '{ast.unparse(node)}' is not supported yet", node)
        name = node.func.id
        if self.lookup(name) is not None:
            raise _err(f"'{name}' is not a function", node)
        if name in self.functions:
            return self.user_call(self.functions[name], node)
        if name in (specs.IMPLIES, specs.IFF) and self.in_spec:
            a, b = (self.cond(x) for x in node.args)
            return binop("==>" if name == specs.IMPLIES else "<==>", a, b), BOOL
        if name in ("all", "any"):
            return self.all_any(name, node)
        if name == "len" and len(node.args) == 1:
            raw, ty = self.expr(node.args[0], share=False)
            if not ty.is_list:
                raise _err(f"len() of '{ty}' is not supported yet", node)
            return length(raw), INT
        if name == "list":
            if len(node.args) == 0:
                if expected is None or not expected.is_list:
                    raise _err("cannot infer the element type of 'list()'; annotate the variable", node)
                return {"kind": "arrayLiteral", "elems": []}, expected
            raw, ty = self.expr(node.args[0], share=False)
            if len(node.args) != 1 or not ty.is_list:
                raise _err("list(...) is supported on a list (a copy)", node)
            return raw, ty  # a copy is the same value in the model
        if name in ("min", "max") and len(node.args) == 1:
            raw, ty = self.expr(node.args[0], share=False)
            if not (ty.is_list and ty.elem == INT):
                raise _err(f"{name}() of '{ty}' is not supported", node)
            # Python raises ValueError on an empty list; MinOfSeq/MaxOfSeq require |s| > 0.
            return {"kind": "call", "fn": var("MinOfSeq" if name == "min" else "MaxOfSeq"), "args": [raw]}, INT
        if name in ("sum", "sorted", "reversed", "range", "zip", "map", "filter"):
            raise _err(f"'{name}(...)' is not supported yet", node)
        args = [self.expr(a, share=False) for a in node.args]
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

    def user_call(self, sig: Signature, node: ast.Call) -> Typed:
        if self.verify_mode and not sig.verified and not self.in_spec:
            raise _err(f"'{self.sig.name}' calls '{sig.name}', which is not marked '#@ verify'", node)
        if len(node.args) != len(sig.params):
            raise _err(f"'{sig.name}' takes {len(sig.params)} argument(s), got {len(node.args)}", node)
        args: list[Raw] = []
        for a, (pname, pty) in zip(node.args, sig.params):
            # A list passed to a user function may be returned or stored by it: shared.
            raw, ty = self.expr(a, pty, share=True)
            self.expect(ty, pty, a, f"argument '{pname}' of '{sig.name}' gets")
            args.append(raw)
        return {"kind": "call", "fn": var(sig.name), "args": args}, sig.returns

    def quantifier(self, kind: str, node: ast.Call) -> Typed:
        if len(node.args) != 2 or not isinstance(node.args[0], ast.Name):
            raise _err(f"a quantifier is written {kind}(k, P) or {kind}(k: T, P)", node)
        k = node.args[0].id
        ty_text = next((kw.value.value for kw in node.keywords if kw.arg == specs.QUANT_TYPE
                        and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)), None)
        if self.lookup(k) is not None:
            raise _err(f"quantified variable '{k}' shadows an existing name; use a new name", node)
        if ty_text is None:
            kty, vt = INT, "int"
        elif ty_text == "nat":
            kty, vt = INT, "nat"
        else:
            kty = from_type_text(ty_text, node.lineno)
            vt = kty.raw
        self.push()
        self.frames[-1].names[k] = kty
        try:
            body = self.cond(node.args[1])
        finally:
            self.frames.pop()
        return {"kind": kind, "var": k, "varType": vt, "body": body}, BOOL

    # ── specs ────────────────────────────────────────────────

    def spec_typed(self, text: str, line: int) -> Typed:
        tree = specs.parse(text, line)
        self.in_spec = True
        try:
            return self.expr(tree)
        finally:
            self.in_spec = False

    def spec_value(self, text: str, line: int) -> Raw:
        """A non-boolean spec expression (e.g. a ``decreases`` measure)."""
        return self.spec_typed(text, line)[0]

    def spec(self, text: str, line: int, *, allow_result: bool = False) -> Raw:
        tree = specs.parse(text, line)
        self.in_spec, self.allow_result = True, allow_result
        try:
            return self.cond(tree)
        finally:
            self.in_spec, self.allow_result = False, False

    # ── aliasing ─────────────────────────────────────────────

    def check_aliasing(self) -> None:
        """The fresh-local rule: a list changed in place must be a local bound only to
        fresh lists and never shared, so no other reference can observe the change."""
        for name, sites in self.mutations.items():
            first = sites[0]
            if name in self.params:
                raise _err(f"cannot modify parameter '{name}' in place (Python would change the caller's "
                           f"list); copy it first, e.g. 'ys = list({name})'", first)
            for b in self.bindings.get(name, []):
                if not b.fresh:
                    raise _err(f"'{name}' is modified in place, so it must be bound to a fresh list "
                               "(a literal, comprehension, list(...), slice or concatenation)", b.node)
            for u in self.uses.get(name, []):
                if u.shared:
                    raise _err(f"'{name}' is modified in place, so it cannot be shared: store or pass "
                               f"a copy instead (e.g. list({name}))", u.node)


def _is_fresh(node: ast.expr) -> bool:
    """``node`` evaluates to a new list no other reference can see."""
    if isinstance(node, (ast.List, ast.ListComp)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list":
        return True
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return True
    if isinstance(node, ast.IfExp):
        return _is_fresh(node.body) and _is_fresh(node.orelse)
    return False


def _assigned_names(stmts: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Assign):
                out.update(t.id for t in n.targets if isinstance(t, ast.Name))
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
    return out


def always_exits(stmts: list[ast.stmt]) -> bool:
    """Every path through ``stmts`` ends in ``return`` or ``raise``."""
    for s in stmts:
        if isinstance(s, (ast.Return, ast.Raise)):
            return True
        if isinstance(s, ast.Assert) and isinstance(s.test, ast.Constant) and s.test.value is False:
            return True
        if isinstance(s, ast.If) and always_exits(s.body) and always_exits(s.orelse):
            return True
        if (isinstance(s, ast.While) and isinstance(s.test, ast.Constant) and s.test.value is True
                and not any(isinstance(n, ast.Break) for n in _same_loop_nodes(s.body))):
            return True
    return False


def _same_loop_nodes(stmts: list[ast.stmt]) -> list[ast.AST]:
    """Nodes in ``stmts`` that belong to this loop (not to nested loops or functions)."""
    out: list[ast.AST] = []
    todo: list[ast.AST] = list(stmts)
    while todo:
        n = todo.pop()
        out.append(n)
        if isinstance(n, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            todo.extend(n.orelse if isinstance(n, (ast.For, ast.While)) else [])
            continue
        todo.extend(ast.iter_child_nodes(n))
    return out


def binding_counts(fn: ast.FunctionDef) -> Counter[str]:
    """How often each name is (re)bound or changed in place; > 1 means a mutable `let`."""
    counts: Counter[str] = Counter()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    counts[t.id] += 1
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    counts[t.value.id] += 2
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            t = node.target
            if isinstance(t, ast.Name):
                counts[t.id] += 1 if isinstance(node, ast.AnnAssign) else 2
            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                counts[t.value.id] += 2
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)):
            counts[node.value.func.value.id] += 2
    return counts
