"""Module-level extraction: Python source → Raw IR envelope for ``lsc``."""
from __future__ import annotations

import ast
import builtins
from collections import Counter
from pathlib import Path
from typing import Any

from lemmascript_python import __version__
from lemmascript_python.annotations import (
    FILE_KEYWORDS,
    FUNCTION_KEYWORDS,
    KNOWN_KEYWORDS,
    LOOP_KEYWORDS,
    STATEMENT_KEYWORDS,
    Annotation,
    collect,
)
from lemmascript_python.errors import UnsupportedError
from lemmascript_python.lower import FunctionLowerer, Signature, always_exits, binding_counts
from lemmascript_python.pytypes import from_annotation
from lemmascript_python.reserved import reserved_reason

RAW_IR_FORMAT = "lemmascript-raw-ir"
RAW_IR_VERSION = 1

#: Modules whose imports carry no runtime behaviour the model must know about.
ALLOWED_IMPORTS = frozenset({"__future__", "typing", "typing_extensions"})
FUTURE_NAMES = frozenset({"annotations"})
#: Names usable only in annotations; none of them is callable in the subset.
TYPING_NAMES = frozenset({"Final", "List", "Optional", "Union", "Literal", "NamedTuple", "TypedDict",
                          "TypeAlias", "Dict", "Set", "Tuple", "FrozenSet", "Sequence", "Mapping"})

#: File-overridable lsc options (tools/src/config.ts), with their allowed values.
FILE_OPTIONS: dict[str, tuple[str, ...]] = {
    "safe-slice": ("true",),
    "extern-default": ("pure", "impure"),
    "string-semantics": ("unicode-scalar",),
    "dafny-library": ("stdlib", "local"),
}


def _err(msg: str, node: ast.AST) -> UnsupportedError:
    return UnsupportedError(msg, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _is_docstring(s: ast.stmt) -> bool:
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def _is_main_guard(s: ast.stmt) -> bool:
    if not isinstance(s, ast.If) or s.orelse:
        return False
    t = s.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__")


class _Module:
    def __init__(self, source: str, filename: str) -> None:
        self.source = source
        self.filename = filename
        self.tree = ast.parse(source, filename)
        self.annotations = collect(source)
        for a in self.annotations:
            if a.keyword not in KNOWN_KEYWORDS:
                raise UnsupportedError(f"unknown annotation '#@ {a.keyword}'", a.line, a.col)

    # ── file directives ──────────────────────────────────────

    def directives(self) -> dict[str, Any]:
        first = next((s.lineno for s in self.tree.body if not _is_docstring(s)), 10 ** 9)
        backend: str | None = None
        options: dict[str, Any] = {}
        for a in self.annotations:
            if a.line >= first or a.keyword not in FILE_KEYWORDS:
                continue
            a.claimed = True
            if a.keyword == "backend":
                if a.text != "dafny":
                    raise UnsupportedError(f"'#@ backend {a.text}': the Python frontend supports only the Dafny "
                                           "backend", a.line, a.col)
                backend = a.text
            else:
                key, _, value = a.text.partition(" ")
                value = value.strip()
                if key == "safe-slice" and value == "false":
                    raise UnsupportedError("'safe-slice false' is not allowed: Python slices always clamp",
                                           a.line, a.col)
                if key == "string-semantics" and value == "javascript-utf16":
                    raise UnsupportedError("'string-semantics javascript-utf16' does not model Python strings "
                                           "(Python str is a sequence of code points)", a.line, a.col)
                if key not in FILE_OPTIONS:
                    raise UnsupportedError(f"unknown option '{key}' (file options: {', '.join(FILE_OPTIONS)})",
                                           a.line, a.col)
                if value not in FILE_OPTIONS[key]:
                    raise UnsupportedError(f"option '{key}' must be one of: {', '.join(FILE_OPTIONS[key])}",
                                           a.line, a.col)
                options[key] = {"true": True, "false": False}.get(value, value)
        # Python slices clamp out-of-range bounds and count negative ones from the
        # end: exactly the engine's `safe-slice` model, so it is always on.
        options["safe-slice"] = True
        # Python's `==` compares contents (lists, records) — tell the engine.
        return {"backend": backend, "options": options, "leanModule": None, "equality": "structural"}

    # ── functions ────────────────────────────────────────────

    def head_annotations(self, fn: ast.FunctionDef) -> list[Annotation]:
        body = fn.body
        first = next((s for s in body if not _is_docstring(s)), None)
        stop = first.lineno if first is not None else (fn.end_lineno or fn.lineno) + 1
        doc_lines: set[int] = set()
        for s in body:
            if _is_docstring(s):
                doc_lines.update(range(s.lineno, (s.end_lineno or s.lineno) + 1))
            else:
                break
        out = []
        for a in self.annotations:
            # Only function-level keywords; `#@ assert`/`#@ ghost` before the first
            # statement belong to the body, and anything else is reported unclaimed.
            if (fn.lineno < a.line < stop and a.line not in doc_lines and a.col > fn.col_offset
                    and a.keyword in FUNCTION_KEYWORDS):
                a.claimed = True
                out.append(a)
        return out

    def signature(self, fn: ast.FunctionDef) -> Signature:
        args = fn.args
        if args.vararg or args.kwarg:
            raise _err("'*args'/'**kwargs' parameters are not supported", fn)
        if args.kwonlyargs:
            raise _err("keyword-only parameters are not supported", fn)
        if args.posonlyargs:
            raise _err("positional-only parameters are not supported yet", fn)
        if args.defaults or args.kw_defaults:
            raise _err("default parameter values are not supported", fn)
        params = []
        for a in args.args:
            if a.annotation is None:
                raise UnsupportedError(f"parameter '{a.arg}' needs a type annotation", a.lineno, a.col_offset)
            params.append((a.arg, from_annotation(a.annotation)))
        if fn.returns is None:
            raise _err(f"'{fn.name}' needs a return type annotation", fn)
        return Signature(fn.name, params, from_annotation(fn.returns))

    def check_import(self, s: ast.Import | ast.ImportFrom, after_def: bool) -> None:
        """Imports exist only for annotations, and must not bind a name verified code could call."""
        if after_def:
            raise _err("imports must come before the first function", s)
        if any(a.asname for a in s.names):
            raise _err("import aliases ('as') are not supported", s)
        if isinstance(s, ast.Import):
            for alias in s.names:
                if alias.name not in ALLOWED_IMPORTS:
                    raise _err(f"import of '{alias.name}' is not supported", s)
            return
        module = s.module or ""
        if s.level or module not in ALLOWED_IMPORTS:
            raise _err(f"import of '{'.' * s.level}{module}' is not supported", s)
        for alias in s.names:
            if alias.name == "*":
                raise _err(f"'from {module} import *' is not supported", s)
            allowed = FUTURE_NAMES if module == "__future__" else TYPING_NAMES
            if alias.name not in allowed:
                raise _err(f"'{alias.name}' cannot be imported (only annotation names: "
                           f"{', '.join(sorted(allowed))})", s)

    def check_main_guard(self, guard: ast.If, functions: set[str]) -> None:
        """The `__main__` block runs only as a script, but it shares the module namespace:
        it must not rebind a name verified code calls."""
        for n in ast.walk(guard):
            if n is guard:
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
                              ast.Global, ast.Nonlocal, ast.Delete)):
                raise _err("the '__main__' block may not bind or rebind names (definitions, imports, "
                           "global, del); it shares the namespace verified code runs in", n)
            targets: list[ast.expr] = []
            if isinstance(n, ast.Assign):
                targets = list(n.targets)
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
                targets = [n.target]
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                targets = [i.optional_vars for i in n.items if i.optional_vars is not None]
            for t in targets:
                for name in ast.walk(t):
                    if isinstance(name, ast.Name) and (name.id in functions or hasattr(builtins, name.id)):
                        raise _err(f"the '__main__' block may not bind '{name.id}': verified code "
                                   "calls that name", n)

    def check_names(self) -> None:
        for n in ast.walk(self.tree):
            name = (n.id if isinstance(n, ast.Name) else n.arg if isinstance(n, ast.arg)
                    else n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None)
            if name is not None:
                reason = reserved_reason(name)
                if reason is not None:
                    raise _err(reason, n)

    def extract(self) -> dict[str, Any]:
        directives = self.directives()
        self.check_names()
        defs: list[ast.FunctionDef] = []
        guards: list[ast.If] = []
        for s in self.tree.body:
            if _is_docstring(s):
                continue
            if _is_main_guard(s):
                assert isinstance(s, ast.If)
                guards.append(s)
                continue
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                self.check_import(s, bool(defs))
                continue
            if isinstance(s, ast.AsyncFunctionDef):
                raise _err("async functions are not supported", s)
            if isinstance(s, ast.FunctionDef):
                if s.decorator_list:
                    raise _err("decorators are not supported", s.decorator_list[0])
                if s.type_params:
                    raise _err("generic functions are not supported", s)
                if any(d.name == s.name for d in defs):
                    raise _err(f"'{s.name}' is defined twice", s)
                defs.append(s)
                continue
            if isinstance(s, ast.ClassDef):
                raise _err("classes are not supported yet", s)
            raise _err("top-level statement is not supported (only functions, imports and docstrings)", s)

        for g in guards:
            self.check_main_guard(g, {d.name for d in defs})
        heads = {fn.name: self.head_annotations(fn) for fn in defs}
        self.check_head_misplacements(defs)
        sigs = {fn.name: self.signature(fn) for fn in defs}
        verify_mode = any(a.keyword == "verify" for hs in heads.values() for a in hs)
        for name, sig in sigs.items():
            sig.verified = not verify_mode or any(a.keyword == "verify" for a in heads[name])

        functions = []
        for fn in defs:
            if sigs[fn.name].verified:
                functions.append(self.function(fn, sigs[fn.name], heads[fn.name], sigs, verify_mode))
            else:
                # Not verified: its annotations are not used, but must still be well placed.
                for a in self.annotations:
                    if fn.lineno < a.line <= self.extent_end(fn) and a.col > fn.col_offset:
                        a.claimed = True
        self.check_unclaimed(defs)

        return {
            "format": RAW_IR_FORMAT,
            "version": RAW_IR_VERSION,
            "producer": {"name": "lemmascript-python", "version": __version__},
            "source": self.filename,
            "identifiers": sorted(self.identifiers()),
            "directives": directives,
            "module": {"file": self.filename, "typeDecls": [], "externs": [], "constants": [],
                       "functions": functions, "classes": []},
        }

    def extent_end(self, fn: ast.FunctionDef) -> int:
        """A function's extent runs to the next top-level statement, so a comment
        indented under its last line still counts as inside it."""
        starts = sorted(s.lineno for s in self.tree.body)
        return next((l - 1 for l in starts if l > fn.lineno), len(self.source.splitlines()) + 1)

    def check_head_misplacements(self, defs: list[ast.FunctionDef]) -> None:
        """Function-head annotations (requires, ensures, ...) found later in a body."""
        for a in self.annotations:
            if a.claimed or a.keyword not in FUNCTION_KEYWORDS or a.keyword in LOOP_KEYWORDS:
                continue
            if any(fn.lineno < a.line <= self.extent_end(fn) and a.col > fn.col_offset for fn in defs):
                raise UnsupportedError(f"'#@ {a.keyword}' must come before the first statement of the function body",
                                       a.line, a.col)

    def check_unclaimed(self, defs: list[ast.FunctionDef]) -> None:
        for a in self.annotations:
            if a.claimed:
                continue
            inside = next((fn for fn in defs
                           if fn.lineno < a.line <= self.extent_end(fn) and a.col > fn.col_offset), None)
            if inside is not None and a.keyword in LOOP_KEYWORDS:
                raise UnsupportedError(f"'#@ {a.keyword}' belongs at the start of a loop body, before its first "
                                       "statement", a.line, a.col)
            if inside is not None and a.keyword in STATEMENT_KEYWORDS:
                raise UnsupportedError(f"'#@ {a.keyword}' must be on its own line, indented like the statement "
                                       "it precedes", a.line, a.col)
            if inside is not None and a.keyword in FUNCTION_KEYWORDS:
                raise UnsupportedError(f"'#@ {a.keyword}' must come before the first statement of the function body",
                                       a.line, a.col)
            if inside is None and a.keyword in FUNCTION_KEYWORDS:
                if a.keyword == "verify":
                    raise UnsupportedError("'#@ verify' belongs inside a function body, before its first statement",
                                           a.line, a.col)
                raise UnsupportedError(f"stray '#@' annotation: '#@ {a.keyword}' is not attached to a function",
                                       a.line, a.col)
            if a.keyword in FILE_KEYWORDS:
                raise UnsupportedError(f"'#@ {a.keyword}' must come before the first statement of the file",
                                       a.line, a.col)
            raise UnsupportedError(f"'#@ {a.keyword}' is not supported here yet", a.line, a.col)

    def function(self, fn: ast.FunctionDef, sig: Signature, head: list[Annotation],
                 sigs: dict[str, Signature], verify_mode: bool) -> dict[str, Any]:
        if any(p == "result" for p, _ in sig.params):
            raise _err("a parameter named 'result' clashes with the return value in '#@ ensures'; rename it", fn)
        low = FunctionLowerer(sig, sigs, verify_mode, binding_counts(fn), self.annotations, self.identifiers(),
                              local_names=_bound_names(fn) | {p for p, _ in sig.params},
                              source_lines=self.source.splitlines())
        requires, ensures, contract = [], [], []
        decreases = None
        pure = False
        type_annotations = []
        for a in head:
            if a.keyword == "requires":
                requires.append(low.spec(a.text, a.line))
            elif a.keyword == "ensures":
                ensures.append(low.spec(a.text, a.line, allow_result=True))
            elif a.keyword == "decreases":
                decreases = low.spec_value(a.text, a.line)
            elif a.keyword == "pure":
                if any(isinstance(n, (ast.Raise, ast.Assert)) for n in ast.walk(fn)):
                    raise UnsupportedError("'#@ pure' is not supported on a function that can raise", a.line, a.col)
                pure = True
            elif a.keyword == "contract":
                contract.append(a.text)
            elif a.keyword == "type":
                parts = a.text.split()
                if len(parts) != 2 or parts[1] not in ("nat", "int"):
                    raise UnsupportedError("'#@ type' is written '#@ type <name> nat'", a.line, a.col)
                if parts[0] == "result":
                    raise UnsupportedError("'#@ type result' is not supported; state the bound in '#@ ensures'",
                                           a.line, a.col)
                if parts[0] not in {p for p, _ in sig.params} | set(_bound_names(fn)):
                    raise UnsupportedError(f"'#@ type {parts[0]}': no parameter or local named '{parts[0]}'",
                                           a.line, a.col)
                type_annotations.append({"name": parts[0], "type": parts[1]})
        body_stmts = [s for s in fn.body if not _is_docstring(s)] if fn.body else []
        body = low.body(body_stmts, fn.lineno, self.extent_end(fn) + 1)
        low.check_aliasing()
        if sig.returns.kind != "none" and not always_exits(body_stmts):
            raise _err(f"'{fn.name}' may reach the end without returning (Python would return None)", fn)
        return {
            "name": fn.name,
            "exported": not fn.name.startswith("_"),
            "typeParams": [],
            "params": [{"name": n, "tsType": t.raw} for n, t in sig.params],
            "tsParams": [{"kind": "simple", "binds": [n]} for n, _ in sig.params],
            "returnType": sig.returns.raw,
            "requires": requires,
            "ensures": ensures,
            "contract": contract,
            "decreases": decreases,
            "pure": pure,
            "autohavoc": False,
            "typeAnnotations": type_annotations,
            "body": body,
            "line": fn.lineno,
        }

    def identifiers(self) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
        for a in self.annotations:
            names.update(w for w in a.text.replace("(", " ").replace(")", " ").split() if w.isidentifier())
        return names


def extract_source(source: str, filename: str = "<string>") -> dict[str, Any]:
    try:
        return _Module(source, filename).extract()
    except UnsupportedError as e:
        e.file = e.file or filename
        raise


def extract_file(path: str) -> dict[str, Any]:
    p = Path(path)
    return extract_source(p.read_text(encoding="utf-8"), p.name)


def _bound_names(fn: ast.FunctionDef) -> set[str]:
    """Every name a function binds anywhere (Python makes them local to the whole function)."""
    out: set[str] = set()
    for n in ast.walk(fn):
        targets: list[ast.expr] = []
        if isinstance(n, ast.Assign):
            targets = list(n.targets)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
            targets = [n.target]
        for t in targets:
            out.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store))
    return out
