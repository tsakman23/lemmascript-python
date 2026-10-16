"""Module-level extraction: Python source → Raw IR envelope for ``lsc``."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

from lemmascript_python import __version__
from lemmascript_python.annotations import (
    FILE_KEYWORDS,
    FUNCTION_KEYWORDS,
    KNOWN_KEYWORDS,
    Annotation,
    collect,
)
from lemmascript_python.errors import UnsupportedError
from lemmascript_python.lower import FunctionLowerer, Signature, always_exits, binding_counts
from lemmascript_python.pytypes import from_annotation

RAW_IR_FORMAT = "lemmascript-raw-ir"
RAW_IR_VERSION = 1

#: Modules whose imports carry no runtime behaviour the model must know about.
ALLOWED_IMPORTS = frozenset({"__future__", "typing", "typing_extensions"})

#: File-overridable lsc options (tools/src/config.ts), with their allowed values.
FILE_OPTIONS: dict[str, tuple[str, ...]] = {
    "safe-slice": ("true", "false"),
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
                if a.text not in ("dafny", "lean"):
                    raise UnsupportedError(f"'#@ backend {a.text}': expected dafny or lean", a.line, a.col)
                backend = a.text
            else:
                key, _, value = a.text.partition(" ")
                value = value.strip()
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
        return {"backend": backend, "options": options, "leanModule": None}

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
            if fn.lineno < a.line < stop and a.line not in doc_lines and a.col > fn.col_offset:
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

    def extract(self) -> dict[str, Any]:
        directives = self.directives()
        defs: list[ast.FunctionDef] = []
        for s in self.tree.body:
            if _is_docstring(s) or _is_main_guard(s):
                continue
            if isinstance(s, ast.Import):
                for alias in s.names:
                    if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                        raise _err(f"import of '{alias.name}' is not supported", s)
                continue
            if isinstance(s, ast.ImportFrom):
                if (s.module or "").split(".")[0] not in ALLOWED_IMPORTS or s.level:
                    raise _err(f"import of '{'.' * s.level}{s.module or ''}' is not supported", s)
                continue
            if isinstance(s, ast.AsyncFunctionDef):
                raise _err("async functions are not supported", s)
            if isinstance(s, ast.FunctionDef):
                if s.decorator_list:
                    raise _err("decorators are not supported", s.decorator_list[0])
                if s.type_params:
                    raise _err("generic functions are not supported", s)
                defs.append(s)
                continue
            if isinstance(s, ast.ClassDef):
                raise _err("classes are not supported yet", s)
            raise _err("top-level statement is not supported (only functions, imports and docstrings)", s)

        heads = {fn.name: self.head_annotations(fn) for fn in defs}
        self.check_unclaimed(defs)
        sigs = {fn.name: self.signature(fn) for fn in defs}
        verify_mode = any(a.keyword == "verify" for hs in heads.values() for a in hs)
        for name, sig in sigs.items():
            sig.verified = not verify_mode or any(a.keyword == "verify" for a in heads[name])

        functions = []
        for fn in defs:
            if sigs[fn.name].verified:
                functions.append(self.function(fn, sigs[fn.name], heads[fn.name], sigs, verify_mode))

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

    def check_unclaimed(self, defs: list[ast.FunctionDef]) -> None:
        # A function's extent runs to the next top-level statement, so a
        # comment indented under its last line still counts as inside it.
        starts = sorted(s.lineno for s in self.tree.body)
        def extent_end(fn: ast.FunctionDef) -> int:
            return next((l - 1 for l in starts if l > fn.lineno), 10 ** 9)
        for a in self.annotations:
            if a.claimed:
                continue
            inside = next((fn for fn in defs
                           if fn.lineno < a.line <= extent_end(fn) and a.col > fn.col_offset), None)
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
        low = FunctionLowerer(sig, sigs, verify_mode, binding_counts(fn))
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
                pure = True
            elif a.keyword == "contract":
                contract.append(a.text)
            elif a.keyword == "type":
                parts = a.text.split()
                if len(parts) != 2 or parts[1] not in ("nat", "int"):
                    raise UnsupportedError("'#@ type' is written '#@ type <name> nat'", a.line, a.col)
                type_annotations.append({"name": parts[0], "type": parts[1]})
        body_stmts = [s for s in fn.body if not _is_docstring(s)] if fn.body else []
        body = low.block(body_stmts)
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
