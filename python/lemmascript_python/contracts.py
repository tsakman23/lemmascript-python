"""Executable views of a module's contracts, for the differential harness.

Each verified function's ``#@ requires`` / ``#@ ensures`` are re-parsed with
the same spec parser the extractor uses and rendered as plain Python
(``A ==> B`` → ``not A or B``), so they can be evaluated on real Python values
and compared against the Dafny model's reading of the same contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lemmascript_python import specs
from lemmascript_python.extract import _Module


@dataclass(frozen=True)
class FnContract:
    name: str
    #: (name, kind); kind is "int", "bool", or "list[<kind>]" (e.g. "list[list[int]]")
    params: list[tuple[str, str]]
    returns: str
    requires: list[str]  # executable Python expressions over the params
    ensures: list[str]  # executable Python expressions over the params and `result`
    #: True when a contract uses a construct with no executable reading
    #: (an unbounded quantifier); the harness then reports "not checked".
    unevaluable: bool = False
    #: The original spec texts with their source lines, for evaluators that
    #: need the parsed form (``specs.parse(text, line)``), e.g. to enumerate
    #: a bounded ``forall``. The implicit ``n >= 0`` of a nat parameter is
    #: included in ``requires_src`` too.
    requires_src: tuple[tuple[str, int], ...] = ()
    ensures_src: tuple[tuple[str, int], ...] = ()


def load(path: str | Path) -> dict[str, FnContract]:
    p = Path(path)
    return load_source(p.read_text(encoding="utf-8"), p.name)


def load_source(source: str, filename: str = "<string>") -> dict[str, FnContract]:
    mod = _Module(source, filename)
    env = mod.extract()  # validates the module exactly as `lsc` would see it
    verified = {f["name"] for f in env["module"]["functions"]}
    out: dict[str, FnContract] = {}
    import ast
    for node in mod.tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in verified:
            continue
        sig = mod.signature(node)
        heads = [a for a in mod.annotations if a.claimed and node.lineno < a.line <= (node.end_lineno or node.lineno)]
        req: list[str] = []
        ens: list[str] = []
        req_src: list[tuple[str, int]] = []
        ens_src: list[tuple[str, int]] = []
        unevaluable = False
        params = {n for n, _ in sig.params}
        for a in heads:
            # `#@ type p nat` on a parameter is a precondition in disguise: the
            # Dafny parameter is a `nat`, so the proof covers only p >= 0.
            words = a.text.split()
            if a.keyword == "type" and len(words) == 2 and words[1] == "nat" and words[0] in params:
                req.append(f"{words[0]} >= 0")
                req_src.append((f"{words[0]} >= 0", a.line))
        for a in heads:
            if a.keyword not in ("requires", "ensures"):
                continue
            tree = specs.parse(a.text, a.line)
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("forall", "exists")
                   for n in ast.walk(tree)):
                unevaluable = True
            (req if a.keyword == "requires" else ens).append(specs.to_python(tree))
            (req_src if a.keyword == "requires" else ens_src).append((a.text, a.line))
        out[node.name] = FnContract(node.name, [(n, str(t)) for n, t in sig.params], str(sig.returns),
                                    req, ens, unevaluable, tuple(req_src), tuple(ens_src))
    return out
