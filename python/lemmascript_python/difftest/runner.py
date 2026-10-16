"""Per-function differential checks and verdicts.

For each input the checks are, in order:

1. **precondition agreement** — the Python reading of ``requires`` (evaluated
   on the real values, with the module's globals in scope) must equal the
   Dafny reading (``lsc_pre_<f>`` compiled from the generated Dafny);
2. if the precondition holds: the original Python function must not raise,
   the compiled model must return the same value (and kind), and every
   ``ensures`` must hold on the *Python* result.

A function PASSES only if every check holds on every input and at least
``min_accepted`` distinct inputs satisfied the precondition; otherwise it is
NOT CHECKED. Any failed check is an ISSUE with Hypothesis' shrunk input.
"""
from __future__ import annotations

import hashlib
import importlib.util
import signal
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import Any, Literal

from hypothesis import HealthCheck, Phase, given, settings

from lemmascript_python import contracts as contracts_mod
from lemmascript_python.contracts import FnContract
from lemmascript_python.difftest import model as model_mod
from lemmascript_python.difftest import strategies
from lemmascript_python.errors import UnsupportedError

HYPOTHESIS_PROFILE = "lemmascript-difftest"
settings.register_profile(
    HYPOTHESIS_PROFILE,
    derandomize=True,
    database=None,
    deadline=None,
    report_multiple_bugs=False,
    suppress_health_check=list(HealthCheck),
    phases=[Phase.explicit, Phase.generate, Phase.shrink],
    print_blob=False,
)

Status = Literal["PASSED", "ISSUE", "NOT CHECKED"]
Fn = Callable[..., Any]


@dataclass
class Config:
    max_examples: int = 300
    min_accepted: int = 50
    cache_dir: Path | None = None
    #: Wall-clock limit (seconds) for one input's evaluation; None disables it.
    #: An input that runs out of time is skipped, not counted as accepted.
    call_timeout: float | None = 5.0
    #: Replace the compiled Dafny-side precondition of a function (testing).
    dafny_pre_overrides: Mapping[str, Fn] = field(default_factory=dict)
    #: Replace the compiled model of a function (testing).
    model_overrides: Mapping[str, Fn] = field(default_factory=dict)


@dataclass
class Verdict:
    function: str
    status: Status
    #: For ISSUE, the kind of finding (e.g. "result mismatch"); else "".
    kind: str = ""
    detail: str = ""
    counterexample: dict[str, Any] | None = None
    python_result: Any = None
    model_result: Any = None
    accepted: int = 0
    tried: int = 0

    def describe(self) -> str:
        s = f"{self.function}: {self.status}"
        if self.kind:
            s += f" [{self.kind}]"
        if self.counterexample is not None:
            s += " at " + ", ".join(f"{k}={v!r}" for k, v in self.counterexample.items())
        if self.detail:
            s += f" — {self.detail}"
        return s


@dataclass
class Target:
    """Everything one function's check needs, injected so tests can swap parts."""
    contract: FnContract
    python_fn: Fn
    model_fn: Fn
    dafny_pre: Fn
    globals: dict[str, Any]


class _Finding(Exception):
    def __init__(self, kind: str, xs: tuple[Any, ...], detail: str,
                 python_result: Any = None, model_result: Any = None) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind, self.xs, self.detail = kind, xs, detail
        self.python_result, self.model_result = python_result, model_result


class _Timeout(BaseException):
    """Raised by SIGALRM; a BaseException so user code's `except Exception` can't eat it."""


@contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    if (seconds is None or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def on_alarm(signum: int, frame: FrameType | None) -> None:
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _same(kind: str, a: Any, b: Any) -> bool:
    if kind == "bool":
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if kind == "int":
        return (isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int)
                and not isinstance(b, bool) and a == b)
    return type(a) is type(b) and bool(a == b)


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


class _Checker:
    def __init__(self, t: Target) -> None:
        self.t = t
        c = t.contract
        self.names = [n for n, _ in c.params]
        self.requires: list[tuple[str, CodeType]] = [(r, compile(r, f"<requires of {c.name}>", "eval"))
                                                     for r in c.requires]
        self.ensures: list[tuple[str, CodeType]] = [(e, compile(e, f"<ensures of {c.name}>", "eval"))
                                                    for e in c.ensures]
        self.accepted: set[tuple[Any, ...]] = set()
        self.tried: set[tuple[Any, ...]] = set()
        self.timeouts = 0

    def run(self, xs: tuple[Any, ...], timeout: float | None) -> None:
        try:
            with _time_limit(timeout):
                ok = self.evaluate(xs)
        except _Timeout:
            self.timeouts += 1
            return
        self.tried.add(xs)
        if ok:
            self.accepted.add(xs)

    def evaluate(self, xs: tuple[Any, ...]) -> bool:
        """True if the precondition held and every check passed; raises _Finding."""
        t, c = self.t, self.t.contract
        env = dict(t.globals)
        env.update(zip(self.names, xs))
        py_pre = True
        for text, code in self.requires:
            try:
                held = bool(eval(code, env))  # noqa: S307 - the module's own contract
            except Exception as e:
                raise _Finding("requires raised", xs, f"`{text}` raised {_exc(e)}") from None
            if not held:
                py_pre = False
                break
        try:
            d_pre = t.dafny_pre(*xs)
        except Exception as e:
            raise _Finding("model precondition raised", xs, _exc(e)) from None
        if not isinstance(d_pre, bool) or d_pre != py_pre:
            raise _Finding("precondition disagreement", xs,
                           f"Python reads requires as {py_pre}, the Dafny model as {d_pre!r}")
        if not py_pre:
            return False
        try:
            r = t.python_fn(*xs)
        except Exception as e:
            raise _Finding("python raised", xs, f"{c.name} raised {_exc(e)}") from None
        try:
            m = t.model_fn(*xs)
        except Exception as e:
            raise _Finding("model raised", xs, _exc(e), python_result=r) from None
        if not _same(c.returns, r, m):
            raise _Finding("result mismatch", xs,
                           f"Python returned {r!r} ({type(r).__name__}), "
                           f"the model {m!r} ({type(m).__name__})", r, m)
        env["result"] = r
        for text, code in self.ensures:
            try:
                held = bool(eval(code, env))  # noqa: S307
            except Exception as e:
                raise _Finding("ensures raised", xs, f"`{text}` raised {_exc(e)}", r, m) from None
            if not held:
                raise _Finding("ensures false on the Python result", xs,
                               f"ensures `{text}` is false (result = {r!r})", r, m)
        return True


def check_function(t: Target, cfg: Config) -> Verdict:
    c = t.contract
    kinds = [k for _, k in c.params]
    ch = _Checker(t)
    space = strategies.finite_space(kinds)
    need = cfg.min_accepted
    try:
        if space is not None:
            for xs in space:
                ch.run(xs, cfg.call_timeout)
            need = 1  # every input was tried
        else:
            def body(xs: tuple[Any, ...]) -> None:
                ch.run(xs, cfg.call_timeout)

            # Pinned edge tuples run after the search, not as @example: Hypothesis
            # never shrinks an explicit example, and a failure the search also
            # finds is better reported shrunk.
            test: Any = given(strategies.for_params(kinds))(body)
            test = settings(settings.get_profile(HYPOTHESIS_PROFILE), max_examples=cfg.max_examples)(test)
            test()
            for xs in strategies.pinned(kinds):
                ch.run(xs, cfg.call_timeout)
    except _Finding as f:
        return Verdict(c.name, "ISSUE", f.kind, f.detail, dict(zip(ch.names, f.xs)),
                       f.python_result, f.model_result, len(ch.accepted), len(ch.tried))
    except Exception as e:  # a harness or Hypothesis failure: never a pass
        return Verdict(c.name, "NOT CHECKED", "", f"harness error: {_exc(e)}",
                       accepted=len(ch.accepted), tried=len(ch.tried))
    acc, tried = len(ch.accepted), len(ch.tried)
    note = f"; {ch.timeouts} input(s) skipped after {cfg.call_timeout}s" if ch.timeouts else ""
    if acc < need:
        return Verdict(c.name, "NOT CHECKED", "",
                       f"only {acc} of {tried} distinct inputs accepted by the precondition "
                       f"(need {need}){note}", accepted=acc, tried=tried)
    how = "exhaustive" if space is not None else "sampled"
    return Verdict(c.name, "PASSED", "", f"{acc} of {tried} distinct inputs accepted ({how}){note}",
                   accepted=acc, tried=tried)


def _load_python(path: Path) -> ModuleType:
    name = "lsc_difftest_src_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # leave the source tree alone
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


def _not_checked(cons: Mapping[str, FnContract] | Sequence[str], reason: str) -> list[Verdict]:
    return [Verdict(n, "NOT CHECKED", "", reason) for n in cons]


def check_file(path: str | Path, cfg: Config | None = None) -> list[Verdict]:
    cfg = cfg or Config()
    src = Path(path).resolve()
    try:
        cons = contracts_mod.load(src)
    except (UnsupportedError, SyntaxError, OSError) as e:
        return [Verdict(src.name, "NOT CHECKED", "", f"cannot read contracts: {e}")]
    if not cons:
        return []
    try:
        built = model_mod.build(src, cache_dir=cfg.cache_dir, cons=cons)
    except model_mod.BuildError as e:
        return _not_checked(cons, f"model build failed: {e}")
    try:
        pymod = _load_python(src)
    except Exception as e:
        return _not_checked(cons, f"importing the Python module failed: {_exc(e)}")

    out: list[Verdict] = []
    for name, c in cons.items():
        if c.unevaluable:
            out.append(Verdict(name, "NOT CHECKED", "", "contract uses a quantifier with no executable reading"))
            continue
        bad = [k for _, k in c.params if k not in strategies.SUPPORTED_KINDS]
        if bad or c.returns not in strategies.SUPPORTED_KINDS:
            out.append(Verdict(name, "NOT CHECKED", "",
                               f"unsupported kinds (params {bad}, returns {c.returns!r})"))
            continue
        py_fn = getattr(pymod, name, None)
        if not callable(py_fn):
            out.append(Verdict(name, "NOT CHECKED", "", f"the Python module has no function {name}"))
            continue
        try:
            model_fn = cfg.model_overrides.get(name) or built.model_fn(name)
            pre_fn = cfg.dafny_pre_overrides.get(name) or built.pre_fn(name)
        except LookupError as e:
            out.append(Verdict(name, "NOT CHECKED", "", str(e)))
            continue
        out.append(check_function(Target(c, py_fn, model_fn, pre_fn, vars(pymod)), cfg))
    return out
