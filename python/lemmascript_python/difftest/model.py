"""Build, cache and import the Dafny model of a Python module.

Two Dafny programs are compiled to Python with ``dafny translate py
--no-verify`` and imported side by side:

* **Model** — the committed proof ``foo.dfy`` (the verified artifact), or
  ``foo.dfy.gen`` when no proof exists, or a fresh ``lsc gen`` of the source
  (in a temporary directory) when neither does. Artifacts are looked up in the
  directory ``lsc config`` reports, so a ``proof-dir`` is honoured.
* **Pre** — ``lsc gen`` of a synthetic module: the original source plus one
  ``lsc_pre_<f>`` function per checked function whose body is the conjunction
  of ``f``'s Python ``requires``. Calling it runs the *Dafny* reading of the
  precondition, so the harness can compare it with the Python reading.

Every untranslated Dafny program is module ``module_``, so each program is
wrapped in a module named after the content hash (``LscModel<key>``,
``LscPre<key>``) before translation; distinct builds then import side by side
in one process. Translations are cached under ``$XDG_CACHE_HOME`` (or
``~/.cache``) ``/lemmascript-python/difftest/<key>``, keyed by both program
texts and the Dafny version.
"""
from __future__ import annotations

import builtins
import functools
import hashlib
import importlib
import json
import keyword
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from lemmascript_python import contracts as contracts_mod
from lemmascript_python.contracts import FnContract
from lemmascript_python.difftest.strategies import SUPPORTED_KINDS

#: Bump when the wrapping or the synthetic module changes shape.
FORMAT_VERSION = "1"
PRE_PREFIX = "lsc_pre_"
_VERIFY = re.compile(r"^\s*#@\s*verify\b")

REPO = Path(__file__).resolve().parents[3]
LSC_TIMEOUT = 300
DAFNY_TIMEOUT = 600


class BuildError(Exception):
    """The model could not be built; every function is then NOT CHECKED."""


# --- external tools ---------------------------------------------------------


def lsc_command() -> list[str]:
    if env := os.environ.get("LSC_CMD"):
        return shlex.split(env)
    tsx, src = REPO / "node_modules" / ".bin" / "tsx", REPO / "tools" / "src" / "lsc.ts"
    if tsx.exists() and src.exists():
        return [str(tsx), str(src)]
    dist = REPO / "tools" / "dist" / "lsc.js"
    if dist.exists() and (node := shutil.which("node")):
        return [node, str(dist)]
    raise BuildError("lsc not found (run npm ci at the repository root, or set LSC_CMD)")


def run_lsc(args: list[str], cwd: Path) -> str:
    cmd = [*lsc_command(), *args]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=LSC_TIMEOUT,
                              env={**os.environ, "LSC_RUN_LOG": "false"})
    except (OSError, subprocess.TimeoutExpired) as e:
        raise BuildError(f"lsc {args[0]} failed: {e}") from e
    if proc.returncode != 0:
        raise BuildError(f"lsc {' '.join(args)} failed:\n{(proc.stderr or proc.stdout).strip()}")
    return proc.stdout


@functools.cache
def dafny_version() -> str:
    if shutil.which("dafny") is None:
        raise BuildError("dafny not on PATH")
    proc = subprocess.run(["dafny", "--version"], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise BuildError(f"dafny --version failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def dafny_translate(files: list[Path], out: Path, std: bool) -> None:
    """``dafny translate py --no-verify -o <out> files...`` → ``<out>-py/``.

    A module-level function so tests can count calls.
    """
    cmd = ["dafny", "translate", "py", "--no-verify", *(["--standard-libraries"] if std else []),
           "-o", str(out), *map(str, files)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DAFNY_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise BuildError(f"dafny translate failed: {e}") from e
    if proc.returncode != 0:
        raise BuildError(f"dafny translate failed:\n{(proc.stdout + proc.stderr).strip()}")


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "lemmascript-python" / "difftest"


# --- names ------------------------------------------------------------------


def python_candidates(dafny_name: str) -> list[str]:
    """Names Dafny's Python backend may give a Dafny identifier.

    It doubles underscores (``floor_div`` → ``floor__div``) and appends ``_``
    to names that clash with Python keywords and builtins (``sorted`` →
    ``sorted_``). The mangled spelling is tried first: it is unambiguous,
    because a doubled-underscore source name would itself be doubled again.
    """
    mangled = dafny_name.replace("_", "__")
    out = [mangled, mangled + "_", dafny_name, dafny_name + "_"]
    if keyword.iskeyword(mangled) or hasattr(builtins, mangled):
        out.insert(0, mangled + "_")
    return list(dict.fromkeys(out))


def resolve(cls: object, dafny_name: str) -> Callable[..., Any] | None:
    for cand in python_candidates(dafny_name):
        fn = getattr(cls, cand, None)
        if callable(fn):
            return fn  # type: ignore[no-any-return]
    return None


# --- the synthetic precondition module --------------------------------------


def pre_eligible(c: FnContract) -> bool:
    return not c.unevaluable and all(k in SUPPORTED_KINDS for _, k in c.params)


def pre_source(source: str, cons: Mapping[str, FnContract]) -> str:
    """The original source plus one ``lsc_pre_<f>`` per eligible function."""
    verify_mode = any(_VERIFY.match(line) for line in source.splitlines())
    parts = [source.rstrip("\n"), ""]
    for c in cons.values():
        if not pre_eligible(c):
            continue
        params = ", ".join(f"{n}: {k}" for n, k in c.params)
        body = " and ".join(f"({r})" for r in c.requires) or "True"
        parts.append("")
        parts.append(f"def {PRE_PREFIX}{c.name}({params}) -> bool:")
        if verify_mode:
            parts.append("    #@ verify")
        parts.append(f"    return {body}")
    return "\n".join(parts) + "\n"


def wrap(module: str, text: str) -> str:
    return f"module {module} {{\n{text}\n}}\n"


# --- build --------------------------------------------------------------------


@dataclass
class Built:
    key: str
    source: Path
    #: The committed ``.dfy`` / ``.dfy.gen`` used as the model; None if generated.
    model_source: Path | None
    out_dir: Path
    model_module: ModuleType
    pre_module: ModuleType
    #: Python source name → Dafny name, from ``lsc info --typed``.
    emitted: dict[str, str] = field(default_factory=dict)

    def dafny_name(self, py_name: str) -> str:
        return self.emitted.get(py_name, py_name)

    def _fn(self, module: ModuleType, py_name: str) -> Callable[..., Any]:
        fn = resolve(getattr(module, "default__", None), self.dafny_name(py_name))
        if fn is None:
            raise LookupError(f"no translated function for '{py_name}' "
                              f"(Dafny '{self.dafny_name(py_name)}') in {module.__name__}")
        return fn

    def model_fn(self, py_name: str) -> Callable[..., Any]:
        return self._fn(self.model_module, py_name)

    def pre_fn(self, py_name: str) -> Callable[..., Any]:
        return self._fn(self.pre_module, PRE_PREFIX + py_name)


def _artifact_model(config_json: str, src: Path) -> tuple[str, Path] | None:
    art = Path(json.loads(config_json).get("artifactDir") or src.parent)
    for cand in (art / f"{src.stem}.dfy", art / f"{src.stem}.dfy.gen"):
        if cand.exists():
            return cand.read_text(encoding="utf-8"), cand
    return None


def _generated_model(src: Path, work: Path) -> str:
    plain = work / "plain"
    plain.mkdir()
    shutil.copy(src, plain / src.name)
    run_lsc(["gen", "--backend=dafny", src.name], plain)
    return (plain / f"{src.stem}.dfy.gen").read_text(encoding="utf-8")


def _import(out_dir: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    saved = sys.modules.pop("module_", None)
    sys.path.insert(0, str(out_dir))
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise BuildError(f"importing translated module {name} failed: {e}") from e
    finally:
        sys.path.remove(str(out_dir))
        sys.modules.pop("module_", None)
        if saved is not None:
            sys.modules["module_"] = saved


def _translate_into(entry: Path, root: Path, version: str, programs: list[tuple[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
    try:
        files = []
        for i, (module, text) in enumerate(programs):
            f = stage / f"p{i}.dfy"
            f.write_text(wrap(module, text), encoding="utf-8")
            files.append(f)
        dafny_translate(files, stage / "out", any("Std." in text for _, text in programs))
        (stage / "ok").write_text(version + "\n", encoding="utf-8")
        try:
            if entry.exists():
                shutil.rmtree(entry)  # an incomplete earlier attempt
            stage.rename(entry)
        except OSError:
            if not (entry / "ok").exists():  # lost a race to a concurrent build: fine
                raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def build(path: str | Path, *, cache_dir: Path | None = None,
          cons: Mapping[str, FnContract] | None = None) -> Built:
    src = Path(path).resolve()
    source = src.read_text(encoding="utf-8")
    if cons is None:
        cons = contracts_mod.load(src)
    version = dafny_version()
    with tempfile.TemporaryDirectory(prefix="lsc-difftest-") as tmp:
        work = Path(tmp)
        pre_dir = work / "pre"
        pre_dir.mkdir()
        (pre_dir / src.name).write_text(pre_source(source, cons), encoding="utf-8")
        # Three independent lsc runs (~1s each, mostly start-up): run them together.
        with ThreadPoolExecutor(max_workers=3) as pool:
            config_f = pool.submit(run_lsc, ["config", str(src)], src.parent)
            gen_f = pool.submit(run_lsc, ["gen", "--backend=dafny", src.name], pre_dir)
            info_f = pool.submit(run_lsc, ["info", "--typed", src.name], pre_dir)
            config_json, _, info_json = config_f.result(), gen_f.result(), info_f.result()
        pre_text = (pre_dir / f"{src.stem}.dfy.gen").read_text(encoding="utf-8")
        found = _artifact_model(config_json, src)
        model_source: Path | None
        if found is not None:
            model_text, model_source = found
        else:
            model_text, model_source = _generated_model(src, work), None
    emitted: dict[str, str] = dict((json.loads(info_json).get("dafny") or {}).get("emittedNames") or {})

    h = hashlib.sha256()
    for part in (FORMAT_VERSION, version, model_text, pre_text):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    key = h.hexdigest()[:16]
    model_name, pre_name = f"LscModel{key}", f"LscPre{key}"

    root = cache_dir if cache_dir is not None else default_cache_dir()
    entry = root / key
    if not (entry / "ok").exists():
        _translate_into(entry, root, version, [(model_name, model_text), (pre_name, pre_text)])
    out_dir = entry / "out-py"
    return Built(key=key, source=src, model_source=model_source, out_dir=out_dir,
                 model_module=_import(out_dir, model_name), pre_module=_import(out_dir, pre_name),
                 emitted=emitted)
