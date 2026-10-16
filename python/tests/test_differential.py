"""Differential soundness harness: Python vs. the compiled Dafny model.

The harness must pass the verified examples and, just as importantly, must be
able to fail: each `difftest_fixtures/` model below diverges from its Python
source in one specific way, and the matching test asserts the harness says so.
"""
from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

import pytest
from hypothesis import settings

from conftest import LSC, has_dafny
from lemmascript_python import contracts
from lemmascript_python.difftest import model, runner
from lemmascript_python.difftest.runner import Config, Verdict

pytestmark = [
    pytest.mark.dafny,
    pytest.mark.skipif(not has_dafny(), reason="dafny not on PATH"),
    pytest.mark.skipif(not Path(LSC[0]).exists(), reason="lsc not installed (run npm ci at the repository root)"),
]

HERE = Path(__file__).resolve().parent
EXAMPLES = sorted((HERE.parent / "examples").glob("*.py"))
GOLDEN = sorted((HERE / "golden").glob("*.py"))
FIXTURES = HERE / "difftest_fixtures"


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One translation cache for the whole module, kept out of the user's cache."""
    return tmp_path_factory.mktemp("difftest-cache")


def check(path: Path, cache: Path, **kw: object) -> dict[str, Verdict]:
    cfg = Config(cache_dir=cache, **kw)  # type: ignore[arg-type]
    return {v.function: v for v in runner.check_file(path, cfg)}


def test_profile_is_deterministic() -> None:
    prof = settings.get_profile(runner.HYPOTHESIS_PROFILE)
    assert prof.derandomize is True
    assert prof.database is None


@pytest.mark.parametrize("path", EXAMPLES + GOLDEN, ids=[f"{p.parent.name}/{p.stem}" for p in EXAMPLES + GOLDEN])
def test_examples_pass(path: Path, cache: Path) -> None:
    verdicts = check(path, cache)
    assert verdicts, f"no functions checked in {path.name}"
    for v in verdicts.values():
        assert v.status == "PASSED", v.describe()
        # Small (all-bool or no-argument) spaces are enumerated exhaustively.
        assert v.accepted >= Config().min_accepted or "exhaustive" in v.detail, v.describe()


def test_example_names_cover_every_function(cache: Path) -> None:
    verdicts = check(HERE.parent / "examples" / "sign.py", cache)
    assert set(verdicts) == {"sign", "abs_val", "max3", "is_even"}


# --- the guard must be able to fail ----------------------------------------


def test_bad_mod_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "bad_mod.py", cache)["py_mod"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None
    a, b = v.counterexample["a"], v.counterexample["b"]
    assert b < 0 and a % b != 0  # floored and Euclidean `%` only differ here
    assert v.python_result == a % b


def test_bad_div_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "bad_div.py", cache)["py_div"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None
    a, b = v.counterexample["a"], v.counterexample["b"]
    assert a % b != 0 and b < 0


def test_precondition_disagreement_is_an_issue(cache: Path) -> None:
    # The fixture's own model reads `x > 0` faithfully; inject a Dafny-side
    # reading that also admits 0 and check the agreement check notices.
    v = check(FIXTURES / "bad_pre.py", cache,
              dafny_pre_overrides={"positive_id": lambda x: x >= 0})["positive_id"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "precondition disagreement"
    assert v.counterexample == {"x": 0}


def test_bad_pre_fixture_passes_without_injection(cache: Path) -> None:
    v = check(FIXTURES / "bad_pre.py", cache)["positive_id"]
    assert v.status == "PASSED", v.describe()


def test_ensures_false_on_python_result_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "post_violation.py", cache)["ident"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "ensures false on the Python result"
    assert v.counterexample == {"x": 0}
    assert "result > x" in v.detail


def test_python_raising_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "raises.py", cache)["checked_half"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "python raised"
    assert v.counterexample == {"x": 101}
    assert "ValueError" in v.detail


def test_injected_model_divergence_is_an_issue(cache: Path) -> None:
    v = check(HERE.parent / "examples" / "clamp.py", cache,
              model_overrides={"clamp": lambda x, lo, hi: x})["clamp"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"


def test_result_type_mismatch_is_an_issue(cache: Path) -> None:
    v = check(HERE.parent / "examples" / "sign.py", cache,
              model_overrides={"is_even": lambda n: int(n % 2 == 0)})["is_even"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"


# --- not checked -------------------------------------------------------------


def test_narrow_precondition_is_not_checked(cache: Path) -> None:
    v = check(FIXTURES / "narrow_pre.py", cache)["needle"]
    assert v.status == "NOT CHECKED", v.describe()
    assert v.accepted < Config().min_accepted
    assert "accepted" in v.detail


def test_min_accepted_is_configurable(cache: Path) -> None:
    v = check(FIXTURES / "bad_pre.py", cache, max_examples=40, min_accepted=1000)["positive_id"]
    assert v.status == "NOT CHECKED", v.describe()


def test_slow_inputs_are_skipped_not_passed(cache: Path) -> None:
    def slow(x: int, lo: int, hi: int) -> int:
        time.sleep(1)
        return x

    v = check(HERE.parent / "examples" / "clamp.py", cache, max_examples=5, min_accepted=1,
              call_timeout=0.05, model_overrides={"clamp": slow})["clamp"]
    assert v.status == "NOT CHECKED", v.describe()
    assert v.accepted == 0 and "skipped" in v.detail


def test_unevaluable_contract_is_not_checked(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The M1 frontend rejects quantifiers outright, so mark a real contract
    # unevaluable the way `contracts.load` would for an unbounded quantifier.
    real = contracts.load
    monkeypatch.setattr(contracts, "load",
                        lambda p: {n: dataclasses.replace(c, unevaluable=True) for n, c in real(p).items()})
    v = check(HERE.parent / "examples" / "clamp.py", cache)["clamp"]
    assert v.status == "NOT CHECKED", v.describe()
    assert "quantifier" in v.detail


def test_unsupported_source_is_not_checked(tmp_path: Path, cache: Path) -> None:
    src = tmp_path / "quant.py"
    src.write_text(
        "def q(x: int) -> int:\n"
        "    #@ ensures forall(k, k * 0 == 0)\n"
        "    return x\n"
    )
    [v] = runner.check_file(src, Config(cache_dir=cache))
    assert v.status == "NOT CHECKED", v.describe()
    assert "cannot read contracts" in v.detail


def test_verify_mode_module(tmp_path: Path, cache: Path) -> None:
    src = tmp_path / "opt_in.py"
    src.write_text(
        "def helper(x: int) -> int:\n"
        "    return x * x\n"
        "\n"
        "\n"
        "def nonneg_sq(x: int) -> int:\n"
        "    #@ verify\n"
        "    #@ requires x != 0\n"
        "    #@ ensures result > 0\n"
        "    return x * x\n"
    )
    verdicts = check(src, cache)
    assert set(verdicts) == {"nonneg_sq"}
    assert verdicts["nonneg_sq"].status == "PASSED", verdicts["nonneg_sq"].describe()


def test_requires_calling_a_module_function(tmp_path: Path, cache: Path) -> None:
    src = tmp_path / "callpre.py"
    src.write_text(
        "def is_pos(x: int) -> bool:\n"
        "    return x > 0\n"
        "\n"
        "\n"
        "def half_pos(x: int) -> int:\n"
        "    #@ requires is_pos(x)\n"
        "    #@ ensures 0 <= result < x\n"
        "    return x // 2\n"
    )
    verdicts = check(src, cache)
    assert verdicts["half_pos"].status == "PASSED", verdicts["half_pos"].describe()
    assert verdicts["is_pos"].status == "PASSED", verdicts["is_pos"].describe()


def test_no_artifacts_generates_model_without_touching_source_dir(tmp_path: Path, cache: Path) -> None:
    src = tmp_path / "clamp.py"
    shutil.copy(HERE.parent / "examples" / "clamp.py", src)
    v = check(src, cache)["clamp"]
    assert v.status == "PASSED", v.describe()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clamp.py"]


# --- caching -----------------------------------------------------------------


def test_second_build_hits_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    real = model.dafny_translate

    def counting(files: list[Path], out: Path, std: bool) -> None:
        calls.append([f.name for f in files])
        real(files, out, std)

    monkeypatch.setattr(model, "dafny_translate", counting)
    path = HERE.parent / "examples" / "clamp.py"
    first = model.build(path, cache_dir=tmp_path)
    assert len(calls) == 1
    second = model.build(path, cache_dir=tmp_path)
    assert len(calls) == 1, "second build re-ran dafny"
    assert first.key == second.key
    assert second.model_fn("clamp")(5, 0, 3) == 3


def test_cache_key_tracks_model_content(tmp_path: Path) -> None:
    src = tmp_path / "clamp.py"
    shutil.copy(HERE.parent / "examples" / "clamp.py", src)
    a = model.build(src, cache_dir=tmp_path / "c")
    proof = (HERE.parent / "examples" / "clamp.dfy").read_text() + "\n// a proof addition\n"
    (tmp_path / "clamp.dfy").write_text(proof)
    b = model.build(src, cache_dir=tmp_path / "c")
    assert a.key != b.key
    assert b.model_source == tmp_path / "clamp.dfy"


def test_cli_exit_codes(cache: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from lemmascript_python.difftest.__main__ import main

    base = ["--cache-dir", str(cache), "--max-examples", "100", "--min-accepted", "20"]
    assert main([str(HERE.parent / "examples" / "clamp.py"), *base]) == 0
    assert "PASSED" in capsys.readouterr().out
    assert main([str(FIXTURES / "bad_mod.py"), *base]) == 1
    out = capsys.readouterr().out
    assert "ISSUE" in out and "py_mod" in out
    assert main([str(FIXTURES / "narrow_pre.py"), *base]) == 2
    assert "NOT CHECKED" in capsys.readouterr().out
