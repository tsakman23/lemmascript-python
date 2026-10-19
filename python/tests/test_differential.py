"""Differential soundness harness: Python vs. the compiled Dafny model.

The harness must pass the verified examples and, just as importantly, must be
able to fail: each `difftest_fixtures/` model below diverges from its Python
source in one specific way, and the matching test asserts the harness says so.
"""
from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
from hypothesis import given, settings

from conftest import LSC, has_dafny
from lemmascript_python import contracts, specs
from lemmascript_python.contracts import FnContract
from lemmascript_python.difftest import model, runner, specs_eval, strategies, values
from lemmascript_python.difftest.runner import Config, Target, Verdict

F = TypeVar("F", bound=Callable[..., Any])
_DAFNY_MARKS = [
    pytest.mark.dafny,
    pytest.mark.skipif(not has_dafny(), reason="dafny not on PATH"),
    pytest.mark.skipif(not Path(LSC[0]).exists(), reason="lsc not installed (run npm ci at the repository root)"),
]


def needs_dafny(f: F) -> F:
    """Tests that build a model (lsc + dafny); the rest are pure unit tests."""
    for m in reversed(_DAFNY_MARKS):
        f = m(f)
    return f


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


@needs_dafny
@pytest.mark.parametrize("path", EXAMPLES + GOLDEN, ids=[f"{p.parent.name}/{p.stem}" for p in EXAMPLES + GOLDEN])
def test_examples_pass(path: Path, cache: Path) -> None:
    verdicts = check(path, cache)
    assert verdicts, f"no functions checked in {path.name}"
    for v in verdicts.values():
        assert v.status == "PASSED", v.describe()
        # Small (all-bool or no-argument) spaces are enumerated exhaustively.
        assert v.accepted >= Config().min_accepted or "exhaustive" in v.detail, v.describe()


@needs_dafny
def test_example_names_cover_every_function(cache: Path) -> None:
    verdicts = check(HERE.parent / "examples" / "sign.py", cache)
    assert set(verdicts) == {"sign", "abs_val", "max3", "is_even"}


# --- the guard must be able to fail ----------------------------------------


@needs_dafny
def test_bad_mod_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "bad_mod.py", cache)["py_mod"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None
    a, b = v.counterexample["a"], v.counterexample["b"]
    assert b < 0 and a % b != 0  # floored and Euclidean `%` only differ here
    assert v.python_result == a % b


@needs_dafny
def test_bad_div_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "bad_div.py", cache)["py_div"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None
    a, b = v.counterexample["a"], v.counterexample["b"]
    assert a % b != 0 and b < 0


@needs_dafny
def test_precondition_disagreement_is_an_issue(cache: Path) -> None:
    # The fixture's own model reads `x > 0` faithfully; inject a Dafny-side
    # reading that also admits 0 and check the agreement check notices.
    v = check(FIXTURES / "bad_pre.py", cache,
              dafny_pre_overrides={"positive_id": lambda x: x >= 0})["positive_id"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "precondition disagreement"
    assert v.counterexample == {"x": 0}


@needs_dafny
def test_bad_pre_fixture_passes_without_injection(cache: Path) -> None:
    v = check(FIXTURES / "bad_pre.py", cache)["positive_id"]
    assert v.status == "PASSED", v.describe()


@needs_dafny
def test_ensures_false_on_python_result_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "post_violation.py", cache)["ident"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "ensures false on the Python result"
    assert v.counterexample == {"x": 0}
    assert "result > x" in v.detail


@needs_dafny
def test_python_raising_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "raises.py", cache)["checked_half"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "python raised"
    assert v.counterexample == {"x": 101}
    assert "ValueError" in v.detail


@needs_dafny
def test_injected_model_divergence_is_an_issue(cache: Path) -> None:
    v = check(HERE.parent / "examples" / "clamp.py", cache,
              model_overrides={"clamp": lambda x, lo, hi: x})["clamp"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"


@needs_dafny
def test_result_type_mismatch_is_an_issue(cache: Path) -> None:
    v = check(HERE.parent / "examples" / "sign.py", cache,
              model_overrides={"is_even": lambda n: int(n % 2 == 0)})["is_even"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"


# --- not checked -------------------------------------------------------------


@needs_dafny
def test_narrow_precondition_is_not_checked(cache: Path) -> None:
    v = check(FIXTURES / "narrow_pre.py", cache)["needle"]
    assert v.status == "NOT CHECKED", v.describe()
    assert v.accepted < Config().min_accepted
    assert "accepted" in v.detail


@needs_dafny
def test_min_accepted_is_configurable(cache: Path) -> None:
    v = check(FIXTURES / "bad_pre.py", cache, max_examples=40, min_accepted=1000)["positive_id"]
    assert v.status == "NOT CHECKED", v.describe()


@needs_dafny
def test_slow_inputs_are_skipped_not_passed(cache: Path) -> None:
    def slow(x: int, lo: int, hi: int) -> int:
        time.sleep(1)
        return x

    v = check(HERE.parent / "examples" / "clamp.py", cache, max_examples=5, min_accepted=1,
              call_timeout=0.05, model_overrides={"clamp": slow})["clamp"]
    assert v.status == "NOT CHECKED", v.describe()
    assert v.accepted == 0 and "skipped" in v.detail


def _patch_clamp(monkeypatch: pytest.MonkeyPatch, **extra: tuple[tuple[str, int], ...]) -> None:
    """Make `contracts.load` add spec clauses to clamp's contract (the M1
    frontend rejects quantifiers, so a real source cannot carry them yet)."""
    real = contracts.load

    def load(p: str | Path) -> dict[str, FnContract]:
        out = {}
        for n, c in real(p).items():
            for field, clauses in extra.items():
                c = dataclasses.replace(c, **{field: getattr(c, field) + clauses})
                plain = field.removesuffix("_src")
                c = dataclasses.replace(c, unevaluable=True, **{plain: getattr(c, plain) + [
                    specs.to_python(specs.parse(t, ln)) for t, ln in clauses]})
            out[n] = c
        return out

    monkeypatch.setattr(contracts, "load", load)


@needs_dafny
def test_unbounded_quantifier_is_not_checked(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clamp(monkeypatch, ensures_src=(("forall(k, k * 0 == 0)", 7),))
    v = check(HERE.parent / "examples" / "clamp.py", cache)["clamp"]
    assert v.status == "NOT CHECKED", v.describe()
    assert "unbounded quantifier" in v.detail


@needs_dafny
def test_bounded_quantifier_is_checked(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `unevaluable` is set (as contracts.load does for any quantifier), but the
    # evaluator can bound it, so the function is checked after all.
    _patch_clamp(monkeypatch, ensures_src=(("forall(k, 0 <= k < 3 ==> result + k >= lo + k)", 7),))
    v = check(HERE.parent / "examples" / "clamp.py", cache)["clamp"]
    assert v.status == "PASSED", v.describe()
    assert "precondition agreement not checked" not in v.detail


@needs_dafny
def test_quantified_requires_skips_precondition_agreement(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clamp(monkeypatch, requires_src=(("forall(k, 0 <= k < 3 ==> lo + k <= hi + k)", 5),))
    v = check(HERE.parent / "examples" / "clamp.py", cache)["clamp"]
    assert v.status == "PASSED", v.describe()
    assert "precondition agreement not checked: quantified requires" in v.detail


@needs_dafny
def test_quantified_ensures_false_is_an_issue(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clamp(monkeypatch, ensures_src=(("exists(k, lo <= k <= hi and k == result + 1)", 7),))
    v = check(HERE.parent / "examples" / "clamp.py", cache)["clamp"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "ensures false on the Python result"


@needs_dafny
def test_unsupported_source_is_not_checked(tmp_path: Path, cache: Path) -> None:
    src = tmp_path / "quant.py"
    src.write_text(
        "def q(x: int) -> int:\n"
        "    #@ ensures forall(k, k * 0 == 0)\n"
        "    return x\n"
    )
    [v] = runner.check_file(src, Config(cache_dir=cache))
    assert v.status == "NOT CHECKED", v.describe()
    # M1 rejects the quantifier outright; once the frontend accepts it, the
    # evaluator must still refuse to sample an unbounded one.
    assert "cannot read contracts" in v.detail or "unbounded quantifier" in v.detail


@needs_dafny
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


@needs_dafny
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


@needs_dafny
def test_no_artifacts_generates_model_without_touching_source_dir(tmp_path: Path, cache: Path) -> None:
    src = tmp_path / "clamp.py"
    shutil.copy(HERE.parent / "examples" / "clamp.py", src)
    v = check(src, cache)["clamp"]
    assert v.status == "PASSED", v.describe()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clamp.py"]


# --- caching -----------------------------------------------------------------


@needs_dafny
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


@needs_dafny
def test_cache_key_tracks_model_content(tmp_path: Path) -> None:
    src = tmp_path / "clamp.py"
    shutil.copy(HERE.parent / "examples" / "clamp.py", src)
    a = model.build(src, cache_dir=tmp_path / "c")
    proof = (HERE.parent / "examples" / "clamp.dfy").read_text() + "\n// a proof addition\n"
    (tmp_path / "clamp.dfy").write_text(proof)
    b = model.build(src, cache_dir=tmp_path / "c")
    assert a.key != b.key
    assert b.model_source == tmp_path / "clamp.dfy"


@needs_dafny
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


# --- list kinds: strategies --------------------------------------------------


def _collect(kinds: list[str], n: int = 300) -> list[tuple[Any, ...]]:
    seen: list[tuple[Any, ...]] = []

    @settings(settings.get_profile(runner.HYPOTHESIS_PROFILE), max_examples=n)
    @given(strategies.for_params(kinds))
    def collect(xs: tuple[Any, ...]) -> None:
        seen.append(xs)

    collect()
    return seen


def test_supported_kinds() -> None:
    for k in ("int", "bool", "list[int]", "list[bool]", "list[list[int]]"):
        assert strategies.supported(k), k
    for k in ("None", "str", "list[str]", "list[]", "list[int", "dict[int, int]", "list[list[float]]"):
        assert not strategies.supported(k), k


def test_list_strategy_shapes() -> None:
    lists = [xs for (xs,) in _collect(["list[int]"])]
    assert all(isinstance(x, list) and len(x) <= 8 for x in lists)
    assert all(isinstance(e, int) and not isinstance(e, bool) for x in lists for e in x)
    assert [] in lists
    assert any(len(x) >= 3 and x == sorted(x) and len(set(x)) > 1 for x in lists), "no sorted lists"
    assert any(x != sorted(x) for x in lists), "only sorted lists"
    assert any(len(x) >= 3 and 0 in x and min(x) >= 0 and len(set(x)) < len(x) for x in lists), "no small naturals"


def test_nested_list_strategy() -> None:
    lists = [xs for (xs,) in _collect(["list[list[bool]]"], 100)]
    assert all(isinstance(inner, list) and all(isinstance(b, bool) for b in inner) for x in lists for inner in x)
    assert any(x and any(inner for inner in x) for x in lists)


def test_pinned_lists() -> None:
    pins = [xs for (xs,) in strategies.pinned(["list[int]"])]
    for want in ([], [0], [-1], [1, 1]):
        assert want in pins, want
    assert any(len(x) >= 3 and x == sorted(x) for x in pins)
    assert any(len(x) >= 3 and x == sorted(x, reverse=True) for x in pins)
    assert any(2**63 in x for x in pins)
    nested = [xs for (xs,) in strategies.pinned(["list[list[int]]"])]
    assert [] in nested and [[]] in nested


def test_index_like_values() -> None:
    def hits(rows: list[tuple[Any, ...]]) -> set[str]:
        out = set()
        for xs, i in rows:
            n = len(xs)
            for name, v in (("-len-1", -n - 1), ("-len", -n), ("-1", -1), ("len-1", n - 1), ("len", n)):
                if n >= 2 and i == v:
                    out.add(name)
        return out

    want = {"-len-1", "-len", "-1", "len-1", "len"}
    assert hits(strategies.pinned(["list[int]", "int"])) == want
    assert hits(_collect(["list[int]", "int"])) == want


def test_finite_space_excludes_lists() -> None:
    assert strategies.finite_space(["list[bool]"]) is None


# --- list kinds: values --------------------------------------------------------


def test_kind_aware_equality() -> None:
    assert values.same("list[int]", [1, 2], [1, 2])
    assert not values.same("list[int]", [1], [True])
    assert not values.same("list[bool]", [True], [1])
    assert not values.same("list[int]", [1, 2], (1, 2))
    assert not values.same("list[int]", [1, 2], [1, 2, 3])
    assert values.same("list[list[int]]", [[1], []], [[1], []])
    assert not values.same("list[list[int]]", [[1], []], [[1], [0]])
    assert not values.same("int", 1, True)


def test_freeze_is_hashable_and_kind_preserving() -> None:
    a = values.freeze(([[1], [2, 3]], 4))
    assert hash(a) == hash(values.freeze(([[1], [2, 3]], 4)))
    assert a != values.freeze(([[1], [2]], 4))
    assert values.freeze(([],)) != values.freeze(((),))


def test_dafny_seq_conversion() -> None:
    _dafny = pytest.importorskip("_dafny")
    v = [[1, 2], [], [3]]
    d = values.to_dafny("list[list[int]]", v)
    assert isinstance(d, _dafny.Seq) and all(isinstance(e, _dafny.Seq) for e in d)
    assert values.from_dafny("list[list[int]]", d) == v
    assert values.to_dafny("int", 5) == 5
    # A model returning the wrong shape is passed through for `same` to reject.
    assert values.from_dafny("list[int]", 7) == 7
    adapted = values.adapt(lambda xs, n: _dafny.Seq(list(xs)[n:]), ["list[int]", "int"], "list[int]")
    assert adapted([1, 2, 3], 1) == [2, 3]


# --- the spec evaluator ------------------------------------------------------


def ev(text: str, **env: Any) -> Any:
    return specs_eval.compile_spec(text, 1).evaluate(env)


def test_eval_implies_and_iff() -> None:
    assert ev("x > 0 ==> x != 0", x=0) is True
    assert ev("x > 0 ==> x == 2", x=1) is False
    assert ev("len(xs) > 0 ==> xs[0] > 0", xs=[]) is True  # short-circuits
    assert ev("(x > 0) <==> (x >= 1)", x=0) is True
    assert ev("(x > 0) <==> (x >= 0)", x=0) is False


def test_eval_bounded_forall() -> None:
    spec = "forall(k, 0 <= k < len(xs) ==> xs[k] >= 0)"
    assert ev(spec, xs=[]) is True
    assert ev(spec, xs=[1, 2]) is True
    assert ev(spec, xs=[1, -1]) is False
    assert ev("forall(k: int, 0 <= k < n ==> k * k >= k)", n=50) is True
    assert ev("forall(k, k >= 1 ==> len(xs) > k ==> xs[k - 1] <= xs[k])", xs=[1, 2, 2, 5]) is True
    assert ev("forall(k, k >= 1 ==> len(xs) > k ==> xs[k - 1] <= xs[k])", xs=[1, 2, 0]) is False
    assert ev("forall(k, a < k and k <= b ==> k > a)", a=-3, b=4) is True
    assert ev("forall(k, k in range(a, b) ==> k < b)", a=0, b=10) is True


def test_eval_bounded_exists() -> None:
    spec = "exists(k, 0 <= k < len(xs) and xs[k] == 0)"
    assert ev(spec, xs=[3, 0]) is True
    assert ev(spec, xs=[3, 1]) is False
    assert ev(spec, xs=[]) is False


def test_eval_membership_premise() -> None:
    assert ev("forall(k, k in xs ==> k > 0)", xs=[1, 2]) is True
    assert ev("forall(k, k in xs ==> k > 0)", xs=[1, 0]) is False
    assert ev("exists(k, k in xs and k == 3)", xs=[1, 3]) is True
    assert ev("exists(k, k in xs and k == 3)", xs=[]) is False


def test_eval_nested_quantifiers_and_shadowing() -> None:
    spec = "forall(i, 0 <= i < len(xs) ==> forall(j, i < j < len(xs) ==> xs[i] <= xs[j]))"
    assert ev(spec, xs=[1, 2, 2, 9]) is True
    assert ev(spec, xs=[1, 3, 2]) is False
    assert ev("forall(x, 0 <= x < 3 ==> x < 3)", x=100) is True


def test_eval_guard_inside_same_kind_quantifier() -> None:
    # `i` is bounded only through the inner quantifier's guard, and its upper
    # bound only transitively (i < j < len(xs)).
    dup = "exists(i, exists(j, 0 <= i < j < len(xs) and xs[i] == xs[j]))"
    assert ev(dup, xs=[1, 2, 1]) is True
    assert ev(dup, xs=[1, 2, 3]) is False
    assert ev(dup, xs=[]) is False
    distinct = "forall(i, forall(j, 0 <= i < j < len(xs) ==> xs[i] != xs[j]))"
    assert ev(distinct, xs=[1, 2, 3]) is True
    assert ev(distinct, xs=[3, 2, 3]) is False
    assert ev("forall(k, 0 <= k <= n <= 5 ==> k <= 5)", n=3) is True  # k <= 5 transitively
    assert ev("forall(k, 10 > n >= k > 0 ==> k < 10)", n=4) is True  # descending chain
    assert ev("forall(k, 0 <= k == m < 4 ==> k == m)", m=2) is True


def test_eval_result_globals_and_native_all() -> None:
    def double(n: int) -> int:
        return 2 * n

    assert ev("forall(k, 0 <= k < len(result) ==> result[k] == double(xs[k]))",
              xs=[1, 2], result=[2, 4], double=double) is True
    assert ev("all(xs[k] <= xs[k + 1] for k in range(len(xs) - 1))", xs=[1, 2, 3]) is True
    assert ev("any(x == 2 for x in xs)", xs=[1, 3]) is False


@pytest.mark.parametrize("text", [
    "forall(k, k * 0 == 0)",
    "forall(k, k >= 0 ==> k + 1 > k)",
    "exists(k, k > x)",
    "forall(k, 0 <= k ==> forall(j, j < k ==> j < k))",
    "forall(k, x < y ==> k == k)",
    "forall(k, k <= k + 1 ==> True)",
    "forall(i, exists(j, 0 <= i < j < 3 ==> True))",  # mixed kinds: no look-through
    "forall(k, 0 <= k < x or k < 3 ==> True)",  # a disjunction bounds nothing
    "forall(k, 0 <= k > 3 ==> True)",  # mixed directions: no upper bound
    "forall(k, forall(k, 0 <= k < 3 ==> True))",  # the inner guard bounds the inner k only
])
def test_eval_unbounded_is_unevaluable(text: str) -> None:
    with pytest.raises(specs_eval.Unbounded):
        specs_eval.compile_spec(text, 3).evaluate({"x": 0, "y": 1})


def test_eval_domain_over_cap_is_unevaluable() -> None:
    spec = specs_eval.compile_spec("forall(k, 0 <= k < n ==> k >= 0)", 1)
    assert spec.evaluate({"n": specs_eval.CAP}) is True
    with pytest.raises(specs_eval.Unbounded):
        spec.evaluate({"n": specs_eval.CAP + 1})


def test_eval_picks_the_smallest_domain() -> None:
    # `k == x` alone bounds the domain to one value even though the interval is huge.
    assert ev("exists(k, 0 <= k < 10**9 and k == x)", x=5) is True
    assert ev("forall(k, k in xs and 0 <= k < 10**9 ==> k >= 0)", xs=[1]) is True


# --- list kinds: the checker (hand-built contracts, no frontend needed) --------


def contract(name: str, params: list[tuple[str, str]], returns: str,
             requires: tuple[str, ...] = (), ensures: tuple[str, ...] = (), **kw: Any) -> FnContract:
    req = tuple((t, 1) for t in requires)
    ens = tuple((t, 2) for t in ensures)
    return FnContract(name, params, returns, [specs.to_python(specs.parse(t, ln)) for t, ln in req],
                      [specs.to_python(specs.parse(t, ln)) for t, ln in ens], requires_src=req, ensures_src=ens, **kw)


def target(c: FnContract, py: runner.Fn, mdl: runner.Fn | None = None, pre: runner.Fn | None = None,
           note: str = "", **globs: Any) -> Target:
    return Target(c, py, mdl or py, pre, dict(globs), pre_note=note)


def run(t: Target, **kw: Any) -> Verdict:
    return runner.check_function(t, Config(**kw))


def test_list_function_passes() -> None:
    c = contract("tail", [("xs", "list[int]")], "list[int]", ("len(xs) > 0",), ("len(result) == len(xs) - 1",))
    v = run(target(c, lambda xs: xs[1:], pre=lambda xs: len(xs) > 0))
    assert v.status == "PASSED", v.describe()
    assert v.accepted >= Config().min_accepted


def test_list_model_divergence_is_an_issue() -> None:
    c = contract("tail", [("xs", "list[int]")], "list[int]", ("len(xs) > 0",))
    v = run(target(c, lambda xs: xs[1:], lambda xs: xs[2:], lambda xs: len(xs) > 0))
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None and len(v.counterexample["xs"]) == 2  # shrunk


def test_list_sum_dropping_last_is_an_issue() -> None:
    c = contract("total", [("xs", "list[int]")], "int")
    v = run(target(c, lambda xs: sum(xs), lambda xs: sum(xs[:-1])))
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None and len(v.counterexample["xs"]) == 1 and v.counterexample["xs"][0] != 0


def test_list_result_element_kind_mismatch_is_an_issue() -> None:
    c = contract("flags", [("xs", "list[int]")], "list[bool]")
    v = run(target(c, lambda xs: [x > 0 for x in xs], lambda xs: [int(x > 0) for x in xs]))
    assert v.status == "ISSUE" and v.kind == "result mismatch", v.describe()


def test_argument_mutation_is_an_issue() -> None:
    def sneaky(xs: list[int]) -> int:
        xs.sort()
        return len(xs)

    c = contract("count", [("xs", "list[int]")], "int")
    v = run(target(c, sneaky, lambda xs: len(xs)))
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "argument mutated"
    assert v.counterexample is not None
    xs = v.counterexample["xs"]
    assert xs != sorted(xs), "the counterexample must show the argument before the call"
    assert "xs" in v.detail


def test_nested_argument_mutation_is_an_issue() -> None:
    def sneaky(xss: list[list[int]]) -> int:
        if xss:
            xss[0].append(0)
        return len(xss)

    c = contract("count", [("xss", "list[list[int]]")], "int")
    v = run(target(c, sneaky, lambda xss: len(xss)))
    assert v.status == "ISSUE" and v.kind == "argument mutated", v.describe()


def test_list_precondition_disagreement_is_an_issue() -> None:
    c = contract("head", [("xs", "list[int]")], "int", ("len(xs) > 0",))
    v = run(target(c, lambda xs: xs[0] if xs else 0, pre=lambda xs: True))
    assert v.status == "ISSUE" and v.kind == "precondition disagreement", v.describe()
    assert v.counterexample == {"xs": []}


def test_quantified_requires_and_ensures() -> None:
    sorted_req = "forall(i, 0 <= i < len(xs) - 1 ==> xs[i] <= xs[i + 1])"
    c = contract("member", [("xs", "list[int]"), ("x", "int")], "bool", (sorted_req,),
                 ("result == exists(k, 0 <= k < len(xs) and xs[k] == x)",), unevaluable=True)
    v = run(target(c, lambda xs, x: x in xs, note="precondition agreement not checked: quantified requires"))
    assert v.status == "PASSED", v.describe()
    assert "precondition agreement not checked: quantified requires" in v.detail


def test_quantified_ensures_false_on_list_result() -> None:
    c = contract("ident", [("xs", "list[int]")], "list[int]", (),
                 ("forall(k, 0 <= k < len(result) ==> result[k] > 0)",))
    v = run(target(c, lambda xs: list(xs)))
    assert v.status == "ISSUE" and v.kind == "ensures false on the Python result", v.describe()
    assert v.counterexample == {"xs": [0]}


def test_unbounded_quantifier_in_hand_built_contract() -> None:
    c = contract("f", [("x", "int")], "int", (), ("forall(k, k * 0 == 0)",))
    v = run(target(c, lambda x: x))
    assert v.status == "NOT CHECKED", v.describe()
    assert "unbounded quantifier" in v.detail


def test_unevaluable_without_source_is_not_checked() -> None:
    c = FnContract("f", [("x", "int")], "int", [], ["forall(k, k * 0 == 0)"], unevaluable=True)
    v = run(target(c, lambda x: x))
    assert v.status == "NOT CHECKED", v.describe()
    assert "quantifier" in v.detail


def test_quantifier_domain_over_cap_skips_inputs() -> None:
    c = contract("f", [("n", "int")], "int", ("forall(k, 0 <= k < n ==> k >= 0)",))
    v = run(target(c, lambda n: n))
    assert v.status == "PASSED", v.describe()
    assert "quantifier" in v.detail and "skipped" in v.detail


def test_index_like_precondition_is_reachable() -> None:
    c = contract("at_end", [("xs", "list[int]"), ("i", "int")], "int",
                 ("len(xs) > 0", "i == len(xs) or i == -len(xs) - 1"))
    v = run(target(c, lambda xs, i: len(xs)))
    assert v.status == "PASSED", v.describe()


# --- list kinds against real Dafny ----------------------------------------------

SEQ_PROGRAM = """
function Tail(xs: seq<int>): seq<int>
  requires |xs| > 0
{ xs[1..] }

function Firsts(xss: seq<seq<int>>): seq<int>
{ if |xss| == 0 then [] else (if |xss[0]| == 0 then [] else [xss[0][0]]) + Firsts(xss[1..]) }

function Pos(xs: seq<int>): seq<bool>
{ seq(|xs|, i requires 0 <= i < |xs| => xs[i] > 0) }
"""


@needs_dafny
def test_list_conversion_through_real_dafny(tmp_path: Path) -> None:
    name = "LscSeqTest"
    model._translate_into(tmp_path / "e", tmp_path, model.dafny_version(), [(name, SEQ_PROGRAM)])
    mod = model._import(tmp_path / "e" / "out-py", name)
    def fn(dafny_name: str, kinds: list[str], returns: str) -> runner.Fn:
        f = model.resolve(mod.default__, dafny_name)
        assert f is not None, dafny_name
        return values.adapt(f, kinds, returns)

    tail = fn("Tail", ["list[int]"], "list[int]")
    c = contract("tail", [("xs", "list[int]")], "list[int]", ("len(xs) > 0",))
    v = run(target(c, lambda xs: xs[1:], tail, lambda xs: len(xs) > 0))
    assert v.status == "PASSED", v.describe()

    firsts = fn("Firsts", ["list[list[int]]"], "list[int]")
    c = contract("firsts", [("xss", "list[list[int]]")], "list[int]")
    v = run(target(c, lambda xss: [x[0] for x in xss if x], firsts))
    assert v.status == "PASSED", v.describe()

    pos = fn("Pos", ["list[int]"], "list[bool]")
    c = contract("pos", [("xs", "list[int]")], "list[bool]")
    v = run(target(c, lambda xs: [x >= 0 for x in xs], pos))  # wrong on 0
    assert v.status == "ISSUE" and v.kind == "result mismatch", v.describe()
    assert v.counterexample == {"xs": [0]}


# --- list kinds end to end ------------------------------------------------------


@needs_dafny
def test_list_fixture_passes(cache: Path) -> None:
    verdicts = check(FIXTURES / "list_ok.py", cache)
    assert set(verdicts) == {"head", "tail", "first_of_first"}
    for v in verdicts.values():
        assert v.status == "PASSED", v.describe()


@needs_dafny
def test_list_model_divergence_fixture_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "list_bad_tail.py", cache)["tail"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "result mismatch"
    assert v.counterexample is not None and len(v.counterexample["xs"]) == 2


@needs_dafny
def test_list_argument_mutation_fixture_is_an_issue(cache: Path) -> None:
    def tail(xs: list[int]) -> list[int]:
        out = xs[1:]
        xs.clear()
        return out

    v = check(FIXTURES / "list_ok.py", cache, python_overrides={"tail": tail})["tail"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "argument mutated"


@needs_dafny
def test_list_quantified_fixture(cache: Path) -> None:
    v = check(FIXTURES / "list_quant.py", cache)["smallest"]
    assert v.status == "PASSED", v.describe()
    assert "precondition agreement not checked: quantified requires" in v.detail


def test_requires_quantified_looks_only_at_requires() -> None:
    ens_only = contract("f", [("xs", "list[int]")], "bool", (),
                        ("forall(k, 0 <= k < len(xs) ==> xs[k] > 0)",), unevaluable=True)
    assert not model.requires_quantified(ens_only)
    assert model.pre_eligible(ens_only)
    req = contract("f", [("xs", "list[int]")], "bool", ("exists(k, k in xs and k > 0)",), unevaluable=True)
    assert model.requires_quantified(req) and not model.pre_eligible(req)
    # Without the source texts, an unevaluable contract may quantify anywhere.
    assert model.requires_quantified(FnContract("f", [("x", "int")], "int", [], [], unevaluable=True))
    assert not model.requires_quantified(FnContract("f", [("x", "int")], "int", ["x > 0"], []))


@needs_dafny
def test_pre_module_build_failure_falls_back(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # E.g. a requires the frontend accepts in a spec but not in a function body.
    real = model.run_lsc

    def failing(args: list[str], cwd: Path) -> str:
        files = list(cwd.glob("*.py"))
        if args[0] in ("gen", "info") and files and model.PRE_PREFIX in files[0].read_text():
            raise model.BuildError("lsc gen failed:\nerror: not supported in a function body")
        return real(args, cwd)

    monkeypatch.setattr(model, "run_lsc", failing)
    v = check(HERE.parent / "examples" / "clamp.py", cache)["clamp"]
    assert v.status == "PASSED", v.describe()
    assert "precondition agreement not checked: synthetic precondition module failed to build" in v.detail


# --- BaseExceptions are findings, not harness exits --------------------------


@pytest.mark.parametrize("exc", [SystemExit(0), KeyboardInterrupt(), GeneratorExit()],
                         ids=lambda e: type(e).__name__)
def test_base_exception_in_python_is_an_issue(exc: BaseException) -> None:
    def py(x: int) -> int:
        if x == 3:
            raise exc
        return x

    v = run(target(contract("f", [("x", "int")], "int"), py, lambda x: x))
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "python raised" and type(exc).__name__ in v.detail
    assert v.counterexample == {"x": 3}


def test_base_exception_in_model_is_an_issue() -> None:
    def mdl(x: int) -> int:
        if x == 3:
            raise SystemExit(1)
        return x

    v = run(target(contract("f", [("x", "int")], "int"), lambda x: x, mdl))
    assert v.status == "ISSUE" and v.kind == "model raised" and "SystemExit" in v.detail, v.describe()


def test_base_exception_in_contracts_is_an_issue() -> None:
    def boom(x: int) -> bool:
        if x == 3:
            raise KeyboardInterrupt
        return True

    c = contract("f", [("x", "int")], "int", ("boom(x)",))
    v = run(target(c, lambda x: x, boom=boom))
    assert v.status == "ISSUE" and v.kind == "requires raised" and "KeyboardInterrupt" in v.detail, v.describe()
    c = contract("f", [("x", "int")], "int", (), ("boom(result)",))
    v = run(target(c, lambda x: x, boom=boom))
    assert v.status == "ISSUE" and v.kind == "ensures raised" and "KeyboardInterrupt" in v.detail, v.describe()


@needs_dafny
def test_system_exit_fixture_is_an_issue(cache: Path) -> None:
    v = check(FIXTURES / "raises_exit.py", cache)["stop_at_three"]
    assert v.status == "ISSUE", v.describe()
    assert v.kind == "python raised" and "SystemExit" in v.detail
    assert v.counterexample == {"x": 3}


# --- the synthetic precondition module -------------------------------------------


def test_pre_source_wraps_requires_in_bool() -> None:
    c = contract("f", [("x", "int"), ("y", "int")], "int", ("x", "x ==> y"))
    src = model.pre_source("", {"f": c})
    assert "return bool(x) and bool(not x or y)" in src or "return bool(x) and bool(not x or bool(y))" in src, src


@needs_dafny
def test_truthy_requires_build_and_agree(cache: Path) -> None:
    verdicts = check(FIXTURES / "truthy_pre.py", cache)
    for v in verdicts.values():
        assert v.status == "PASSED", v.describe()
        assert "precondition agreement not checked" not in v.detail, v.describe()


@needs_dafny
def test_model_may_use_the_standard_library(cache: Path) -> None:
    # Generated code names `Std.Collections.Seq.*` directly; the per-build
    # wrapper module must still see the top-level `Std`.
    v = check(HERE / "golden" / "comprehensions.py", cache)
    assert v and all(x.status == "PASSED" for x in v.values()), [x.describe() for x in v.values()]


def test_extra_rounds_when_too_few_inputs_accepted() -> None:
    # About 1 in 6 small-natural lists meets this; one round of 60 examples is
    # not enough for 30 distinct accepted inputs, three rounds are.
    c = contract("f", [("xs", "list[int]")], "int",
                 ("forall(i: nat, i < len(xs) ==> xs[i] >= 0)", "exists(j: nat, j < len(xs) and xs[j] == 0)"))
    one = run(target(c, lambda xs: len(xs)), max_examples=60, min_accepted=30, extra_rounds=0)
    assert one.status == "NOT CHECKED", one.describe()
    more = run(target(c, lambda xs: len(xs)), max_examples=60, min_accepted=30, extra_rounds=4)
    assert more.status == "PASSED", more.describe()
    assert more.tried > one.tried
    again = run(target(c, lambda xs: len(xs)), max_examples=60, min_accepted=30, extra_rounds=4)
    assert (again.accepted, again.tried) == (more.accepted, more.tried)  # still deterministic
