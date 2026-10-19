"""Raw IR for the M2 subset: lists, loops with invariants, recursion, comprehensions."""
from __future__ import annotations

from typing import Any

import pytest

from conftest import extract, fn
from lemmascript_python.errors import UnsupportedError

V = lambda n: {"kind": "var", "name": n}  # noqa: E731
N = lambda v: {"kind": "num", "value": v}  # noqa: E731
R = {"kind": "result"}


def B(op: str, left: Any, right: Any) -> dict[str, Any]:
    return {"kind": "binop", "op": op, "left": left, "right": right}


def U(op: str, e: Any) -> dict[str, Any]:
    return {"kind": "unop", "op": op, "expr": e}


def F(obj: Any, field: str) -> dict[str, Any]:
    return {"kind": "field", "obj": obj, "field": field}


def M(obj: Any, method: str, *args: Any) -> dict[str, Any]:
    return {"kind": "call", "fn": F(obj, method), "args": list(args)}


def LEN(n: str) -> dict[str, Any]:
    return F(V(n), "length")


def ret(stmt: dict[str, Any]) -> Any:
    assert stmt["kind"] == "return", stmt
    return stmt["value"]


def rejects(src: str, message: str, line: int | None = None) -> None:
    with pytest.raises(UnsupportedError) as exc:
        extract(src)
    assert message in str(exc.value), str(exc.value)
    if line is not None:
        assert exc.value.line == line, exc.value


# ── types and expressions ────────────────────────────────────────────

def test_list_types() -> None:
    f = fn("def f(xs: list[int], bs: list[bool], g: list[list[int]]) -> list[int]:\n    return xs\n")
    assert [p["tsType"] for p in f["params"]] == ["int[]", "boolean[]", "int[][]"]
    assert f["returnType"] == "int[]"


def test_python_files_always_use_clamping_slices() -> None:
    assert extract("def f(x: int) -> int:\n    return x\n")["directives"]["options"] == {"safe-slice": True}


def test_safe_slice_cannot_be_turned_off() -> None:
    rejects("#@ option safe-slice false\ndef f(x: int) -> int:\n    return x\n",
            "Python slices always clamp", 1)


def test_len_index_and_negative_literal_index() -> None:
    f = fn("def f(xs: list[int], i: int) -> int:\n    #@ requires len(xs) > 0\n    return len(xs) + xs[i] + xs[0] + xs[-1]\n")
    assert f["requires"] == [B(">", LEN("xs"), N(0))]
    idx = lambda e: {"kind": "index", "obj": V("xs"), "idx": e}  # noqa: E731
    assert ret(f["body"][0]) == B("+", B("+", B("+", LEN("xs"), idx(V("i"))), idx(N(0))),
                                  idx(B("-", LEN("xs"), N(1))))


def test_slices_clamp_and_fill_omitted_bounds() -> None:
    f = fn("def f(xs: list[int], a: int) -> list[int]:\n    return xs[a:] + xs[:a] + xs[1:-1] + xs[:]\n")
    s = lambda lo, hi: M(V("xs"), "slice", lo, hi)  # noqa: E731
    assert ret(f["body"][0]) == B("arrayConcat", B("arrayConcat", B("arrayConcat",
        s(V("a"), LEN("xs")), s(N(0), V("a"))), s(N(1), U("-", N(1)))), s(N(0), LEN("xs")))


def test_list_literals_membership_and_equality() -> None:
    f = fn("def f(xs: list[int], x: int) -> bool:\n    return x in xs and x not in [1, 2] and xs == [x] and xs != []\n")
    lit = lambda *e: {"kind": "arrayLiteral", "elems": list(e)}  # noqa: E731
    assert ret(f["body"][0]) == B("&&", B("&&", B("&&",
        M(V("xs"), "includes", V("x")), U("!", M(lit(N(1), N(2)), "includes", V("x")))),
        B("===", V("xs"), lit(V("x")))), B("!==", V("xs"), lit()))


def test_list_truthiness_is_non_emptiness() -> None:
    f = fn("def f(xs: list[int]) -> bool:\n    if xs:\n        return not xs\n    return bool(xs)\n")
    assert f["body"][0]["cond"] == B(">", LEN("xs"), N(0))
    assert ret(f["body"][0]["then"][0]) == B("===", LEN("xs"), N(0))
    assert ret(f["body"][1]) == B(">", LEN("xs"), N(0))


def test_min_max_of_a_list() -> None:
    f = fn("def f(xs: list[int]) -> int:\n    #@ requires len(xs) > 0\n    return min(xs) + max(xs)\n")
    C = lambda n: {"kind": "call", "fn": V(n), "args": [V("xs")]}  # noqa: E731
    assert ret(f["body"][0]) == B("+", C("MinOfSeq"), C("MaxOfSeq"))


def test_comprehensions_become_filter_and_map() -> None:
    f = fn("def f(xs: list[int]) -> list[int]:\n    return [x * 2 for x in xs if x > 0]\n")
    lam = lambda body, ty: {"kind": "lambda", "params": [{"name": "x", "tsType": "int"}], "body": body,  # noqa: E731
                            "returnTsType": ty}
    assert ret(f["body"][0]) == M(M(V("xs"), "filter", lam(B(">", V("x"), N(0)), "boolean")), "map",
                                  lam(B("*", V("x"), N(2)), "int"))


def test_all_any_in_code() -> None:
    f = fn("def f(xs: list[int]) -> bool:\n    return all(x > 0 for x in xs) or any(x == 3 for x in xs)\n")
    lam = lambda body: {"kind": "lambda", "params": [{"name": "x", "tsType": "int"}], "body": body,  # noqa: E731
                        "returnTsType": "boolean"}
    assert ret(f["body"][0]) == B("||", M(V("xs"), "every", lam(B(">", V("x"), N(0)))),
                                  M(V("xs"), "some", lam(B("===", V("x"), N(3)))))


# ── specs: quantifiers ───────────────────────────────────────────────

def test_quantifiers_in_specs() -> None:
    f = fn("def f(xs: list[int]) -> int:\n"
           "    #@ requires forall(k, 0 <= k < len(xs) ==> xs[k] >= 0)\n"
           "    #@ requires exists(j: nat, j < len(xs) and xs[j] == 0)\n"
           "    #@ ensures all(x >= 0 for x in xs)\n"
           "    #@ ensures all(xs[k] <= xs[k + 1] for k in range(0, len(xs) - 1))\n"
           "    return 0\n")
    xk = {"kind": "index", "obj": V("xs"), "idx": V("k")}
    assert f["requires"][0] == {"kind": "forall", "var": "k", "varType": "int", "body":
        B("==>", B("&&", B("<=", N(0), V("k")), B("<", V("k"), LEN("xs"))), B(">=", xk, N(0)))}
    assert f["requires"][1]["kind"] == "exists" and f["requires"][1]["varType"] == "nat"
    assert f["ensures"][0] == {"kind": "forall", "var": "x", "varType": "int", "body":
        B("==>", M(V("xs"), "includes", V("x")), B(">=", V("x"), N(0)))}
    q = f["ensures"][1]
    assert q["kind"] == "forall" and q["var"] == "k"
    assert q["body"]["left"] == B("&&", B("<=", N(0), V("k")), B("<", V("k"), B("-", LEN("xs"), N(1))))


# ── statements and loops ─────────────────────────────────────────────

ARRAY_SUM = '''
def total(xs: list[int]) -> int:
    #@ type i nat
    #@ ensures result == sum_to(xs, len(xs))
    s = 0
    i = 0
    while i < len(xs):
        #@ invariant 0 <= i <= len(xs)
        #@ invariant s == sum_to(xs, i)
        #@ decreases len(xs) - i
        s += xs[i]
        i += 1
    return s
'''


def test_while_with_invariants_and_decreases() -> None:
    f = fn(ARRAY_SUM)
    w = f["body"][2]
    assert w["kind"] == "while" and w["cond"] == B("<", V("i"), LEN("xs"))
    assert w["invariants"][0] == B("&&", B("<=", N(0), V("i")), B("<=", V("i"), LEN("xs")))
    assert w["invariants"][1] == B("===", V("s"), {"kind": "call", "fn": V("sum_to"), "args": [V("xs"), V("i")]})
    assert w["decreases"] == B("-", LEN("xs"), V("i"))
    assert w["doneWith"] is None
    assert [s["kind"] for s in w["body"]] == ["assign", "assign"]
    assert f["typeAnnotations"] == [{"name": "i", "type": "nat"}]


def test_for_range_desugars_to_a_counted_while() -> None:
    f = fn("def f(n: int) -> int:\n"
           "    s = 0\n"
           "    for i in range(1, n):\n"
           "        #@ invariant s >= 0\n"
           "        if i == 3:\n"
           "            continue\n"
           "        s += i\n"
           "    return s\n")
    b = f["body"]
    assert b[1] == {"kind": "let", "name": "i", "mutable": True, "tsType": "int", "init": N(1), "line": 3}
    w = b[2]
    assert w["kind"] == "while" and w["cond"] == B("<", V("i"), V("n"))
    # Automatic, always-true bound invariant first, then the user's.
    assert w["invariants"][0] == B("&&", B("<=", N(1), V("i")), B("||", B("<=", V("i"), V("n")), B("===", V("i"), N(1))))
    assert w["invariants"][1] == B(">=", V("s"), N(0))
    inc = {"kind": "assign", "target": "i", "value": B("+", V("i"), N(1)), "line": 3}
    # `continue` advances the counter first, like Python's iterator does.
    assert w["body"][0]["then"] == [inc, {"kind": "continue", "line": 6}]
    assert w["body"][-1] == inc


def test_for_range_binds_a_non_constant_stop_once() -> None:
    f = fn("def f(xs: list[int]) -> int:\n    s = 0\n    for i in range(len(xs) - 1):\n        s += xs[i]\n    return s\n")
    b = f["body"]
    assert b[1]["kind"] == "let" and b[1]["name"] == "i_stop" and b[1]["init"] == B("-", LEN("xs"), N(1))
    assert b[3]["cond"] == B("<", V("i"), V("i_stop"))


def test_for_enumerate_and_for_in() -> None:
    f = fn("def f(xs: list[int]) -> int:\n"
           "    s = 0\n"
           "    for i, x in enumerate(xs):\n"
           "        s += i * x\n"
           "    for y in xs:\n"
           "        #@ invariant s == s\n"
           "        s += y\n"
           "    return s\n")
    b = f["body"]
    assert b[1]["name"] == "i" and b[1]["init"] == N(0)
    w = b[2]
    assert w["cond"] == B("<", V("i"), LEN("xs"))
    assert w["invariants"][0] == B("&&", B("<=", N(0), V("i")), B("<=", V("i"), LEN("xs")))
    assert w["body"][0] == {"kind": "let", "name": "x", "mutable": False, "tsType": "int",
                            "init": {"kind": "index", "obj": V("xs"), "idx": V("i")}, "line": 3}
    fo = b[3]
    assert fo["kind"] == "forof" and fo["names"] == ["y"] and fo["iterable"] == V("xs")
    assert fo["invariants"] == [B("===", V("s"), V("s"))]


def test_break_assert_and_ghost() -> None:
    f = fn("def f(xs: list[int]) -> int:\n"
           "    r = -1\n"
           "    #@ ghost seen: int = 0\n"
           "    for i in range(len(xs)):\n"
           "        #@ assert i >= 0\n"
           "        if xs[i] == 0:\n"
           "            r = i\n"
           "            break\n"
           "        #@ ghost seen = seen + 1\n"
           "    return r\n")
    b = f["body"]
    assert b[1] == {"kind": "ghostLet", "name": "seen", "tsType": "int", "init": N(0), "line": 3}
    w = b[4]
    assert w["body"][0] == {"kind": "assert", "expr": B(">=", V("i"), N(0)), "line": 5}
    assert w["body"][1]["then"][1] == {"kind": "break", "line": 8}
    assert w["body"][2] == {"kind": "ghostAssign", "target": "seen", "value": B("+", V("seen"), N(1)), "line": 9}


def test_recursion_with_decreases() -> None:
    f = fn("def count(xs: list[int], k: int) -> int:\n"
           "    #@ decreases len(xs)\n"
           "    if not xs:\n"
           "        return 0\n"
           "    return (1 if xs[0] == k else 0) + count(xs[1:], k)\n")
    assert f["decreases"] == LEN("xs")
    assert f["body"][0]["cond"] == B("===", LEN("xs"), N(0))


# ── mutation and aliasing ────────────────────────────────────────────

def test_in_place_updates_of_a_fresh_local() -> None:
    f = fn("def f(xs: list[int]) -> list[int]:\n"
           "    ys: list[int] = []\n"
           "    zs = xs[:]\n"
           "    for x in xs:\n"
           "        ys.append(x)\n"
           "    zs[0] = 7\n"
           "    ys += zs\n"
           "    return ys\n")
    b = f["body"]
    assert b[0] == {"kind": "let", "name": "ys", "mutable": True, "tsType": "int[]",
                    "init": {"kind": "arrayLiteral", "elems": []}, "line": 2}
    assert b[2]["body"][0] == {"kind": "expr", "expr": M(V("ys"), "push", V("x")), "line": 5}
    assert b[3] == {"kind": "assign", "target": "zs", "value": M(V("zs"), "with", N(7 - 7), N(7)), "line": 6}
    assert b[4] == {"kind": "assign", "target": "ys", "value": B("arrayConcat", V("ys"), V("zs")), "line": 7}


@pytest.mark.parametrize(("src", "message", "line"), [
    ("def f(xs: list[int]) -> int:\n    xs.append(1)\n    return 0\n",
     "cannot modify parameter 'xs' in place", 2),
    ("def f(xs: list[int]) -> int:\n    xs[0] = 1\n    return 0\n", "cannot modify parameter 'xs' in place", 2),
    ("def f(xs: list[int]) -> int:\n    xs += [1]\n    return 0\n", "cannot modify parameter 'xs' in place", 2),
    ("def f(xs: list[int]) -> int:\n    ys = xs\n    ys.append(1)\n    return 0\n",
     "'ys' is modified in place, so it must be bound to a fresh list", 2),
    ("def f(xs: list[int]) -> list[int]:\n    ys = [1]\n    zs = ys\n    ys.append(2)\n    return zs\n",
     "'ys' is modified in place, so it cannot be shared", 3),
    ("def g(a: list[int]) -> int:\n    return 0\n\n"
     "def f() -> int:\n    ys = [1]\n    n = g(ys)\n    ys.append(2)\n    return n\n",
     "'ys' is modified in place, so it cannot be shared", 6),
    ("def f(g: list[list[int]]) -> int:\n    g[0].append(1)\n    return 0\n",
     "nested in-place modification", 2),
    ("def f() -> int:\n    ys = [1]\n    for y in ys:\n        ys.append(y)\n    return 0\n",
     "'ys' is modified inside a loop over it", 4),
    ("def f() -> int:\n    ys = [1]\n    x = ys.pop()\n    return x\n", "'pop' is not supported yet", 3),
])
def test_aliasing_rules(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


@pytest.mark.parametrize(("src", "message", "line"), [
    ("def f(n: int) -> int:\n    for i in range(n):\n        pass\n    return i\n",
     "'i' is not available after its loop", 4),
    ("def f(n: int) -> int:\n    for i in range(n):\n        i = 0\n    return n\n",
     "cannot assign the loop variable 'i'", 3),
    ("def f(n: int) -> int:\n    for i in range(0, n, 2):\n        pass\n    return n\n",
     "only range(stop) and range(start, stop) are supported yet", 2),
    ("def f(n: int) -> int:\n    while n > 0:\n        n -= 1\n    else:\n        n = 1\n    return n\n",
     "'while ... else' is not supported", 2),
    ("def f(xs: list[int]) -> list[int]:\n    return []\n", None, None),
    ("def f(xs: list[int]) -> int:\n    ys = []\n    return 0\n", "cannot infer the element type of '[]'", 2),
    ("def f(xs: list[int]) -> bool:\n    return xs < xs\n", "'<' on 'list[int]' is not supported", 2),
    ("def f(xs: list[int]) -> list[int]:\n    return xs[::2]\n", "slices with a step are not supported", 2),
    ("def f(xs: list[int]) -> list[int]:\n    return [x for x in xs for y in xs]\n",
     "comprehensions with one 'for' clause only", 2),
    ("def f(n: int) -> bool:\n    return all(k > 0 for k in range(n))\n",
     "'range' outside a 'for' loop is only supported in specs", 2),
    ("def f(n: int) -> int:\n    #@ requires forall(k, k)\n    return n\n", None, None),
    ("def f(n: int) -> int:\n    while n > 0:\n        #@ requires n > 0\n        n -= 1\n    return n\n",
     "'#@ requires' must come before the first statement of the function body", 3),
    ("def f(n: int) -> int:\n    #@ invariant n > 0\n    return n\n", "'#@ invariant' belongs at the start of a loop body", 2),
    ("def f(n: int) -> int:\n    #@ ghost g: int = 0\n    return g\n", "ghost variable 'g' cannot be used in code", 3),
])
def test_m2_rejections(src: str, message: str | None, line: int | None) -> None:
    if message is None:
        extract(src)  # accepted
    else:
        rejects(src, message, line)
