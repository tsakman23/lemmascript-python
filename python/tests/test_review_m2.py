"""Regression tests for the findings of the adversarial M2 soundness review.

Each "unsound" case was confirmed to verify in Dafny while Python returned a
different result; each must now be rejected or translated faithfully.
"""
from __future__ import annotations

from typing import Any

import pytest

from conftest import extract, fn
from lemmascript_python.errors import UnsupportedError


def rejects(src: str, message: str, line: int | None = None) -> None:
    with pytest.raises(UnsupportedError) as exc:
        extract(src)
    assert message in str(exc.value), str(exc.value)
    if line is not None:
        assert exc.value.line == line, exc.value


def V(n: str) -> dict[str, Any]:
    return {"kind": "var", "name": n}


# S1. A range bound that names the loop variable is evaluated before the counter is set.
def test_range_bound_naming_the_loop_variable_is_bound_once() -> None:
    b = fn("def f(n: int) -> int:\n    c = 0\n    for n in range(n):\n        c += 1\n    return c\n")["body"]
    assert b[1] == {"kind": "let", "name": "n_stop", "mutable": False, "tsType": "int", "init": V("n"), "line": 3}
    assert b[3]["cond"] == {"kind": "binop", "op": "<", "left": V("n"), "right": V("n_stop")}


# S2. Nested loops may not share a loop variable.
@pytest.mark.parametrize("src", [
    "def f() -> int:\n    c = 0\n    for i in range(3):\n        for i in range(2):\n            c += 1\n    return c\n",
    "def f(xs: list[int]) -> int:\n    c = 0\n    for i in range(3):\n        for i, x in enumerate(xs):\n            c += 1\n"
    "    return c\n",
    "def f(xs: list[int]) -> int:\n    c = 0\n    for x in xs:\n        for x in xs:\n            c += 1\n    return c\n",
])
def test_nested_loops_cannot_reuse_a_loop_variable(src: str) -> None:
    rejects(src, "is already the variable of an enclosing loop", 4)


def test_enumerate_counter_rebinding_a_range_bound() -> None:
    src = ("def f(xs: list[int], n: int) -> int:\n    c = 0\n    for j in range(n):\n"
           "        for n, x in enumerate(xs):\n            c += 1\n    return c\n")
    b = fn(src)["body"]
    assert any(s["kind"] == "let" and s["name"] == "j_stop" for s in b)


# S3. After a loop, its variable is unreadable even if it was bound in an enclosing block.
def test_loop_variable_bound_in_an_outer_block_is_hidden_after_the_loop() -> None:
    rejects("def f(i: int) -> int:\n    if i == i:\n        for i in range(10):\n            pass\n    return i\n",
            "'i' is not available after its loop", 5)


# S4. A loop variable over a list of lists refers to the inner list: no in-place change.
@pytest.mark.parametrize(("src", "line"), [
    ("def f(grid: list[list[int]]) -> int:\n    for r in grid:\n        r.append(1)\n    return len(grid)\n", 2),
    ("def f(grid: list[list[int]]) -> int:\n    for i, r in enumerate(grid):\n        r[0] = 5\n    return 0\n", 2),
    ("def f(grid: list[list[int]]) -> int:\n    for r in grid:\n        r += [1]\n    return 0\n", 2),
])
def test_loop_variables_cannot_be_changed_in_place(src: str, line: int) -> None:
    rejects(src, "is a loop variable bound to an element of the list being iterated", line)


# S5. `xs[i] += ys` on a list of lists extends the inner list in place.
def test_augmented_assignment_to_a_list_element_that_is_a_list() -> None:
    rejects("def f(g: list[list[int]]) -> int:\n    h = list(g)\n    h[0] += [1]\n    return len(g[0])\n",
            "'+=' on an element that is a list changes that inner list in place", 3)


# S6. Python iterates the object it got at loop entry.
def test_for_over_an_expression_iterates_a_snapshot() -> None:
    b = fn("def f(xs: list[int]) -> int:\n    s = 0\n    ys = xs[:]\n    for x in ys[:]:\n        ys[0] = 100\n"
           "        s += x\n    return s\n")["body"]
    snap = b[2]
    assert snap["kind"] == "let" and snap["name"] == "x_iter" and snap["mutable"] is False
    assert b[3]["kind"] == "forof" and b[3]["iterable"] == V("x_iter")


@pytest.mark.parametrize(("src", "message", "line"), [
    ("def f(ys: list[int]) -> int:\n    c = 0\n    for y in ys:\n        ys = [0, 0, 0]\n        c += 1\n    return c\n",
     "'ys' is reassigned inside a loop over it", 4),
    ("def f(ys: list[int]) -> int:\n    c = 0\n    for i, y in enumerate(ys):\n        ys = [0]\n    return c\n",
     "'ys' is reassigned inside a loop over it", 4),
    ("def f(c: bool) -> int:\n    ys = [1]\n    zs = [2]\n    for x in (ys if c else zs):\n        ys.append(3)\n"
     "    return 0\n", "'ys' is modified inside a loop over it", 5),
])
def test_iterated_lists_cannot_change_under_the_loop(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


# S7. Ghost variables exist only in proofs.
@pytest.mark.parametrize(("src", "line"), [
    ("def f(x: int) -> int:\n    #@ ghost g: int = 0\n    g += 1\n    return x\n", 3),
    ("def f(x: int) -> int:\n    #@ ghost ys: list[int] = [0]\n    ys.append(x)\n    return x\n", 3),
    ("def f(x: int) -> int:\n    #@ ghost ys: list[int] = [0]\n    ys[0] = 2\n    return x\n", 3),
])
def test_ghost_variables_cannot_be_changed_by_code(src: str, line: int) -> None:
    rejects(src, "ghost variable", line)


def test_annotated_empty_ghost_list() -> None:
    b = fn("def f(x: int) -> int:\n    #@ ghost ys: list[int] = []\n    return x\n")["body"]
    assert b[0]["kind"] == "ghostLet" and b[0]["tsType"] == "int[]"


# W1. An annotation right after `else:` belongs to the else branch.
def test_annotation_after_else_goes_to_the_else_branch() -> None:
    b = fn("def f(x: int) -> int:\n    if x > 0:\n        y = x\n    else:\n        #@ assert x <= 0\n        y = 0\n"
           "    return 0\n")["body"]
    assert [s["kind"] for s in b[0]["then"]] == ["let"]
    assert [s["kind"] for s in b[0]["else"]] == ["assert", "let"]


def test_annotation_after_elif_goes_to_the_elif_branch() -> None:
    b = fn("def f(x: int) -> int:\n    if x > 0:\n        y = x\n    elif x < 0:\n        #@ assert x < 0\n        y = 0\n"
           "    return 0\n")["body"]
    assert [s["kind"] for s in b[0]["then"]] == ["let"]
    assert [s["kind"] for s in b[0]["else"][0]["then"]] == ["assert", "let"]


# Diagnostics.
def test_minus_zero_index_is_index_zero() -> None:
    f = fn("def f(xs: list[int]) -> int:\n    #@ requires len(xs) > 0\n    return xs[-0]\n")
    assert f["body"][0]["value"]["idx"] == {"kind": "num", "value": 0}
