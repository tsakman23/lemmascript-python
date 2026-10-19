"""Executable contract readings used by the differential harness."""
from __future__ import annotations

from lemmascript_python.contracts import load_source


def test_implication_becomes_executable_python() -> None:
    c = load_source("def f(x: int) -> int:\n    #@ requires x > 0 ==> x < 9\n    #@ ensures result == x\n    return x\n")["f"]
    assert c.requires == ["not x > 0 or bool(x < 9)"]
    assert c.ensures == ["result == x"]
    assert c.params == [("x", "int")] and c.returns == "int"


def test_nat_parameter_is_an_implicit_precondition() -> None:
    # The Dafny parameter is a nat, so the proof says nothing about n < 0.
    c = load_source("def f(n: int, m: int) -> int:\n    #@ type n nat\n    #@ requires m > 0\n    return n\n")["f"]
    assert c.requires == ["n >= 0", "m > 0"]
