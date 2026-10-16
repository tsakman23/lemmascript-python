"""Unsupported constructs fail with a clear, located error — never approximated."""
from __future__ import annotations

import pytest

from conftest import extract
from lemmascript_python.errors import UnsupportedError


def rejects(src: str, message: str, line: int | None = None) -> None:
    with pytest.raises(UnsupportedError) as exc:
        extract(src)
    assert message in str(exc.value), str(exc.value)
    if line is not None:
        assert exc.value.line == line, exc.value


@pytest.mark.parametrize(("src", "message", "line"), [
    ("def f(x) -> int:\n    return 1\n", "parameter 'x' needs a type annotation", 1),
    ("def f(x: int):\n    return x\n", "'f' needs a return type annotation", 1),
    ("def f(x: float) -> int:\n    return 1\n", "'float' is not supported: Python floats round", 1),
    ("def f(x: int) -> int:\n    return x / 2\n", "'/' produces a float", 2),
    ("def f(x: int = 1) -> int:\n    return x\n", "default parameter values are not supported", 1),
    ("def f(*xs: int) -> int:\n    return 1\n", "'*args'/'**kwargs' parameters are not supported", 1),
    ("def f(x: int, *, y: int) -> int:\n    return x\n", "keyword-only parameters are not supported", 1),
    ("def f(x: list[int]) -> int:\n    return 1\n", "type 'list[int]' is not supported yet", 1),
    ("def f(x: bool) -> int:\n    return x + 1\n", "arithmetic on 'bool' is not supported", 2),
    ("def f(x: int, b: bool) -> bool:\n    return x == b\n", "'==' between 'int' and 'bool'", 2),
    ("def f(x: int) -> bool:\n    return x < True\n", "'<' between 'int' and 'bool'", 2),
    ("def f(x: int) -> int:\n    return x and 1\n", "'and' on non-bool operands in a value position", 2),
    ("def f(x: int) -> int:\n    if x > 0:\n        return 1\n", "may reach the end without returning", 1),
    ("def f(x: int) -> int:\n    if x > 0:\n        y = 1\n    else:\n        y = 2\n    return y\n",
     "'y' is used outside the block that first assigns it", 6),
    ("def f(x: int) -> int:\n    return z\n", "unknown name 'z'", 2),
    ("def f(x: int) -> int:\n    return g(x)\n", "unknown function 'g'", 2),
    ("def f(x: int) -> int:\n    return x is 1\n", "'is' is only supported against None", 2),
    ("def f(x: int) -> int:\n    for i in range(x):\n        pass\n    return x\n", "'for' loops are not supported yet", 2),
    ("def f(x: int) -> int:\n    while x > 0:\n        x -= 1\n    return x\n", "'while' loops are not supported yet", 2),
    ("def f(x: int) -> int:\n    try:\n        return x\n    except Exception:\n        return 0\n", "'try' is not supported", 2),
    ("def f(x: int) -> int:\n    def g() -> int:\n        return 1\n    return x\n", "nested functions are not supported", 2),
    ("def f(x: int) -> int:\n    global y\n    return x\n", "'global' is not supported", 2),
    ("def f(x: int) -> int:\n    print(x)\n    return x\n", "expression statements are not supported", 2),
    ("class C:\n    pass\n", "classes are not supported yet", 1),
    ("x = 1\n", "top-level statement is not supported", 1),
    ("import os\n", "import of 'os' is not supported", 1),
    ("@staticmethod\ndef f(x: int) -> int:\n    return x\n", "decorators are not supported", 1),
    ("async def f(x: int) -> int:\n    return x\n", "async functions are not supported", 1),
    ("def f(x: int) -> int:\n    return 0 < g(x) < 2\n\ndef g(x: int) -> int:\n    return x\n",
     "the middle of a chained comparison must be a name or literal", 2),
    ("def f(x: int) -> int:\n    return x ** -1\n", "'**' needs a literal exponent between 0 and 4", 2),
    ("def f(x: int) -> int:\n    return f(x=x)\n", "keyword arguments are not supported", 2),
    ("def f(x: int) -> int:\n    x: int = 2\n    return x\n", "'x' is already bound", 2),
])
def test_rejects(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


@pytest.mark.parametrize(("src", "message", "line"), [
    ("def f(x: int) -> int:\n    return x\n    #@ ensures result > 0\n", "'#@ ensures' must come before the first statement", 3),
    ("def f(x: int) -> int:\n    #@ frobnicate x\n    return x\n", "unknown annotation '#@ frobnicate'", 2),
    ("def f(x: int) -> int:\n    #@ requires result > 0\n    return x\n", "'result' is only valid in '#@ ensures'", 2),
    ("def f(result: int) -> int:\n    #@ ensures result > 0\n    return result\n",
     "parameter named 'result' clashes with the return value", 1),
    ("def f(x: int) -> int:\n    #@ requires x >\n    return x\n", "invalid spec expression", 2),
    ("def f(x: int) -> int:\n    #@ requires x / 2 > 0\n    return x\n", "'/' produces a float", 2),
    ("#@ verify\ndef f(x: int) -> int:\n    return x\n", "'#@ verify' belongs inside a function body", 1),
    ("def f(x: int) -> int:\n    return x\n\n#@ requires x > 0\n", "stray '#@' annotation", 4),
    ("#@ option no-such-option true\ndef f(x: int) -> int:\n    return x\n", "unknown option 'no-such-option'", 1),
    ("#@ option string-semantics javascript-utf16\ndef f(x: int) -> int:\n    return x\n",
     "'string-semantics javascript-utf16' does not model Python strings", 1),
])
def test_rejects_bad_annotations(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


def test_verify_mode_rejects_calls_to_unverified_functions() -> None:
    src = ("def g(x: int) -> int:\n    return x\n\n"
           "def f(x: int) -> int:\n    #@ verify\n    return g(x)\n")
    rejects(src, "'f' calls 'g', which is not marked '#@ verify'", 6)
