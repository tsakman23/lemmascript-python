"""Regression tests for the findings of the adversarial M1 soundness review.

Each case was confirmed to verify in Dafny while the Python behaved differently
(or to be silently mis-translated). Every one must now be rejected.
"""
from __future__ import annotations

import pytest

from conftest import extract
from lemmascript_python import specs
from lemmascript_python.errors import UnsupportedError


def rejects(src: str, message: str, line: int | None = None) -> None:
    with pytest.raises(UnsupportedError) as exc:
        extract(src)
    assert message in str(exc.value), str(exc.value)
    if line is not None:
        assert exc.value.line == line, exc.value


# 1. A name assigned anywhere in a function is local to the whole function.
@pytest.mark.parametrize(("src", "line"), [
    ("def f(x: int) -> int:\n    if x > 100:\n        abs = 0\n    return abs(x)\n", 4),
    ("def g(x: int) -> int:\n    return x\n\ndef h(x: int) -> int:\n    y = g(x)\n    g = 3\n    return y\n", 5),
    ("def f(min: int) -> int:\n    return min(min, 2)\n", 2),
    ("def f(xs: list[int]) -> int:\n    for len in xs:\n        pass\n    return len(xs)\n", 4),
])
def test_calls_to_local_names_are_rejected(src: str, line: int) -> None:
    rejects(src, "is a local variable", line)


# 2. Imports may not rebind names.
@pytest.mark.parametrize(("src", "message", "line"), [
    ("from typing import cast as min\ndef f(a: int, b: int) -> int:\n    return min(a, b)\n",
     "import aliases ('as') are not supported", 1),
    ("import typing as abs\ndef f(a: int) -> int:\n    return abs(a)\n", "import aliases ('as') are not supported", 1),
    ("from typing import *\ndef f(a: int) -> int:\n    return a\n", "'from typing import *' is not supported", 1),
    ("def g(x: int) -> int:\n    return x\nfrom typing import Final\n",
     "imports must come before the first function", 3),
    ("from typing import cast\ndef f(a: int) -> int:\n    return a\n", "'cast' cannot be imported", 1),
])
def test_imports_cannot_rebind_names(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


def test_plain_typing_imports_are_fine() -> None:
    extract("from __future__ import annotations\nimport typing\nfrom typing import Final, Optional\n\n"
            "def f(a: int) -> int:\n    return a\n")


# 3. The __main__ block runs only as a script, but must not change what verified code calls.
@pytest.mark.parametrize("body", [
    "    def g(x): return -1\n",
    "    g = lambda x: -1\n",
    "    abs = print\n",
    "    import typing\n",
    "    global g\n",
    "    for g in [1]:\n        pass\n",
])
def test_main_block_cannot_rebind_names(body: str) -> None:
    src = ("def g(x: int) -> int:\n    return 1\n\ndef f(x: int) -> int:\n    return g(x)\n\n"
           "if __name__ == '__main__':\n" + body)
    rejects(src, "the '__main__' block may not bind", 8)


def test_main_block_may_call_and_print() -> None:
    extract("def f(x: int) -> int:\n    return x\n\nif __name__ == '__main__':\n    r = f(1)\n    print(r)\n")


# 5. Spec calls must not silently bind to something other than the Python function.
@pytest.mark.parametrize(("src", "message", "line"), [
    ("def g(x: int) -> int:\n    return -1\n\ndef f(x: int) -> int:\n    #@ verify\n    #@ ensures result == g(x)\n"
     "    return x\n", "calls 'g', which is not marked '#@ verify'", 6),
    ("def f(x: int) -> int:\n    #@ ensures result == sorted(x)\n    return x\n",
     "'sorted' is not supported in specs", 2),
    ("def f(b: bool) -> int:\n    #@ ensures result == int(b)\n    return 0\n", "int(...) is supported on an 'int'", 2),
    ("def f(x: int) -> int:\n    #@ ensures result == min(result)\n    return x\n", "min() of 'int' is not supported", 2),
    ("def f(b: bool) -> int:\n    return int(b)\n", "int(...) is supported on an 'int'", 2),
    ("def f(x: int, y: int) -> int:\n    return abs(x, y)\n", "abs() takes one argument", 2),
])
def test_builtins_and_unverified_calls_in_specs(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


# 6 and 10. Annotations and names that were silently dropped or mis-emitted.
@pytest.mark.parametrize(("src", "message", "line"), [
    ("def f(x: int) -> int:\n    #@ invariant False\n    return x\n", "belongs at the start of a loop body", 2),
    ("def f(x: int) -> int:\n    #@ done_with False\n    return x\n", "belongs at the start of a loop body", 2),
    ("def f(x: int) -> int:\n    #@ option safe-slice true\n    return x\n",
     "must come before the first statement of the file", 2),
    ("def f(x: int) -> int:\n    #@ type y nat\n    return x\n", "'#@ type y': no parameter or local named 'y'", 2),
    ("def f(x: int) -> int:\n    #@ type result nat\n    return x\n", "'#@ type result'", 2),
    ("def f(x: int) -> int:\n    #@ requires x > 0  #@ ensures result < 0\n    return x\n",
     "one annotation per line", 2),
    ("def f(x: int) -> int:\n    return x\n\ndef f(x: int) -> int:\n    return -x\n", "'f' is defined twice", 4),
    ("def MathAbs(x: int) -> int:\n    return x\n", "'MathAbs' is reserved", 1),
    ("def f(FloorDiv: int) -> int:\n    return FloorDiv // 2\n", "'FloorDiv' is reserved", 1),
    ("def f(undefined: int) -> int:\n    return undefined\n", "'undefined' is reserved", 1),
    ("def f(x: int) -> int:\n    this = x\n    return this\n", "'this' is reserved", 2),
    ("def f(x: int) -> int:\n    #@ pure\n    assert x > 0\n    return x\n",
     "'#@ pure' is not supported on a function that can raise", 2),
    ("#@ backend lean\ndef f(x: int) -> int:\n    return x\n", "supports only the Dafny backend", 1),
    ("def f(x: int) -> int:\n    #@ ensures __lsc_implies__(x > 0, x > 1, x > 2)\n    return x\n",
     "names starting with '__lsc_' are reserved", 2),
])
def test_annotation_and_name_findings(src: str, message: str, line: int) -> None:
    rejects(src, message, line)


def test_type_annotation_on_a_local_is_accepted() -> None:
    f = extract("def f(n: int) -> int:\n    #@ type i nat\n    i = 0\n    return i + n\n")["module"]["functions"][0]
    assert f["typeAnnotations"] == [{"name": "i", "type": "nat"}]


# 11. A plain trailing comment after a spec is ignored, even after `==>`.
def test_trailing_comment_after_a_spec() -> None:
    f = extract("def f(x: int) -> int:\n    #@ ensures x > 0 ==> result > 0  # positive in, positive out\n"
                "    return x\n")["module"]["functions"][0]
    assert f["ensures"][0]["op"] == "==>"


# 8 (frontend half). `==>` renders as a bool in executable Python.
def test_implication_renders_as_bool() -> None:
    assert specs.to_python(specs.parse("a ==> b", 1)) == "not a or bool(b)"
