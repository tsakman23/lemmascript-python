"""Raw IR produced for the M1 subset: int/bool functions with contracts."""
from __future__ import annotations

from typing import Any

from conftest import extract, fn

V = lambda n: {"kind": "var", "name": n}  # noqa: E731
N = lambda v: {"kind": "num", "value": v}  # noqa: E731


def B(op: str, left: Any, right: Any) -> dict[str, Any]:
    return {"kind": "binop", "op": op, "left": left, "right": right}


def ret(stmt: dict[str, Any]) -> Any:
    assert stmt["kind"] == "return"
    return stmt["value"]


CLAMP = '''
def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x into [lo, hi]."""
    #@ requires lo <= hi
    #@ ensures lo <= result <= hi
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x
'''


def test_envelope_header() -> None:
    env = extract(CLAMP, "clamp.py")
    assert env["format"] == "lemmascript-raw-ir"
    assert env["version"] == 1
    assert env["source"] == "clamp.py"
    assert env["producer"]["name"] == "lemmascript-python"
    assert env["directives"] == {"backend": None, "options": {"safe-slice": True}, "leanModule": None}
    m = env["module"]
    assert m["file"] == "clamp.py"
    assert (m["typeDecls"], m["externs"], m["constants"], m["classes"]) == ([], [], [], [])
    assert {"clamp", "x", "lo", "hi"} <= set(env["identifiers"])


def test_signature_and_contracts() -> None:
    f = fn(CLAMP)
    assert f["name"] == "clamp" and f["exported"] is True
    assert f["params"] == [{"name": "x", "tsType": "int"}, {"name": "lo", "tsType": "int"}, {"name": "hi", "tsType": "int"}]
    assert f["tsParams"] == [{"kind": "simple", "binds": [p]} for p in ("x", "lo", "hi")]
    assert f["returnType"] == "int"
    assert f["requires"] == [B("<=", V("lo"), V("hi"))]
    # Chained comparison splits into a conjunction.
    assert f["ensures"] == [B("&&", B("<=", V("lo"), {"kind": "result"}), B("<=", {"kind": "result"}, V("hi")))]
    assert f["decreases"] is None and f["pure"] is False and f["typeAnnotations"] == []


def test_if_elif_else() -> None:
    body = fn(CLAMP)["body"]
    assert len(body) == 1 and body[0]["kind"] == "if"
    assert body[0]["cond"] == B("<", V("x"), V("lo"))
    inner = body[0]["else"]
    assert len(inner) == 1 and inner[0]["kind"] == "if"
    assert inner[0]["cond"] == B(">", V("x"), V("hi"))
    assert ret(inner[0]["else"][0]) == V("x")


def test_private_names_are_not_exported() -> None:
    assert fn("def _h(x: int) -> int:\n    return x\n")["exported"] is False


def test_bool_type_and_operators() -> None:
    f = fn("def f(a: bool, b: bool, x: int, y: int) -> bool:\n    return (a and not b) or x == y or x != y\n")
    assert f["params"][0]["tsType"] == "boolean" and f["returnType"] == "boolean"
    assert ret(f["body"][0]) == B("||", B("||", B("&&", V("a"), {"kind": "unop", "op": "!", "expr": V("b")}),
                                         B("===", V("x"), V("y"))), B("!==", V("x"), V("y")))


def test_floor_division_and_modulo() -> None:
    f = fn("def f(a: int, b: int) -> int:\n    #@ requires b != 0\n    return a // b + a % b\n")
    assert ret(f["body"][0]) == B("+", B("floordiv", V("a"), V("b")), B("floormod", V("a"), V("b")))


def test_unary_and_power() -> None:
    f = fn("def f(a: int) -> int:\n    return -a + (+a) + a ** 2 + a ** 0\n")
    assert ret(f["body"][0]) == B("+", B("+", B("+", {"kind": "unop", "op": "-", "expr": V("a")}, V("a")),
                                         B("*", V("a"), V("a"))), N(1))


def test_builtins_abs_min_max() -> None:
    f = fn("def f(a: int, b: int) -> int:\n    return abs(a) + min(a, b) + max(a, b)\n")
    M = lambda m, *args: {"kind": "call", "fn": {"kind": "field", "obj": V("Math"), "field": m}, "args": list(args)}  # noqa: E731
    assert ret(f["body"][0]) == B("+", B("+", M("abs", V("a")), M("min", V("a"), V("b"))), M("max", V("a"), V("b")))


def test_locals_assign_and_augassign() -> None:
    f = fn("def f(a: int) -> int:\n    r = a\n    t: int = 1\n    r += t\n    r //= 2\n    return r\n")
    b = f["body"]
    assert b[0] == {"kind": "let", "name": "r", "mutable": True, "tsType": "int", "init": V("a"), "line": 2}
    assert b[1] == {"kind": "let", "name": "t", "mutable": False, "tsType": "int", "init": N(1), "line": 3}
    assert b[2] == {"kind": "assign", "target": "r", "value": B("+", V("r"), V("t")), "line": 4}
    assert b[3] == {"kind": "assign", "target": "r", "value": B("floordiv", V("r"), N(2)), "line": 5}


def test_parameter_rebinding_is_an_assignment() -> None:
    b = fn("def f(a: int) -> int:\n    a = a + 1\n    return a\n")["body"]
    assert b[0] == {"kind": "assign", "target": "a", "value": B("+", V("a"), N(1)), "line": 2}


def test_int_truthiness_is_explicit() -> None:
    f = fn("def f(x: int) -> bool:\n    if x:\n        return not x\n    return x != 0 and bool(x)\n")
    assert f["body"][0]["cond"] == B("!==", V("x"), N(0))
    assert ret(f["body"][0]["then"][0]) == B("===", V("x"), N(0))
    assert ret(f["body"][1]) == B("&&", B("!==", V("x"), N(0)), B("!==", V("x"), N(0)))


def test_int_operands_of_and_in_condition_position() -> None:
    f = fn("def f(x: int, y: int) -> int:\n    if x and y:\n        return 1\n    return 0\n")
    assert f["body"][0]["cond"] == B("&&", B("!==", V("x"), N(0)), B("!==", V("y"), N(0)))


def test_conditional_expression() -> None:
    f = fn("def f(x: int) -> int:\n    return x if x >= 0 else -x\n")
    assert ret(f["body"][0]) == {"kind": "conditional", "cond": B(">=", V("x"), N(0)), "then": V("x"),
                                 "else": {"kind": "unop", "op": "-", "expr": V("x")}}


def test_raise_and_assert_are_unreachable_throws() -> None:
    b = fn("def f(x: int) -> int:\n    assert x > 0\n    if x > 10:\n        raise ValueError('big')\n    return x\n")["body"]
    assert b[0] == {"kind": "if", "cond": {"kind": "unop", "op": "!", "expr": B(">", V("x"), N(0))},
                    "then": [{"kind": "throw", "line": 2}], "else": [], "line": 2}
    assert b[1]["then"] == [{"kind": "throw", "line": 4}]


def test_calls_to_module_functions() -> None:
    src = "def sq(x: int) -> int:\n    return x * x\n\ndef f(a: int) -> int:\n    return sq(a) + sq(2)\n"
    f = fn(src, "f")
    C = lambda *a: {"kind": "call", "fn": V("sq"), "args": list(a)}  # noqa: E731
    assert ret(f["body"][0]) == B("+", C(V("a")), C(N(2)))


def test_large_int_literals_are_exact() -> None:
    f = fn("def f() -> int:\n    return 9007199254740993\n")
    assert ret(f["body"][0]) == {"kind": "bigint", "value": "9007199254740993"}


def test_implication_and_iff_in_specs() -> None:
    f = fn("def f(x: int) -> int:\n"
           "    #@ ensures x > 0 ==> result > 0\n"
           "    #@ ensures (x == 0 ==> result == 0) and (result >= 0 <==> x >= 0)\n"
           "    #@ ensures x < 0 ==> x < 1 ==> result < 0\n"
           "    return x\n")
    R = {"kind": "result"}
    assert f["ensures"][0] == B("==>", B(">", V("x"), N(0)), B(">", R, N(0)))
    assert f["ensures"][1] == B("&&", B("==>", B("===", V("x"), N(0)), B("===", R, N(0))),
                                B("<==>", B(">=", R, N(0)), B(">=", V("x"), N(0))))
    # right-associative
    assert f["ensures"][2] == B("==>", B("<", V("x"), N(0)), B("==>", B("<", V("x"), N(1)), B("<", R, N(0))))


def test_contracts_may_precede_the_docstring() -> None:
    f = fn('def f(x: int) -> int:\n    #@ requires x > 0\n    """Doc."""\n    #@ ensures result > 0\n    return x\n')
    assert f["requires"] == [B(">", V("x"), N(0))] and f["ensures"] == [B(">", {"kind": "result"}, N(0))]


def test_type_and_pure_annotations() -> None:
    f = fn("def f(x: int) -> int:\n    #@ pure\n    #@ type x nat\n    return x\n")
    assert f["pure"] is True and f["typeAnnotations"] == [{"name": "x", "type": "nat"}]


def test_verify_opt_in_mode() -> None:
    src = ("def a(x: int) -> int:\n    return x\n\n"
           "def b(x: int) -> int:\n    #@ verify\n    return x + 1\n")
    assert [f["name"] for f in extract(src)["module"]["functions"]] == ["b"]


def test_file_directives() -> None:
    env = extract("#@ backend dafny\n#@ option safe-slice true\n\ndef f(x: int) -> int:\n    return x\n")
    assert env["directives"] == {"backend": "dafny", "options": {"safe-slice": True}, "leanModule": None}


def test_main_guard_and_typing_imports_are_ignored() -> None:
    src = ("from __future__ import annotations\nimport typing\nfrom typing import Final\n\n"
           "def f(x: int) -> int:\n    return x\n\n"
           "if __name__ == '__main__':\n    print(f(1))\n")
    assert [f["name"] for f in extract(src)["module"]["functions"]] == ["f"]
