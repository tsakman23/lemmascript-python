"""Comparison chains, ==/!=, and/or/not, bool parameters, conditional expressions."""


def between(lo: int, x: int, hi: int) -> bool:
    return lo <= x < hi


def xor(a: bool, b: bool) -> bool:
    return (a or b) and not (a and b)


def pick(c: bool, x: int, y: int) -> int:
    return x if c and x != y else y
