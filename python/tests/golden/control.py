"""Locals, rebinding, if/elif/else, raise and assert as unreachable points, calls."""


def helper(x: int) -> int:
    #@ requires x >= 0
    #@ ensures result >= x
    return x + 1


def steps(n: int) -> int:
    #@ requires 0 <= n <= 100
    assert n >= 0
    if n > 100:
        raise ValueError("too big")
    total: int = 0
    total = helper(n)
    n = n + 1
    if total > n:
        raise AssertionError("unreachable")
    return abs(total) + min(n, total, 3) + max(n, 1)
