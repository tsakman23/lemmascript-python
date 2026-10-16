"""Raises for inputs its precondition admits."""


def checked_half(x: int) -> int:
    #@ requires x >= 0
    if x > 100:
        raise ValueError("too large")
    return x // 2
