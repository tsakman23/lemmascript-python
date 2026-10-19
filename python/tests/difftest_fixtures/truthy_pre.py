"""Preconditions that are valid specs but not bool-typed body code."""


def nonzero_id(x: int) -> int:
    #@ requires x
    return x


def implied(x: int, y: int) -> int:
    #@ requires x ==> y
    return x
