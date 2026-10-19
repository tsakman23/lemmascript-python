"""List functions with a faithful (generated) model; also the target of the
argument-mutation test, which swaps in a Python `tail` that clears `xs`."""


def head(xs: list[int]) -> int:
    #@ requires len(xs) > 0
    return xs[0]


def tail(xs: list[int]) -> list[int]:
    #@ requires len(xs) > 0
    #@ ensures len(result) == len(xs) - 1
    return xs[1:]


def first_of_first(xss: list[list[int]]) -> int:
    #@ requires len(xss) > 0 and len(xss[0]) > 0
    return xss[0][0]
