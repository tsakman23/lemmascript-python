"""Python's floored `//`; the hand-edited model uses Dafny's Euclidean `/`."""


def py_div(a: int, b: int) -> int:
    #@ requires b != 0
    return a // b
