"""Python's floored `%`; the hand-edited model uses Dafny's Euclidean `%`."""


def py_mod(a: int, b: int) -> int:
    #@ requires b != 0
    return a % b
