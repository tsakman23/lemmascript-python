"""A faithful model; the test injects a Dafny-side precondition that disagrees."""


def positive_id(x: int) -> int:
    #@ requires x > 0
    #@ ensures result == x
    return x
