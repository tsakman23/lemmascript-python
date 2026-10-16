"""A precondition that rejects nearly every input."""


def needle(x: int, y: int) -> int:
    #@ requires x == 12345 and y == 678
    #@ ensures result == x + y
    return x + y
