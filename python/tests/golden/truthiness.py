"""Python truthiness made explicit: an int is true iff it is non-zero."""


def nonzero(x: int) -> bool:
    return bool(x)


def first_nonzero(x: int, y: int) -> int:
    if x:
        return x
    elif not y:
        return 0
    return y


def both(x: int, y: int) -> bool:
    if x and not y:
        return False
    return x != 0 and y != 0
