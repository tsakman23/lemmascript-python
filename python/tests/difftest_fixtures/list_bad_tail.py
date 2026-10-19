"""`xs[1:]`; the hand-edited model drops two elements instead of one."""


def tail(xs: list[int]) -> list[int]:
    #@ requires len(xs) > 0
    return xs[1:]
