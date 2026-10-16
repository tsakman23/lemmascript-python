"""Clamp an integer into a closed range."""


def clamp(x: int, lo: int, hi: int) -> int:
    #@ requires lo <= hi
    #@ ensures lo <= result <= hi
    #@ ensures lo <= x <= hi ==> result == x
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
