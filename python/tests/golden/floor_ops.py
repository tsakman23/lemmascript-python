"""Python's floored // and %, in code and in specs."""


def qr(a: int, b: int) -> int:
    #@ requires b != 0
    #@ ensures result == a // b * b + a % b
    q = a // b
    r = a % b
    q *= b
    q += r
    r //= 1
    r %= 7
    return q
