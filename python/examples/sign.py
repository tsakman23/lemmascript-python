"""Sign, absolute value and the maximum of three integers."""


def sign(x: int) -> int:
    #@ ensures -1 <= result <= 1
    #@ ensures (result == 0) == (x == 0)
    #@ ensures result * x == abs(x)
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def abs_val(x: int) -> int:
    #@ ensures result >= 0
    #@ ensures result == x or result == -x
    return x if x >= 0 else -x


def max3(a: int, b: int, c: int) -> int:
    #@ ensures result >= a and result >= b and result >= c
    #@ ensures result == a or result == b or result == c
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    return m


def is_even(n: int) -> bool:
    #@ ensures result == (n % 2 == 0)
    return not n % 2
