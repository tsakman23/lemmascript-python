"""Loops: while with invariants and decreases, range loops with continue and break,
enumerate, for-in, and statement-level #@ assert / #@ ghost."""


def count_down(n: int) -> int:
    #@ requires n >= 0
    #@ ensures result == 0
    while n > 0:
        #@ invariant n >= 0
        #@ decreases n
        n -= 1
    return n


def first_zero(xs: list[int]) -> int:
    #@ ensures -1 <= result < len(xs)
    #@ ensures result >= 0 ==> xs[result] == 0
    r = -1
    #@ ghost steps: int = 0
    for i in range(len(xs)):
        #@ invariant r == -1
        #@ assert 0 <= i < len(xs)
        if xs[i] == 0:
            r = i
            break
        #@ ghost steps = steps + 1
    return r


def odd_positions(xs: list[int]) -> int:
    s = 0
    for i, x in enumerate(xs):
        if i % 2 == 0:
            continue
        s += x
    for y in xs:
        s += 0 * y
    return s
