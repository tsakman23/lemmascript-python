"""Spec-only syntax: ==>, <==>, result, chains, and #@ pure / #@ type."""


def sgn(x: int) -> int:
    #@ pure
    #@ ensures x > 0 ==> result == 1
    #@ ensures x < 0 ==> result == -1
    #@ ensures (result == 0) <==> (x == 0)
    #@ ensures -1 <= result <= 1
    return 1 if x > 0 else (-1 if x < 0 else 0)


def countdown(n: int) -> int:
    #@ type n nat
    #@ ensures result >= 0 and result == n
    return n
