"""Summing a list with a loop, proven equal to a recursive definition."""


def sum_to(xs: list[int], n: int) -> int:
    """The sum of the first n elements."""
    #@ requires 0 <= n <= len(xs)
    #@ decreases n
    if n == 0:
        return 0
    return sum_to(xs, n - 1) + xs[n - 1]


def total(xs: list[int]) -> int:
    #@ ensures result == sum_to(xs, len(xs))
    s = 0
    for i in range(len(xs)):
        #@ invariant s == sum_to(xs, i)
        s += xs[i]
    return s
