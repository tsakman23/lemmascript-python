"""Binary search over a sorted list (a port of LemmaScript's examples/binarySearch.ts).

`sorted_from` is an ordinary recursive Python function, so it runs, and the
prover can also use it in contracts.
"""


def sorted_from(xs: list[int], i: int) -> bool:
    """Adjacent pairs from index i onward are non-decreasing."""
    #@ type i nat
    #@ decreases len(xs) - i
    if i + 1 >= len(xs):
        return True
    return xs[i] <= xs[i + 1] and sorted_from(xs, i + 1)


def is_sorted(xs: list[int]) -> bool:
    return sorted_from(xs, 0)


def binary_search(xs: list[int], target: int) -> int:
    #@ requires is_sorted(xs)
    #@ ensures -1 <= result < len(xs)
    #@ ensures result >= 0 ==> xs[result] == target
    #@ ensures result == -1 ==> forall(k, 0 <= k < len(xs) ==> xs[k] != target)
    lo = 0
    hi = len(xs) - 1
    while lo <= hi:
        #@ invariant 0 <= lo <= len(xs)
        #@ invariant -1 <= hi < len(xs)
        #@ invariant forall(k, 0 <= k < lo ==> xs[k] != target)
        #@ invariant forall(k, hi < k < len(xs) ==> xs[k] != target)
        #@ decreases hi - lo + 1
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        elif xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
