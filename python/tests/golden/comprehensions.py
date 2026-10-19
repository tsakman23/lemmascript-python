"""Comprehensions, all/any in code, and quantifiers in specs."""


def doubled_positives(xs: list[int]) -> list[int]:
    #@ ensures len(result) <= len(xs)
    #@ ensures all(y > 0 for y in result)
    return [x * 2 for x in xs if x > 0]


def checks(xs: list[int]) -> bool:
    return all(x >= 0 for x in xs) or any(x == 7 for x in xs if x > 0)


def sorted_pairs(xs: list[int]) -> bool:
    #@ requires forall(k, 0 <= k < len(xs) - 1 ==> xs[k] <= xs[k + 1])
    #@ ensures result == all(xs[k] <= xs[k + 1] for k in range(0, len(xs) - 1))
    return True


def typed_binders(xs: list[int]) -> int:
    #@ requires forall(i: nat, i < len(xs) ==> xs[i] >= 0)
    #@ requires exists(j: nat, j < len(xs) and xs[j] == 0)
    return len(xs)
