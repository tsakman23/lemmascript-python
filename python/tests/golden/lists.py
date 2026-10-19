"""List expressions: len, indexing (negative literals count from the end), clamping slices,
concatenation, membership, equality, truthiness, min/max of a list."""


def ends(xs: list[int]) -> int:
    #@ requires len(xs) >= 2
    return xs[0] + xs[-1] + xs[len(xs) - 2]


def middle(xs: list[int], a: int) -> list[int]:
    #@ ensures len(result) <= len(xs) + len(xs) + 1
    return xs[a:] + xs[:-1] + [a]


def member(xs: list[int], x: int) -> bool:
    return x in xs and x not in [0] and (xs == [x] or xs != [])


def nonempty(xs: list[int]) -> int:
    if xs:
        return min(xs) + max(xs)
    return 0
