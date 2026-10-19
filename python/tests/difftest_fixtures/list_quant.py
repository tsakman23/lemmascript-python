"""A quantified (sortedness) precondition: the Python reading is enumerated
over the bounded index range; the Dafny-side agreement check is skipped."""


def smallest(xs: list[int]) -> int:
    #@ requires len(xs) > 0
    #@ requires forall(i, 0 <= i < len(xs) - 1 ==> xs[i] <= xs[i + 1])
    #@ ensures forall(k, 0 <= k < len(xs) ==> result <= xs[k])
    return xs[0]
