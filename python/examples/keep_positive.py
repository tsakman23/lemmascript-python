"""Building a new list with append, under the fresh-local rule."""


def keep_positive(xs: list[int]) -> list[int]:
    #@ ensures len(result) <= len(xs)
    #@ ensures all(y > 0 for y in result)
    #@ ensures all(x in result for x in xs if x > 0)
    out: list[int] = []
    for i, x in enumerate(xs):
        #@ invariant len(out) <= i
        #@ invariant all(y > 0 for y in out)
        #@ invariant forall(k, 0 <= k < i and xs[k] > 0 ==> xs[k] in out)
        if x > 0:
            out.append(x)
    return out
