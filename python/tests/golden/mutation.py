"""In-place list updates on fresh locals: append, extend, item assignment, +=."""


def build(n: int, xs: list[int]) -> list[int]:
    #@ requires n >= 0
    #@ ensures len(result) == n + 2 * len(xs) + 1
    out: list[int] = []
    for i in range(n):
        #@ invariant len(out) == i
        out.append(i)
    copy = xs[:]
    out.extend(copy)
    out += list(xs)
    out += [0]
    out[0 if n == 0 else n - 1] = 9
    out[-1] += 1
    return out
