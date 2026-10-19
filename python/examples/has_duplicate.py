"""Does a list contain the same value twice? (Pairwise search; sets arrive in M4.)"""


def has_duplicate(xs: list[int]) -> bool:
    #@ ensures result ==> exists(i, exists(j, 0 <= i < j < len(xs) and xs[i] == xs[j]))
    #@ ensures not result ==> forall(i, forall(j, 0 <= i < j < len(xs) ==> xs[i] != xs[j]))
    for i in range(len(xs)):
        #@ invariant forall(a, forall(b, 0 <= a < b < len(xs) and a < i ==> xs[a] != xs[b]))
        for j in range(i + 1, len(xs)):
            #@ invariant forall(a, forall(b, 0 <= a < b < len(xs) and a < i ==> xs[a] != xs[b]))
            #@ invariant forall(b, i < b < j ==> xs[i] != xs[b])
            if xs[i] == xs[j]:
                return True
    return False
