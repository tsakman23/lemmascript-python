"""Opt-in mode: once any function has #@ verify, only marked functions are emitted."""
#@ backend dafny


def unverified(x: int) -> int:
    return x * 1000


def verified(x: int) -> int:
    #@ verify
    #@ ensures result == x + 1
    return x + 1
