"""The model agrees with Python, but the postcondition is false for every input."""


def ident(x: int) -> int:
    #@ ensures result > x
    return x
