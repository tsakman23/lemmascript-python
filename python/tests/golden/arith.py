"""Integer arithmetic: + - * unary minus, ** with a literal exponent, exact big literals."""


def poly(x: int, y: int) -> int:
    return -x + 3 * y - x ** 2 + y ** 3 - (+x)


def big() -> int:
    return 2 ** 3 + 123456789012345678901234567890
