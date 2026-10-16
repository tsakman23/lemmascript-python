"""Python's floor division and modulo, including negative divisors.

Dafny's native `/` and `%` are Euclidean and disagree with Python whenever the
divisor is negative; these contracts pin down Python's behaviour.
"""


def floor_div(a: int, b: int) -> int:
    #@ requires b != 0
    #@ ensures b * result <= a < b * result + b or b * result + b < a <= b * result
    return a // b


def floor_mod(a: int, b: int) -> int:
    #@ requires b != 0
    #@ ensures b > 0 ==> 0 <= result < b
    #@ ensures b < 0 ==> b < result <= 0
    return a % b


def divmod_identity(a: int, b: int) -> bool:
    #@ requires b != 0
    #@ ensures result
    return b * (a // b) + a % b == a
