"""Raises SystemExit (a BaseException, not an Exception) on an admitted input."""


def stop_at_three(x: int) -> int:
    if x == 3:
        raise SystemExit(0)
    return x
