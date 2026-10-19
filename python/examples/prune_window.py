"""Sliding-window pruning for a rate limiter.

A port of `pruneWindow` / `activeCount` from midspiral/hono-rate-limiter-with-lemmascript.
The log holds admitted request timestamps (integer ms); pruning keeps exactly
the admissions inside the window (now - w, now].
"""


def count_in(log: list[int], now: int, w: int) -> int:
    """How many timestamps fall in (now - w, now]."""
    #@ decreases len(log)
    if not log:
        return 0
    return (1 if now - w < log[0] <= now else 0) + count_in(log[1:], now, w)


def prune_window(log: list[int], now: int, w: int) -> list[int]:
    #@ requires forall(k, 0 <= k < len(log) ==> log[k] <= now)
    #@ decreases len(log)
    #@ ensures len(result) <= len(log)
    #@ ensures forall(k, 0 <= k < len(result) ==> now - w < result[k] <= now)
    #@ ensures len(result) == count_in(log, now, w)
    if not log:
        return []
    rest = prune_window(log[1:], now, w)
    if log[0] > now - w:
        return [log[0]] + rest
    return rest


def active_count(log: list[int], now: int, w: int) -> int:
    #@ requires forall(k, 0 <= k < len(log) ==> log[k] <= now)
    #@ ensures result <= len(log)
    #@ ensures result == count_in(log, now, w)
    return len(prune_window(log, now, w))
