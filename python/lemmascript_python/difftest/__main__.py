"""``python -m lemmascript_python.difftest foo.py [...]`` — print verdicts.

Exit status: 0 when every function PASSED, 1 on any ISSUE, 2 when nothing is
an ISSUE but something was NOT CHECKED.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lemmascript_python.difftest.runner import Config, Verdict, check_file


def table(rows: list[tuple[str, Verdict]]) -> str:
    head = ("file", "function", "verdict", "accepted", "detail")
    body = [(f, v.function, v.status + (f" ({v.kind})" if v.kind else ""), f"{v.accepted}/{v.tried}",
             (("counterexample: " + ", ".join(f"{k}={x!r}" for k, x in v.counterexample.items()) + "; ")
              if v.counterexample is not None else "") + v.detail)
            for f, v in rows]
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(4)]
    lines = []
    for r in [head, *body]:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r[:4], widths)) + "  " + r[4])
    return "\n".join(line.rstrip() for line in lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lemmascript_python.difftest",
                                 description="Differentially test Python functions against their Dafny model.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--max-examples", type=int, default=Config.max_examples)
    ap.add_argument("--min-accepted", type=int, default=Config.min_accepted,
                    help="distinct precondition-satisfying inputs needed for PASSED")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=Config.call_timeout,
                    help="per-input wall-clock limit in seconds (0 disables)")
    args = ap.parse_args(argv)
    cfg = Config(max_examples=args.max_examples, min_accepted=args.min_accepted, cache_dir=args.cache_dir,
                 call_timeout=args.timeout or None)
    rows: list[tuple[str, Verdict]] = []
    for f in args.files:
        rows.extend((f.name, v) for v in check_file(f, cfg))
    print(table(rows) if rows else "no verified functions found")
    statuses = {v.status for _, v in rows}
    if "ISSUE" in statuses:
        return 1
    if "NOT CHECKED" in statuses or not rows:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
