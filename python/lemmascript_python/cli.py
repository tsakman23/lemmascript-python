"""``lsc-python extract foo.py`` — print the Raw IR envelope for ``lsc``."""
from __future__ import annotations

import json
import sys

from lemmascript_python.errors import UnsupportedError
from lemmascript_python.extract import extract_file

USAGE = "usage: lsc-python extract <file.py>"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "extract":
        print(USAGE, file=sys.stderr)
        return 1
    try:
        envelope = extract_file(args[1])
    except UnsupportedError as e:
        print(str(e), file=sys.stderr)
        return 2
    except (OSError, SyntaxError) as e:
        print(f"{args[1]}: error: {e}", file=sys.stderr)
        return 2
    json.dump(envelope, sys.stdout)
    sys.stdout.write("\n")
    return 0
