from __future__ import annotations


class UnsupportedError(Exception):
    """A construct outside the supported Python subset, with its source location.

    The frontend never approximates: anything it cannot translate faithfully
    raises this instead.
    """

    def __init__(self, message: str, line: int, col: int = 0, file: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        self.file = file

    def __str__(self) -> str:
        where = f"{self.file}:" if self.file else ""
        return f"{where}{self.line}:{self.col + 1}: error: {self.message}"
