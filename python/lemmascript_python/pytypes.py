"""Types of the supported subset and their Raw IR spellings.

Raw IR carries types as TypeScript type strings, parsed by the engine's
``parseTsType``. Python ``int`` maps to the engine's ``int`` (an unbounded
integer, exactly Python's semantics) rather than TS ``number``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from lemmascript_python.errors import UnsupportedError


@dataclass(frozen=True)
class PyType:
    kind: str  # "int" | "bool" | "none" | "any" (spec-only: a name the engine resolves)

    def __str__(self) -> str:
        return {"none": "None"}.get(self.kind, self.kind)

    @property
    def raw(self) -> str:
        """The Raw IR (TypeScript-syntax) type string."""
        return {"int": "int", "bool": "boolean", "none": "void"}[self.kind]


INT = PyType("int")
BOOL = PyType("bool")
NONE = PyType("none")
ANY = PyType("any")


def from_annotation(node: ast.expr) -> PyType:
    if isinstance(node, ast.Name):
        if node.id == "int":
            return INT
        if node.id == "bool":
            return BOOL
        if node.id == "float":
            raise UnsupportedError(
                "'float' is not supported: Python floats round, so a proof about exact numbers "
                "would not hold for them (use int, and '//' for division)", node.lineno, node.col_offset)
    if isinstance(node, ast.Constant) and node.value is None:
        return NONE
    raise UnsupportedError(f"type '{ast.unparse(node)}' is not supported yet", node.lineno, node.col_offset)
