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
    #: "int" | "bool" | "none" | "list" | "any" (spec-only: a name the engine resolves)
    kind: str
    elem: PyType | None = None  # for "list"

    def __str__(self) -> str:
        if self.kind == "list":
            return f"list[{self.elem}]"
        return {"none": "None"}.get(self.kind, self.kind)

    @property
    def raw(self) -> str:
        """The Raw IR (TypeScript-syntax) type string."""
        if self.kind == "list":
            assert self.elem is not None
            return f"{self.elem.raw}[]"
        return {"int": "int", "bool": "boolean", "none": "void"}[self.kind]

    @property
    def is_list(self) -> bool:
        return self.kind == "list"

    @property
    def immutable(self) -> bool:
        """Values of this type cannot be changed in place (so sharing them is unobservable)."""
        return self.kind in ("int", "bool")


INT = PyType("int")
BOOL = PyType("bool")
NONE = PyType("none")
ANY = PyType("any")


def list_of(elem: PyType) -> PyType:
    return PyType("list", elem)


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
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id in ("list", "List") and not isinstance(node.slice, ast.Tuple)):
        return list_of(from_annotation(node.slice))
    raise UnsupportedError(f"type '{ast.unparse(node)}' is not supported yet", node.lineno, node.col_offset)


def from_type_text(text: str, line: int) -> PyType:
    """A type written inside an annotation comment (e.g. a quantifier binder)."""
    if text == "nat":
        return INT
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        raise UnsupportedError(f"invalid type '{text}'", line) from None
    for n in ast.walk(node):
        if isinstance(n, (ast.expr,)):
            n.lineno = line
            n.col_offset = 0
    return from_annotation(node)
