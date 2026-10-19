"""Names the frontend refuses as user identifiers.

The engine resolves some names specially (``undefined`` is ``None``, ``Math``
holds ``abs``/``min``/``max``, ``this`` is a receiver), and the Dafny preamble
defines helper functions in the same namespace as user code, so a user
function or variable with one of these names would silently resolve to the
engine's meaning instead of the Python one.
"""
from __future__ import annotations

ENGINE_NAMES = frozenset({"undefined", "null", "this", "Math", "Std", "Option", "Some", "None", "Unknown"})

PREAMBLE_NAMES = frozenset({
    "BitAnd", "BitOr", "CeilReal", "FloorDiv", "FloorMod", "FloorReal", "IntToString", "IsJSWhitespace",
    "JSFloorDiv", "JSRem", "JSStringLt", "JSTruncDiv", "MathAbs", "MathMax", "MathMin", "MaxOfSeq",
    "MaxOfSeqConcat", "MinOfSeq", "MinOfSeqConcat", "NatToString", "NormalizeSliceIndex", "Perm", "Pow2",
    "SafeSlice", "SeqAll", "SeqFilter", "SeqFilterSome", "SeqFind", "SeqFindIndex", "SeqFindIndexFrom",
    "SeqFindLast", "SeqFindLastIndex", "SeqFlatten", "SeqFoldLeft", "SeqIndexOf", "SeqIndexOfFrom",
    "SeqJoin", "SeqSort", "SeqSortBy", "SetFromSeq", "SetToSeq", "StringFromCharCode", "StringIndexOf",
    "StringIndexOfFrom", "StringIndexOfFromN", "StringRepeat", "StringSplit", "StringToLower",
    "StringToUpper", "StringTrim", "StringTrimLeft", "StringTrimRight",
})

RESERVED = ENGINE_NAMES | PREAMBLE_NAMES

#: Prefix of the spec parser's internal marker names.
MARKER_PREFIX = "__lsc_"


def reserved_reason(name: str) -> str | None:
    if name in RESERVED:
        return f"'{name}' is reserved by the LemmaScript engine; rename it"
    if name.startswith(MARKER_PREFIX):
        return f"names starting with '{MARKER_PREFIX}' are reserved"
    return None
