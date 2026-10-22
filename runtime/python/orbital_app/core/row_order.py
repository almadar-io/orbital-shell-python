"""
Row order of a collection fetch (EXEC-FETCH-1).

A fetch with no `orderBy` is ordered by `createdAt` ascending, a row without one last, ties by `id` ascending.
This is the in-process twin of `@almadar/core` `orderRows` and `orbital-core` `compare_rows`; the Python shell lists
the collection and orders it here, so the default must match theirs.
"""
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Tuple

Row = Dict[str, Any]
Order = Tuple[str, str]

DEFAULT_ORDER: Order = ("createdAt", "asc")


def parse_fetch_order(spec: Optional[str]) -> Optional[Order]:
    """`"field"` or `"field:asc|desc"` as (field, direction); None for no order."""
    if not isinstance(spec, str) or spec == "":
        return None
    field, _, direction = spec.partition(":")
    return (field, "desc" if direction == "desc" else "asc")


def _missing(value: Any) -> bool:
    return value is None


def _rank(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return 1
    if isinstance(value, str):
        return 2
    return 3


def _compare_values(a: Any, b: Any) -> int:
    if _rank(a) != _rank(b):
        return -1 if _rank(a) < _rank(b) else 1
    if _rank(a) == 3 or a == b:
        return 0
    # Strings compare by code point, as the other paths do.
    return -1 if a < b else 1


def _compare_by_field(a: Row, b: Row, field: str, direction: str) -> int:
    x, y = a.get(field), b.get(field)
    if _missing(x) or _missing(y):
        if _missing(x) and _missing(y):
            return 0
        return 1 if _missing(x) else -1
    c = _compare_values(x, y)
    return c if direction == "asc" else -c


def order_rows(rows: List[Row], order: Optional[Order] = None) -> List[Row]:
    """Rows ordered by `order` (the default order when None), ties by `id` ascending; the input is not mutated."""
    field, direction = order if order is not None else DEFAULT_ORDER

    def compare(a: Row, b: Row) -> int:
        return _compare_by_field(a, b, field, direction) or _compare_by_field(a, b, "id", "asc")

    return sorted(rows, key=cmp_to_key(compare))
