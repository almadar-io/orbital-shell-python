"""
Row order of a collection fetch (EXEC-FETCH-1). Twins: `@almadar/core` row-query.test.ts, `orbital-core` row_query_pushdown.rs.
"""
import pytest
from orbital_app.core.effect_executor import EffectExecutor
from orbital_app.core.repository import InMemoryRepository
from orbital_app.core.row_order import order_rows, parse_fetch_order

ROWS = [
    {"id": "a", "createdAt": "2026-03-02T00:00:00.000Z"},
    {"id": "b"},
    {"id": "c", "createdAt": "2026-03-01T00:00:00.000Z"},
    {"id": "d", "createdAt": "2026-03-02T00:00:00.000Z"},
    {"id": "e", "createdAt": None},
]


def ids(rows):
    return [r["id"] for r in rows]


def test_default_order_is_created_at_then_id_with_missing_last():
    assert ids(order_rows(ROWS)) == ["c", "a", "d", "b", "e"]


def test_control_an_authored_order_replaces_the_default():
    assert ids(order_rows(ROWS, parse_fetch_order("id:desc"))) == ["e", "d", "c", "b", "a"]


def test_mixed_kinds_have_one_total_order_number_then_string():
    rows = [{"id": "a", "createdAt": "2026-03-01T00:00:00.000Z"}, {"id": "b", "createdAt": 5}, {"id": "c", "createdAt": 3}]
    assert ids(order_rows(rows)) == ["c", "b", "a"]


def test_does_not_mutate_its_input():
    rows = list(ROWS)
    order_rows(rows)
    assert rows == ROWS


@pytest.mark.asyncio
async def test_an_unordered_fetch_effect_is_ordered_by_the_default():
    repository = InMemoryRepository()
    for row in ROWS:
        await repository.create("Note", dict(row))
    executor = EffectExecutor(repository)
    assert ids(await executor.execute(["fetch", "Note", {}], {})) == ["c", "a", "d", "b", "e"]
    assert ids(await executor.execute(["fetch", "Note", {"orderBy": "id:desc"}], {})) == ["e", "d", "c", "b", "a"]
