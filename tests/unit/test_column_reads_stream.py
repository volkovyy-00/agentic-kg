"""Peak-memory shape tests for the four distinct-value column checks.

These measure a SHAPE, not an absolute: peak Python allocation at 4x the rows,
with the distinct-value count held constant, must not grow. Measured on
2026-09-20 on this harness: the list-based implementation grows 3.08x-3.97x
across these scenarios and a streaming one stays at or below 1.00x, so 1.5 sits
clear of both and does not need re-tuning on another machine. The file takes
roughly 10 s on its own, and roughly 29 s under CI's `--cov` run.

tracemalloc counts Python allocations deterministically, unlike RSS, so this
does not flake. The sources are real memory:// files carrying the right header
-- the readers call source_exists, and the reachability check reads headers
through read_csv_header -- with only the row stream fabricated.
"""

import gc
import tracemalloc

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import file_tools
from agentic_kg.tools import reference_reachability as rr

SMALL_ROWS = 10_000
LARGE_ROWS = 4 * SMALL_ROWS
MAX_GROWTH = 1.5
FLAGSHIP_KEYS = 5
BATCH_ROWS = 100
"""The flagship scenario needs both. With ten keys and thousand-row batches a
full-value-set regression grows only 1.5x-1.95x against a 1.5 threshold, which is
not a margin. Five keys means each reported group holds a fifth of every row's
value rather than a tenth, and small batches stop a batch list dominating the
peak: the regression then grows 2.70x against 0.55x for the correct code."""


class FakeToolContext:
    def __init__(self):
        self.state = {}


@pytest.fixture
def streamed_source(monkeypatch):
    """Real headers on disk; rows fabricated on demand."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/big.csv", "w") as handle:
        handle.write("key,value\n")
    with fs.open("/src/home.csv", "w") as handle:
        handle.write("value,other\n" + "".join(f"v{i},o{i}\n" for i in range(10)))
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _fabricate(monkeypatch, make_row, row_count, header=("key", "value")):
    """Patch read_csv_batches to yield row_count fabricated rows in batches."""

    def batches(path, *args, **kwargs):
        if path != "big.csv":
            # Anything but the fabricated file reads normally.
            return _real_batches(path, *args, **kwargs)
        return _fabricated(path)

    def _fabricated(path):
        batch = []
        for index in range(row_count):
            batch.append(make_row(index))
            if len(batch) >= BATCH_ROWS:
                yield list(header), batch
                batch = []
        if batch:
            yield list(header), batch

    _real_batches = file_tools.read_csv_batches
    monkeypatch.setattr(file_tools, "read_csv_batches", batches)


def _succeeds(call):
    """Wrap a tool call so the measurement cannot pass by doing nothing.

    Without this, a reader that errors on every call makes every scenario hold
    nothing and every threshold pass -- the tests go green on code that answers
    no questions at all.
    """

    def work():
        result = call()
        assert result["status"] == "success", result
        return result

    return work


def _peak_bytes(work):
    gc.collect()
    tracemalloc.start()
    try:
        work()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _growth(monkeypatch, make_row, work, header=("key", "value")):
    """Peak at 4N rows divided by peak at N rows."""
    with monkeypatch.context() as patch:
        _fabricate(patch, make_row, SMALL_ROWS, header)
        small = _peak_bytes(work)
    with monkeypatch.context() as patch:
        _fabricate(patch, make_row, LARGE_ROWS, header)
        large = _peak_bytes(work)
    assert small > 0
    return large / small


def test_column_stats_over_ten_distinct_values_does_not_grow_with_rows(
    streamed_source, monkeypatch
):
    growth = _growth(
        monkeypatch,
        lambda i: {"key": f"k{i % 10}", "value": f"v{i % 10}"},
        _succeeds(
            lambda: file_tools.column_stats("big.csv", "value", FakeToolContext())
        ),
    )
    assert growth < MAX_GROWTH, f"peak grew {growth:.2f}x with 4x the rows"


def test_join_preview_over_ten_distinct_values_does_not_grow_with_rows(
    streamed_source, monkeypatch
):
    growth = _growth(
        monkeypatch,
        lambda i: {"key": f"k{i % 10}", "value": f"v{i % 10}"},
        _succeeds(
            lambda: file_tools.join_preview(
                "big.csv", "value", "home.csv", "value", FakeToolContext()
            )
        ),
    )
    assert growth < MAX_GROWTH, f"peak grew {growth:.2f}x with 4x the rows"


def test_collapse_check_over_ten_keys_and_values_does_not_grow_with_rows(
    streamed_source, monkeypatch
):
    growth = _growth(
        monkeypatch,
        lambda i: {"key": f"k{i % 10}", "value": f"v{i % 10}"},
        _succeeds(
            lambda: file_tools.collapse_check(
                "big.csv", "key", "value", FakeToolContext()
            )
        ),
    )
    assert growth < MAX_GROWTH, f"peak grew {growth:.2f}x with 4x the rows"


def test_collapse_check_of_a_per_row_candidate_grows_with_keys_not_values(
    streamed_source, monkeypatch
):
    """THE FLAGSHIP SHAPE. A candidate unique per row under five repeating keys is
    exactly what collapse_check exists to catch, and it is the only shape where
    holding full value sets for the five reported keys costs O(rows): those five
    sets hold a fifth of every row's value. Measured on this harness: a
    full-value-set regression grows 2.70x, the correct ten-smallest retention
    0.55x. Delete this test, or widen it back to ten keys and thousand-row
    batches, and that decision silently reverts -- at ten keys the same
    regression grows only 1.5x-1.95x and slips under the threshold."""
    growth = _growth(
        monkeypatch,
        lambda i: {"key": f"k{i % FLAGSHIP_KEYS}", "value": f"v{i:09d}"},
        _succeeds(
            lambda: file_tools.collapse_check(
                "big.csv", "key", "value", FakeToolContext()
            )
        ),
    )
    assert growth < MAX_GROWTH, f"peak grew {growth:.2f}x with 4x the rows"


def test_the_reachability_check_does_not_grow_with_rows(streamed_source, monkeypatch):
    """The large file must be the NON-home side, its node rule must be keyed by a
    REPEATING column, and the plan must contain NO rule keyed by the shared column.

    All three matter, and the third is the trap. `_property_failure` runs only
    inside `if not covered`, and `_covers_a_home` is satisfied by any rule keyed
    by the column whose file is a home file. Add a tidy-looking second rule for
    home.csv keyed by 'value' and coverage succeeds immediately, the retaining
    branch never executes, and a mutant that hoards every row inside
    `_property_failure` passes this test at 1.00x.

    A home file is also unique per row and legitimately costs its distinct values,
    so a large home file measures nothing; and a rule keyed by a per-row-unique
    column makes the grouping hold one state per row for the same legitimate
    reason. home.csv holds ten unique 'value's, big.csv repeats them under ten
    repeating 'key's, and the single rule is keyed by 'key' and retains 'value'.
    """
    plan = {
        "r1": {
            "construction_type": "node",
            "source_file": "big.csv",
            "label": "Grouped",
            "unique_column_name": "key",
            "properties": ["key", "value"],
        }
    }
    reached = []
    real_property_failure = rr._property_failure

    def recording_property_failure(rule, column):
        reached.append(column)
        return real_property_failure(rule, column)

    monkeypatch.setattr(rr, "_property_failure", recording_property_failure)

    def work():
        result = rr.check_reference_columns_are_reachable(plan, ["big.csv", "home.csv"])
        # Without this the test passes when the reader errors on every call and
        # nothing is ever held. Same reason the file tools assert success below.
        assert reached, (
            "scenario never reached _property_failure -- it measures nothing"
        )
        return result

    growth = _growth(
        monkeypatch,
        lambda i: {"key": f"k{i % 10}", "value": f"v{i % 10}"},
        work,
    )
    assert growth < MAX_GROWTH, f"peak grew {growth:.2f}x with 4x the rows"
