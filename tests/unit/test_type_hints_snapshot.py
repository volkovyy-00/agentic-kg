"""Today's type hints, recorded before KG-11 changed how they are computed.

KG-11 moved the hint tools from holding every value of a column to streaming it
through value_types.ColumnTally. That refactor must not change a single hint:
same shape, same suggestion, same counts, same examples in the same order, at any
batch size. This file pins that, for every column of the bundled BOM data and for
every adversarial column and value in test_value_types_invariants.py, against a
snapshot recorded from the code as it was before the refactor began.

It pins THIS refactor. A typing rule changed on purpose later will fail here;
regenerate the snapshot then with

    uv run python tests/unit/test_type_hints_snapshot.py

and know that, once the old code is gone, regenerating records whatever the code
does now -- it is no longer a comparison with anything. The lasting guards are the
oracle tests in test_file_tools.py (the hint's counts equal a plain coerce loop),
the classify tests in test_value_types.py, and the invariant tests.
"""

import csv
import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest

from agentic_kg.common import csv_reader
from agentic_kg.common.config import reset_settings
from agentic_kg.tools import file_tools

SNAPSHOT = Path(__file__).parent / "fixtures" / "type_hints_before.json"
BOM = Path(__file__).resolve().parents[2] / "data" / "bom"

# Values a CSV cannot carry, or carries only padded, run through the list path.
EXTRA_LISTS = [
    [42, 43, 44],
    [1.5, 2.0],
    [True, False, 42],
    [" 42 ", "43"],
    [" 42", " 43 "],
    ["1", None, "", "2"],
]


class FakeToolContext:
    def __init__(self):
        self.state = {}


def _list_hint(name, values):
    """The hint for a list of values, through the code under test."""
    return file_tools._hint_from_values("list.csv", name, values)


def _load():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture
def bom_source(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", "./data/bom")
    reset_settings()
    yield
    reset_settings()


@pytest.mark.parametrize("batch_size", [1, 7, 1000])
def test_every_bom_column_keeps_its_hint_at_any_batch_size(
    bom_source, monkeypatch, batch_size
):
    monkeypatch.setattr(
        file_tools,
        "read_csv_batches",
        partial(csv_reader.read_csv_batches, batch_size=batch_size),
    )
    entries = _load()["bom"]
    assert entries, "the snapshot holds no BOM columns -- it checks nothing"
    by_file: dict[str, list] = {}
    for entry in entries:
        result = file_tools.column_type_hint(
            entry["file"], entry["column"], FakeToolContext()
        )
        assert result["status"] == "success", result
        assert result["column_type_hint"] == entry["hint"], entry["column"]
        by_file.setdefault(entry["file"], []).append(entry)
    # The several-column tool must agree with the one-column tool, entry by entry.
    for file_name, file_entries in by_file.items():
        batched = file_tools.column_type_hints(
            file_name, [entry["column"] for entry in file_entries], FakeToolContext()
        )
        assert batched["status"] == "success", batched
        assert batched["column_type_hints"] == [entry["hint"] for entry in file_entries]


def _list_entries():
    # Read at collection time. Missing only while Step 4 of Task 1 generates it;
    # test_the_snapshot_holds_value_lists then fails loudly if it stays missing.
    return _load()["lists"] if SNAPSHOT.exists() else []


def test_the_snapshot_holds_value_lists():
    assert _load()["lists"], "the snapshot holds no value lists -- it checks nothing"


@pytest.mark.parametrize("entry", _list_entries(), ids=lambda entry: entry["name"])
def test_every_value_list_keeps_its_hint(entry):
    assert _list_hint(entry["name"], entry["values"]) == entry["hint"]


def _record():
    """Write the snapshot from the code as it is now. See the module docstring."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_value_types_invariants import ADVERSARIAL_COLUMNS, ADVERSARIAL_VALUES

    os.environ["SOURCE_URI"] = str(BOM)
    reset_settings()
    bom = []
    for path in sorted(BOM.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        for column in header:
            hint = file_tools.column_type_hint(path.name, column, FakeToolContext())
            assert hint["status"] == "success", hint
            bom.append(
                {"file": path.name, "column": column, "hint": hint["column_type_hint"]}
            )
    lists = []
    named = (
        [(f"column-{n:02d}", list(c)) for n, c in enumerate(ADVERSARIAL_COLUMNS)]
        + [(f"value-{n:02d}", [v]) for n, v in enumerate(ADVERSARIAL_VALUES)]
        + [(f"extra-{n:02d}", list(c)) for n, c in enumerate(EXTRA_LISTS)]
    )
    for name, values in named:
        lists.append({"name": name, "values": values, "hint": _list_hint(name, values)})
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps({"bom": bom, "lists": lists}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"recorded {len(bom)} BOM columns and {len(lists)} value lists")


if __name__ == "__main__":
    _record()
