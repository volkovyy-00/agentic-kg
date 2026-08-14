"""Layer 1 of the orphaned-reference-column check.

Deliberately vocabulary-neutral: a field-survey domain, sharing no column, file
or entity name with the bundled example this ticket was diagnosed from. These
fixtures are the evidence that the rule keys on structure rather than on one
dataset's names, so their vocabulary is load-bearing and is itself tested.

Do NOT name the excluded vocabulary here, even to explain the rule: the test in
tests/unit/test_generality.py that guards this file asserts those very words are
absent from it, and an explanatory sentence listing them fails its own check.
"""

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import reference_reachability as rr


@pytest.fixture
def survey_source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    # plot_label repeats (2 distinct), plot_id is unique per row (4 distinct).
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write(
            "plot_label,plot_id,canopy,tally\n"
            "ridge,PL-1,open,3\n"
            "ridge,PL-2,closed,3\n"
            "hollow,PL-3,open,5\n"
            "hollow,PL-4,closed,5\n"
        )
    # readings.csv references plot_id and carries no label column at all.
    # 'tally' must repeat here as well as in plots.csv: it is the shared-but-
    # unique-in-neither case, and a file where it happened to be per-row unique
    # would make it a genuine candidate and break every count assertion below.
    with fs.open("/src/readings.csv", "w") as handle:
        handle.write("reading_id,plot_id,tally\nR-1,PL-1,3\nR-2,PL-1,3\nR-3,PL-3,5\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_only_columns_present_in_two_files_are_considered(survey_source):
    """Case 4: catches an implementation that drops the cross-file condition and
    treats every per-row-unique column as a reference column. 'reading_id' is
    perfectly unique and answers to nobody."""
    columns, unreadable, notes = rr._columns_by_file(["plots.csv", "readings.csv"])
    shared = {name for name, files in columns.items() if len(files) >= 2}
    assert shared == {"plot_id", "tally"}
    assert "reading_id" not in shared
    assert unreadable == [] and notes == []


def test_a_column_shared_but_unique_in_neither_file_has_no_home(survey_source):
    """Case 2: catches dropping the candidacy test. 'tally' is in both files and
    per-row unique in neither, so it is not an identifier at all."""
    homes, complete, notes = rr._home_files("tally", ["plots.csv", "readings.csv"])
    assert homes == []
    assert complete is True and notes == []


def test_the_home_file_is_the_one_where_the_column_is_per_row_unique(survey_source):
    """Catches treating any file containing the column as a home file: plot_id is
    unique in plots.csv and repeats in readings.csv."""
    homes, complete, notes = rr._home_files("plot_id", ["plots.csv", "readings.csv"])
    assert homes == ["plots.csv"]
    assert complete is True


def test_a_column_renamed_between_files_is_never_shared(survey_source):
    """Case 3: catches value-domain matching. 'canopy' exists only in plots.csv;
    matching is literal column-name equality, nothing cleverer."""
    columns, _, _ = rr._columns_by_file(["plots.csv", "readings.csv"])
    assert columns["canopy"] == ["plots.csv"]


def test_an_unreadable_header_is_noted_and_does_not_raise(survey_source):
    """Catches using read_csv_header without try/except: it raises rather than
    returning an error dict, which would blow up a plan presentation."""
    columns, unreadable, notes = rr._columns_by_file(["plots.csv", "absent.csv"])
    assert unreadable == ["absent.csv"]
    assert len(notes) == 1 and "absent.csv" in notes[0]
    assert columns["plot_id"] == ["plots.csv"]


def test_an_unreadable_value_read_marks_evidence_incomplete(survey_source):
    """Catches an implementation that silently treats an unreadable file as 'not a
    home file'. The flag is what later stops a refusal being built on a failed read."""
    homes, complete, notes = rr._home_files("plot_id", ["plots.csv", "absent.csv"])
    assert homes == ["plots.csv"]
    assert complete is False
    assert len(notes) == 1 and "absent.csv" in notes[0]
