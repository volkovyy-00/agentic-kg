"""KG-26: a failed existence check is an error result, not a raise.

The column readers behind reachability and the hint tools ask `source_exists`
whether a source is there before reading it. These tests make that call itself
fail and leave the CSV reader real. Faking the reader instead can pass without
ever reaching the guard: reachability's header read goes csv_reader ->
open_source -> fs.exists and never calls `file_tools.source_exists`, so a
failing `source_exists` reproduces "the header read succeeded, the existence
check then failed" -- the transient error on a remote source.

The failing fake raises a non-SourceError. A SourceError fake could pass on the
old code, whose narrow guard already caught that one kind.

Neutral vocabulary (a field survey), like tests/unit/test_reference_reachability.py.
"""

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.file_source import SourceError
from agentic_kg.tools import file_tools
from agentic_kg.tools import reference_reachability as rr

EXISTENCE_FAILURES = [PermissionError, RuntimeError]
"""Classes, not instances: `_fail_existence` raises a fresh instance per call. An
instance shared by two tests is raised twice and accumulates both tests' traceback
frames, so a full-file run would show the wrong call path under a failure.

PermissionError is an OSError; a remote backend's error need not be (gcsfs's
HttpError and adlfs's HttpResponseError are plain Exceptions). With the OSError
alone, a catch narrowed to `except OSError` would keep every test green and
Decision 3 -- the catch is broad -- would be unpinned."""
FAILURE_IDS = ["oserror", "not-oserror"]
FAILURE_TEXT = "the existence check failed"

APPROVED = ["plots.csv", "readings.csv"]


class FakeToolContext:
    def __init__(self):
        self.state = {}


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
    # readings.csv references plot_id; 'tally' repeats in both files so it is
    # shared but unique in neither, and never a reference column.
    with fs.open("/src/readings.csv", "w") as handle:
        handle.write("reading_id,plot_id,tally\nR-1,PL-1,3\nR-2,PL-1,3\nR-3,PL-3,5\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _fail_existence(monkeypatch, exc_type):
    """Make the existence check itself raise `exc_type(FAILURE_TEXT)`, for every path."""

    def failing(_path):
        raise exc_type(FAILURE_TEXT)

    monkeypatch.setattr(file_tools, "source_exists", failing)


def _plot_node(unique_column, properties):
    return {
        "Plot": {
            "construction_type": "node",
            "source_file": "plots.csv",
            "label": "Plot",
            "unique_column_name": unique_column,
            "properties": properties,
        }
    }


# --- the readers (AC1) --------------------------------------------------------


@pytest.mark.parametrize("exc_type", EXISTENCE_FAILURES, ids=FAILURE_IDS)
def test_summarize_column_reports_a_failed_existence_check(
    survey_source, monkeypatch, exc_type
):
    _fail_existence(monkeypatch, exc_type)

    summary, error = file_tools.summarize_column("plots.csv", "plot_id")

    assert summary is None
    assert error is not None
    assert error["status"] == "error"
    assert error["error_message"] == f"Error reading CSV file plots.csv: {FAILURE_TEXT}"


def test_summarize_key_groups_reports_a_failed_existence_check(
    survey_source, monkeypatch
):
    _fail_existence(monkeypatch, PermissionError)

    summary, error = file_tools.summarize_key_groups(
        "plots.csv", "plot_label", "plot_id"
    )

    assert summary is None
    assert error is not None
    assert error["status"] == "error"
    assert error["error_message"] == f"Error reading CSV file plots.csv: {FAILURE_TEXT}"


# --- reachability (AC1): a note, not a raise ----------------------------------


def test_a_failed_existence_check_becomes_a_reachability_note(
    survey_source, monkeypatch
):
    """Keying Plot by the repeating label strands readings.csv's plot_id, which a
    verified check would refuse. With the existence check failing after the
    header read succeeded, the check must not raise and must not refuse: it
    reports the column as not verified, and the note can only have come from the
    existence failure because it carries that failure's wording."""
    _fail_existence(monkeypatch, PermissionError)

    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), APPROVED
    )

    assert problems == []
    assert any(
        "plot_id" in note and "Error reading CSV file" in note for note in unverified
    )


def test_a_failed_key_group_read_becomes_a_reachability_note(
    survey_source, monkeypatch
):
    """The retained-property route: `_property_failure` -> summarize_key_groups.

    A blanket failing `source_exists` cannot reach it. `_carrying_rules` drops any
    rule whose source is not in `value_sets`, and `_home_files` fills `value_sets`
    only when `summarize_column` succeeds, so every read must succeed until the
    key-group read itself. The wrapper arms the failure for exactly that call and
    delegates to the real function, so the reader stays real."""
    armed = {"on": False, "reads": 0}
    real_exists = file_tools.source_exists
    real_key_groups = rr.summarize_key_groups

    def exists_unless_armed(path):
        if armed["on"]:
            raise PermissionError("denied")
        return real_exists(path)

    def arm_for_the_read(*args, **kwargs):
        armed["on"] = True
        armed["reads"] += 1
        try:
            return real_key_groups(*args, **kwargs)
        finally:
            armed["on"] = False

    monkeypatch.setattr(file_tools, "source_exists", exists_unless_armed)
    monkeypatch.setattr(rr, "summarize_key_groups", arm_for_the_read)

    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["plot_id"]), APPROVED
    )

    assert armed["reads"] == 1, "the check never reached the key-group read"
    assert problems == []
    assert len(unverified) == 1
    assert "plot_id" in unverified[0]
    assert "Error reading CSV file" in unverified[0]


# --- the hint tools (AC4) -----------------------------------------------------


@pytest.mark.parametrize("exc_type", EXISTENCE_FAILURES, ids=FAILURE_IDS)
def test_column_type_hint_reports_a_failed_existence_check(
    survey_source, monkeypatch, exc_type
):
    _fail_existence(monkeypatch, exc_type)

    result = file_tools.column_type_hint("plots.csv", "plot_id", FakeToolContext())

    assert result["status"] == "error"
    assert (
        result["error_message"] == f"Error reading CSV file plots.csv: {FAILURE_TEXT}"
    )


def test_column_type_hints_reports_a_failed_existence_check(survey_source, monkeypatch):
    _fail_existence(monkeypatch, PermissionError)

    result = file_tools.column_type_hints("plots.csv", ["plot_id"], FakeToolContext())

    assert result["status"] == "error"
    assert (
        result["error_message"] == f"Error reading CSV file plots.csv: {FAILURE_TEXT}"
    )


# --- characterization: green before and after the fix -------------------------


def test_a_source_that_does_not_exist_is_still_reported_as_missing(survey_source):
    """AC3. These pass on the old code too; they catch the refactor changing the
    missing-source message. The existing missing-file tests for column_stats,
    join_preview and collapse_check assert only status == "error", and none
    covered the hint tools at all."""
    expected = "CSV file does not exist: absent.csv"

    _, error = file_tools.summarize_column("absent.csv", "plot_id")
    assert error is not None
    assert error["error_message"] == expected

    hint = file_tools.column_type_hint("absent.csv", "plot_id", FakeToolContext())
    assert hint["status"] == "error"
    assert hint["error_message"] == expected

    hints = file_tools.column_type_hints("absent.csv", ["plot_id"], FakeToolContext())
    assert hints["status"] == "error"
    assert hints["error_message"] == expected


def test_a_source_error_keeps_its_own_message(survey_source, monkeypatch):
    """I2. Passes on the old code too. It pins the branch ORDER in the new helper:
    SourceError is an Exception, so a broad catch above its branch would replace
    this message with the read-failure wording."""
    _fail_existence(monkeypatch, SourceError)

    values, error = file_tools.collect_column_values("plots.csv", "plot_id")

    assert values is None
    assert error is not None
    # Exactly str(exc): no "Error reading CSV file" wrapper around it.
    assert error["error_message"] == FAILURE_TEXT
