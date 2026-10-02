"""KG-47: a source file the program refuses never raises out of a consumer.

open_source raises SourceEncodingError for a file that is not UTF-8 or
Windows-1252 text. Every tool, reader and plan check that reads a source must
turn that into an error result or a note, never let it escape:
.claude/rules/construction-plan.md says approval must never catch a check's
exception, so a refusal that reached a plan check as a raise would crash
approval instead of producing a plan problem.

Neutral vocabulary (a field survey), like tests/unit/test_reference_reachability.py.
"""

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import file_tools
from agentic_kg.tools import reference_reachability as rr
from agentic_kg.tools.construction_plan_tools import (
    PROPOSED_CONSTRUCTION_PLAN,
    approve_proposed_construction_plan,
    find_plan_problems,
)
from agentic_kg.tools.file_tools import APPROVED_FILES

NOT_TEXT = "is not valid UTF-8 or Windows-1252 text, so it was not read"
APPROVED = ["plots.csv", "readings.csv"]


class FakeToolContext:
    def __init__(self, state=None):
        self.state = state if state is not None else {}


@pytest.fixture
def refused_source(monkeypatch):
    """plots.csv reads fine. readings.csv holds 0x81, valid in neither encoding."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write("plot_label,plot_id\nridge,PL-1\nridge,PL-2\nhollow,PL-3\n")
    with fs.open("/src/readings.csv", "wb") as handle:
        handle.write(b"reading_id,plot_id\nR-1,PL-1\nR-2,PL-\x813\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _assert_refused(result):
    assert result["status"] == "error"
    assert "readings.csv" in result["error_message"]
    assert NOT_TEXT in result["error_message"]


# --- the file tools -------------------------------------------------------------


def test_sample_file_refuses(refused_source):
    _assert_refused(file_tools.sample_file("readings.csv", FakeToolContext()))


@pytest.mark.parametrize("query", ["PL", ""], ids=["query", "empty-query"])
def test_search_csv_file_refuses_even_for_an_empty_query(refused_source, query):
    """An empty query still reads the header, so it is refused too."""
    result = file_tools.search_csv_file("readings.csv", query, FakeToolContext())
    _assert_refused(result)


def test_search_file_refuses(refused_source):
    _assert_refused(file_tools.search_file("readings.csv", "PL"))


def test_search_file_with_an_empty_query_opens_nothing_so_is_not_refused(
    refused_source,
):
    """An empty query reads no byte, so search_file does not refuse the file: the
    one place "every tool refuses" does not apply. Pinned so nobody 'fixes' it by
    accident."""
    result = file_tools.search_file("readings.csv", "")
    assert result["status"] == "success"
    assert result[file_tools.SEARCH_RESULTS]["metadata"]["lines_found"] == 0


def test_column_stats_refuses(refused_source):
    _assert_refused(
        file_tools.column_stats("readings.csv", "plot_id", FakeToolContext())
    )


def test_collapse_check_refuses(refused_source):
    _assert_refused(
        file_tools.collapse_check(
            "readings.csv", "reading_id", "plot_id", FakeToolContext()
        )
    )


def test_the_type_hint_tools_refuse(refused_source):
    _assert_refused(
        file_tools.column_type_hint("readings.csv", "plot_id", FakeToolContext())
    )
    _assert_refused(
        file_tools.column_type_hints("readings.csv", ["plot_id"], FakeToolContext())
    )


def test_join_preview_refuses(refused_source):
    _assert_refused(
        file_tools.join_preview(
            "plots.csv", "plot_id", "readings.csv", "plot_id", FakeToolContext()
        )
    )


# --- the shared readers: an error value, not a raise ----------------------------


def test_the_summarisers_return_the_refusal_as_an_error_value(refused_source):
    summary, error = file_tools.summarize_column("readings.csv", "plot_id")
    assert summary is None
    _assert_refused(error)

    groups, error = file_tools.summarize_key_groups(
        "readings.csv", "reading_id", "plot_id"
    )
    assert groups is None
    _assert_refused(error)


def test_the_column_collectors_return_the_refusal_as_an_error_value(refused_source):
    values, error = file_tools.collect_column_values("readings.csv", "plot_id")
    assert values is None
    _assert_refused(error)

    by_column, error = file_tools._collect_columns_values("readings.csv", ["plot_id"])
    assert by_column is None
    _assert_refused(error)


# --- the plan checks: a note, never a raise -------------------------------------


def _plot_node():
    return {
        "Plot": {
            "construction_type": "node",
            "source_file": "plots.csv",
            "label": "Plot",
            "unique_column_name": "plot_label",
            "properties": [],
        }
    }


def test_reachability_reports_a_refused_header_as_a_note(refused_source):
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node(), APPROVED
    )
    assert problems == []
    assert any("readings.csv" in note and NOT_TEXT in note for note in unverified)


def test_the_plan_check_and_approval_do_not_raise_for_a_refused_source(
    refused_source,
):
    """find_plan_problems deliberately does not catch, and approval inherits that.
    A refusal that escaped any check would crash both."""
    state = {PROPOSED_CONSTRUCTION_PLAN: _plot_node(), APPROVED_FILES: APPROVED}
    _problems, unverified = find_plan_problems(state)
    assert any("readings.csv" in note and NOT_TEXT in note for note in unverified)

    result = approve_proposed_construction_plan(FakeToolContext(state))
    assert result["status"] in ("success", "error")
