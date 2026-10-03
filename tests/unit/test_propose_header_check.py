"""KG-50: proposing a key or join column checks it against the file's header.

The propose tools used to ask `search_file` whether the column's text appeared
anywhere in the file, in any case and on any line, so a misspelt, differently
cased or made-up column passed planning and was refused only by the build. The
build compares exactly against the header; so must the propose tools.

These tests read real files from an in-memory source and never fake the header
reader. A fake could pass without reaching the code under test (KG-34's hazard);
the existence-failure test makes `source_exists` itself fail instead.

Neutral vocabulary (a field survey), like tests/unit/test_reference_reachability.py.
"""

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import file_tools
from agentic_kg.tools.construction_plan_tools import (
    PROPOSED_CONSTRUCTION_PLAN,
    propose_node_construction,
    propose_node_constructions,
    propose_relationship_construction,
    propose_relationship_constructions,
)

HEADER = ["plotID", "siteID", "canopy"]
NOT_TEXT = "is not valid UTF-8 or Windows-1252 text, so it was not read"


class FakeToolContext:
    def __init__(self):
        self.state = {}


@pytest.fixture
def ctx():
    return FakeToolContext()


@pytest.fixture
def survey_source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    # 'ridge' is a data value, not a column; 'plot' sits inside 'plotID'.
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write("plotID,siteID,canopy\nPL-1,ridge,open\nPL-2,hollow,closed\n")
    with fs.open("/src/empty.csv", "w") as handle:
        handle.write("")
    with fs.open("/src/refused.csv", "wb") as handle:
        handle.write(b"plotID,siteID\nPL-\x811,ridge\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _node(ctx, column, approved_file="plots.csv"):
    return propose_node_construction(approved_file, "Plot", column, [], ctx)


def _rel(ctx, from_column, to_column, approved_file="plots.csv", **extra):
    return propose_relationship_construction(
        approved_file,
        "ON_SITE",
        "Plot",
        from_column,
        "Site",
        to_column,
        [],
        ctx,
        **extra,
    )


def _not_in(column, header=HEADER, file="plots.csv"):
    return f"Column '{column}' is not in {file}. Available columns: {header}"


def _several_not_in(columns, header=HEADER, file="plots.csv"):
    return f"Column(s) {columns} are not in {file}. Available columns: {header}"


# --- a node's key column ----------------------------------------------------


@pytest.mark.parametrize(
    "column",
    ["plotid", "PLOTID", "plot", "ridge", "id"],
    ids=["lower-case", "upper-case", "inside-longer-header", "data-value", "fragment"],
)
def test_a_node_key_that_is_not_exactly_a_header_is_refused(survey_source, ctx, column):
    result = _node(ctx, column)
    assert result["status"] == "error"
    assert _not_in(column) in result["error_message"]
    assert PROPOSED_CONSTRUCTION_PLAN not in ctx.state


def test_a_node_key_that_is_exactly_a_header_is_accepted(survey_source, ctx):
    result = _node(ctx, "plotID")
    assert result["status"] == "success", result.get("error_message")
    assert (
        ctx.state[PROPOSED_CONSTRUCTION_PLAN]["Plot"]["unique_column_name"] == "plotID"
    )


def test_a_node_batch_refuses_a_wrong_key_and_names_the_entry(survey_source, ctx):
    result = propose_node_constructions(
        [
            {
                "approved_file": "plots.csv",
                "proposed_label": "Plot",
                "unique_column_name": "plotID",
                "proposed_properties": [],
            },
            {
                "approved_file": "plots.csv",
                "proposed_label": "Site",
                "unique_column_name": "siteid",
                "proposed_properties": [],
            },
        ],
        ctx,
    )
    assert result["status"] == "error"
    assert result["error_message"].startswith("node construction 1 (Site) failed: ")
    assert _not_in("siteid") in result["error_message"]
    assert set(ctx.state[PROPOSED_CONSTRUCTION_PLAN]) == {"Plot"}


# --- a relationship's two join columns --------------------------------------


@pytest.mark.parametrize(
    ("from_column", "to_column", "missing"),
    [
        ("plotid", "siteID", "plotid"),
        ("plotID", "SITEID", "SITEID"),
        ("ridge", "siteID", "ridge"),
        ("plotID", "site", "site"),
    ],
    ids=["from-case", "to-case", "from-data-value", "to-inside-longer-header"],
)
def test_a_join_column_that_is_not_exactly_a_header_is_refused(
    survey_source, ctx, from_column, to_column, missing
):
    result = _rel(ctx, from_column, to_column)
    assert result["status"] == "error"
    assert result["error_message"] == _several_not_in([missing])
    assert PROPOSED_CONSTRUCTION_PLAN not in ctx.state


def test_two_wrong_join_columns_are_both_named_as_the_build_does(survey_source, ctx):
    result = _rel(ctx, "plotid", "SITEID")
    assert result["error_message"] == _several_not_in(["plotid", "SITEID"])


def test_one_wrong_column_used_for_both_ends_is_named_once(survey_source, ctx):
    """The repeated column is checked once, so it gets the one-column wording."""
    result = _rel(ctx, "ridge", "ridge")
    assert result["error_message"] == _not_in("ridge")


def test_exact_join_columns_are_accepted(survey_source, ctx):
    result = _rel(ctx, "plotID", "siteID")
    assert result["status"] == "success", result.get("error_message")


def test_a_join_column_is_refused_even_when_the_end_matches_on_a_property(
    survey_source, ctx
):
    """KG-45 (KG-50 comment of 2026-10-01): with to_node_property set, the join
    column need no longer be a property of the node, so this check is the only
    guard against a misspelt one."""
    result = _rel(ctx, "plotID", "siteid", to_node_property="siteCode")
    assert result["status"] == "error"
    assert result["error_message"] == _several_not_in(["siteid"])


def test_a_relationship_batch_refuses_a_wrong_join_column(survey_source, ctx):
    entry = {
        "approved_file": "plots.csv",
        "proposed_relationship_type": "ON_SITE",
        "from_node_label": "Plot",
        "from_node_column": "plotID",
        "to_node_label": "Site",
        "to_node_column": "siteid",
        "proposed_properties": [],
    }
    result = propose_relationship_constructions([entry], ctx)
    assert result["status"] == "error"
    assert result["error_message"].startswith(
        "relationship construction 0 (ON_SITE) failed: "
    )
    assert _several_not_in(["siteid"]) in result["error_message"]


# --- a file with no usable header, or none at all ---------------------------


def _propose_on(kind, ctx, approved_file):
    if kind == "node":
        return _node(ctx, "plotID", approved_file)
    return _rel(ctx, "plotID", "siteID", approved_file)


KINDS = pytest.mark.parametrize("kind", ["node", "relationship"])


@KINDS
def test_an_empty_file_is_refused_with_its_own_message(survey_source, ctx, kind):
    result = _propose_on(kind, ctx, "empty.csv")
    assert result["status"] == "error"
    assert result["error_message"] == "CSV file has no header row: empty.csv"


@KINDS
def test_a_file_that_does_not_exist_is_still_refused(survey_source, ctx, kind):
    result = _propose_on(kind, ctx, "absent.csv")
    assert result["status"] == "error"
    assert "absent.csv" in result["error_message"]
    assert "does not exist" in result["error_message"]


@KINDS
def test_a_refused_file_is_an_error_result_not_a_raise(survey_source, ctx, kind):
    result = _propose_on(kind, ctx, "refused.csv")
    assert result["status"] == "error"
    assert "refused.csv" in result["error_message"]
    assert NOT_TEXT in result["error_message"]


@KINDS
@pytest.mark.parametrize("failure", [PermissionError, RuntimeError])
def test_a_failing_existence_check_is_an_error_result_not_a_raise(
    survey_source, ctx, monkeypatch, kind, failure
):
    def failing(path):
        raise failure("the existence check failed")

    monkeypatch.setattr(file_tools, "source_exists", failing)
    result = _propose_on(kind, ctx, "plots.csv")
    assert result["status"] == "error"
    assert "the existence check failed" in result["error_message"]


# --- cost and order ---------------------------------------------------------


def test_a_relationship_reads_the_header_once_for_both_columns(
    survey_source, ctx, monkeypatch
):
    reads = []
    real = file_tools.read_csv_header

    def counting(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(file_tools, "read_csv_header", counting)
    assert _rel(ctx, "plotID", "siteID")["status"] == "success"
    assert reads == ["plots.csv"]


def test_a_bad_name_is_refused_before_the_header_is_read(
    survey_source, ctx, monkeypatch
):
    def never(*args, **kwargs):
        raise AssertionError("the file must not be read for a bad name")

    monkeypatch.setattr(file_tools, "read_csv_header", never)
    monkeypatch.setattr(file_tools, "source_exists", never)
    assert _node(ctx, "plot id")["status"] == "error"
    assert _rel(ctx, "plotID", "site id")["status"] == "error"
