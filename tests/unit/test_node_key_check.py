"""KG-48: a node may not be proposed on a key whose rows would collapse.

Vocabulary-neutral, like test_join_property_check.py: a field-survey domain. The
bundled example's own cases, and the propose tools' wiring, live in
test_construction_plan_tools.py.
"""

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import node_key_check
from agentic_kg.tools.node_key_check import (
    KeyExample,
    NodeKeySummary,
    node_key_refusal,
    summarize_node_key,
)


@pytest.fixture
def source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()

    def write(name, text):
        with fs.open(f"/src/{name}", "w") as handle:
            handle.write(text)

    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield write
    fs.store.clear()
    fs.pseudo_dirs.clear()


@pytest.fixture
def no_property_reads(monkeypatch):
    """Fail the test if a property column is read at all."""

    def refuse(*args, **kwargs):
        raise AssertionError("a property column was read")

    monkeypatch.setattr(node_key_check, "summarize_key_groups", refuse)


def _summarize(file_path, key, properties):
    summary, error = summarize_node_key(file_path, key, properties)
    assert error is None, error
    return summary


def _conflicting(summary):
    return [conflict.property for conflict in summary.conflicts]


# --- the key's own values -----------------------------------------------------


def test_a_unique_key_is_counted_and_reads_no_property(source, no_property_reads):
    source("plots.csv", "plot,soil\nA1,clay\nA2,sand\nA3,clay\n")

    summary = _summarize("plots.csv", "plot", ["soil"])

    assert summary == NodeKeySummary(3, 3, 0, [])


def test_empty_whitespace_and_short_row_keys_are_all_blank(source, no_property_reads):
    source("plots.csv", "plot,soil\nA1,clay\n,sand\n  ,clay\n")
    source("short.csv", "soil,plot\nclay\nsand,A2\n")

    assert _summarize("plots.csv", "plot", ["soil"]).blank_count == 2
    assert _summarize("short.csv", "plot", ["soil"]).blank_count == 1


def test_a_blank_line_at_the_end_of_the_file_is_a_row_without_a_key(
    source, no_property_reads
):
    source("plots.csv", "plot,soil\nA1,clay\nA2,sand\n\n")

    summary = _summarize("plots.csv", "plot", ["soil"])

    assert (summary.row_count, summary.distinct_count, summary.blank_count) == (3, 2, 1)


def test_a_blank_key_stops_before_any_property_is_read(source, no_property_reads):
    source("plots.csv", "plot,soil\nA1,clay\nA1,sand\n,clay\n")

    summary = _summarize("plots.csv", "plot", ["soil"])

    assert summary.blank_count == 1
    assert summary.conflicts == []


# --- rows sharing a key -------------------------------------------------------


def test_a_repeating_key_whose_rows_agree_on_every_property_has_no_conflict(source):
    source("visits.csv", "plot,soil,slope\nA1,clay,3\nA1,clay,3\nA2,sand,1\n")

    summary = _summarize("visits.csv", "plot", ["soil", "slope"])

    assert summary == NodeKeySummary(3, 2, 0, [])


def test_a_repeating_key_with_no_properties_has_no_conflict(source, no_property_reads):
    source("visits.csv", "plot,soil\nA1,clay\nA1,sand\n")

    assert _summarize("visits.csv", "plot", []).conflicts == []


def test_every_conflicting_property_is_named_in_the_order_proposed_with_an_example(
    source,
):
    source(
        "visits.csv",
        "plot,soil,slope,owner\nA1,clay,3,kim\nA1,sand,4,kim\nA2,clay,3,lee\n",
    )

    summary = _summarize("visits.csv", "plot", ["slope", "owner", "soil"])

    assert _conflicting(summary) == ["slope", "soil"]
    assert [(c.node_key, c.first_value, c.second_value) for c in summary.conflicts] == [
        ("A1", "3", "4"),
        ("A1", "clay", "sand"),
    ]


def test_an_example_shows_the_two_smallest_values_of_the_earliest_conflicting_key(
    source,
):
    source(
        "visits.csv",
        "plot,soil,slope\nA2,clay,3\nA2,clay,3\nA1,sand,4\nA1,clay,5\nA1,peat,4\n",
    )

    summary = _summarize("visits.csv", "plot", ["soil", "slope"])

    # A1's soil values sort clay < peat < sand; the two smallest are shown.
    assert summary.conflicts[0] == KeyExample("soil", "A1", "clay", "peat")


def test_a_blank_cell_is_a_second_value(source):
    """The loader writes a present blank cell over an earlier value (KG-22)."""
    source("blank.csv", "plot,soil\nA1,clay\nA1,\n")

    assert _conflicting(_summarize("blank.csv", "plot", ["soil"])) == ["soil"]


def test_a_row_too_short_to_reach_a_property_is_not_a_second_value(source):
    """The loader skips the write, so the node keeps the earlier value (KG-22)."""
    source("visits.csv", "plot,soil,slope\nA1,clay,3\nA1,clay\n")

    assert _summarize("visits.csv", "plot", ["soil", "slope"]).conflicts == []


# --- which properties are judged ----------------------------------------------


def test_the_key_column_and_repeated_names_are_not_read_as_properties(
    source, monkeypatch
):
    source("visits.csv", "plot,soil\nA1,clay\nA1,clay\n")
    read = []
    real = node_key_check.summarize_key_groups

    def spy(file_path, key_column, value_column, **kwargs):
        read.append(value_column)
        return real(file_path, key_column, value_column, **kwargs)

    monkeypatch.setattr(node_key_check, "summarize_key_groups", spy)

    _summarize("visits.csv", "plot", ["plot", "soil", "soil"])

    assert read == ["soil"]


def test_a_property_the_file_does_not_have_is_skipped(source):
    source("visits.csv", "plot,soil\nA1,clay\nA1,sand\n")

    summary = _summarize("visits.csv", "plot", ["no_such_column", "soil"])

    assert _conflicting(summary) == ["soil"]


def test_unreadable_properties_skip_the_agreement_check_but_not_the_blank_check(
    source, no_property_reads
):
    source("visits.csv", "plot,soil\nA1,clay\nA1,sand\n,clay\n")

    summary = _summarize("visits.csv", "plot", None)

    assert summary.blank_count == 1
    assert summary.conflicts == []


# --- failures come back as results, never as raises ----------------------------


def test_a_key_column_the_file_lacks_is_an_error_result(source):
    source("visits.csv", "plot,soil\nA1,clay\n")

    summary, error = summarize_node_key("visits.csv", "site", ["soil"])

    assert summary is None
    assert error["status"] == "error"
    assert "'site'" in error["error_message"]


def test_a_missing_file_is_an_error_result(source):
    summary, error = summarize_node_key("gone.csv", "plot", [])

    assert summary is None
    assert error["status"] == "error"


def test_a_header_that_cannot_be_read_is_an_error_result(source, monkeypatch):
    source("visits.csv", "plot,soil\nA1,clay\nA1,sand\n")

    def broken(file_path):
        raise OSError("disk fell over")

    monkeypatch.setattr(node_key_check, "read_csv_header", broken)

    summary, error = summarize_node_key("visits.csv", "plot", ["soil"])

    assert summary is None
    assert "disk fell over" in error["error_message"]


# --- the refusal text ---------------------------------------------------------


@pytest.mark.parametrize("distinct", [3, 1])
def test_a_clean_summary_is_not_refused(distinct):
    summary = NodeKeySummary(3, distinct, 0, [])

    assert node_key_refusal("Plot", "plots.csv", "plot", summary) is None


def test_a_blank_key_is_refused_with_the_counts_and_a_way_forward():
    text = node_key_refusal(
        "Plot", "plots.csv", "plot", NodeKeySummary(2155, 830, 3, [])
    )

    assert text is not None
    assert "'plot'" in text and "plots.csv" in text and "Plot" in text
    assert "2155 rows, 830 distinct values, 3 blank" in text
    assert "a blank line at the end of the file counts" in text
    assert "another key" in text and "relationship" in text
    # Whitespace-only keys neither merge nor fail, so the text never says every
    # blank row merges.
    assert "every blank row" not in text


def test_a_collapsing_key_is_refused_naming_every_property_and_one_example():
    summary = NodeKeySummary(
        2155,
        830,
        0,
        [
            KeyExample("unitPrice", "10248", "14.0", "9.8"),
            KeyExample("quantity", "10249", "9", "40"),
        ],
    )

    text = node_key_refusal("OrderDetail", "order_details.csv", "orderID", summary)

    assert text is not None
    assert "2155 rows, 830 distinct values, 0 blank" in text
    assert "'unitPrice', 'quantity'" in text
    assert "'10248'" in text and "'14.0'" in text and "'9.8'" in text
    assert "'10249'" not in text, "only the first property is shown with an example"
    assert "another key" in text
    assert "relationship" in text
    assert "leave those properties off the node" in text


def test_a_long_or_multiline_example_value_cannot_swamp_or_break_the_message():
    summary = NodeKeySummary(
        4, 2, 0, [KeyExample("note", "A1", "x" * 500, "line one\nline two")]
    )

    text = node_key_refusal("Plot", "plots.csv", "plot", summary)

    assert text is not None
    assert "x" * 500 not in text
    assert "\n" not in text
    assert "'line one\\nline two'" in text
