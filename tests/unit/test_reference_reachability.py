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
from agentic_kg.common.tool_result import tool_error
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
    assert unreadable == []
    assert notes == []


def test_a_column_shared_but_unique_in_neither_file_has_no_home(survey_source):
    """Case 2: catches dropping the candidacy test. 'tally' is in both files and
    per-row unique in neither, so it is not an identifier at all."""
    homes, complete, notes, _ = rr._home_files("tally", ["plots.csv", "readings.csv"])
    assert homes == []
    assert complete is True
    assert notes == []


def test_the_home_file_is_the_one_where_the_column_is_per_row_unique(survey_source):
    """Catches treating any file containing the column as a home file: plot_id is
    unique in plots.csv and repeats in readings.csv."""
    homes, complete, notes, _ = rr._home_files("plot_id", ["plots.csv", "readings.csv"])
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
    assert len(notes) == 1
    assert "absent.csv" in notes[0]
    assert columns["plot_id"] == ["plots.csv"]


def test_home_files_returns_the_values_of_every_readable_file_without_blanks(
    survey_source,
):
    """Catches a value set that keeps a blank, or that covers only the home files: a
    witness built from a file where the column repeats needs that file's own values,
    and an unreadable file has none to give."""
    with survey_source.open("/src/readings.csv", "w") as handle:
        handle.write("reading_id,plot_id,tally\nR-1,PL-1,3\nR-2,,3\nR-3,PL-3,5\n")
    _, _, _, value_sets = rr._home_files(
        "plot_id", ["plots.csv", "readings.csv", "absent.csv"]
    )
    assert value_sets == {
        "plots.csv": {"PL-1", "PL-2", "PL-3", "PL-4"},
        "readings.csv": {"PL-1", "PL-3"},
    }


def test_an_unreadable_value_read_marks_evidence_incomplete(survey_source):
    """Catches an implementation that silently treats an unreadable file as 'not a
    home file'. The flag is what later stops a refusal being built on a failed read."""
    homes, complete, notes, _ = rr._home_files("plot_id", ["plots.csv", "absent.csv"])
    assert homes == ["plots.csv"]
    assert complete is False
    assert len(notes) == 1
    assert "absent.csv" in notes[0]


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


APPROVED = ["plots.csv", "readings.csv"]


def test_a_key_that_strands_the_referencing_file_is_reported(survey_source):
    """Case 1, the base case: catches a check that never fires. Keying Plot by the
    repeating label leaves readings.csv's plot_id pointing at nothing."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), APPROVED
    )
    assert len(problems) == 1
    assert "plot_id" in problems[0]
    assert "plots.csv" in problems[0]
    assert "readings.csv" in problems[0]
    assert unverified == []


def test_the_key_itself_is_always_reachable(survey_source):
    """Catches a check that ignores the node key and goes straight to properties."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_id", ["canopy"]), APPROVED
    )
    assert problems == []
    assert unverified == []


def test_a_property_that_survives_collapsing_is_reachable(survey_source):
    """Case 5: catches treating 'not the key' as unreachable. Keyed by plot_slug,
    which is also per-row unique, so each Plot keeps its own plot_id."""
    fs = survey_source
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write(
            "plot_label,plot_id,plot_slug\n"
            "ridge,PL-1,ridge-a\n"
            "ridge,PL-2,ridge-b\n"
            "hollow,PL-3,hollow-a\n"
        )
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_slug", ["plot_id"]), APPROVED
    )
    assert problems == []
    assert unverified == []


def test_a_property_that_does_not_survive_collapsing_is_reported(survey_source):
    """Case 6: catches accepting {key} + properties without the collapse test. Four
    rows collapse onto two labels, so each Plot keeps one arbitrary plot_id and the
    join matches almost nothing -- silently."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["plot_id"]), APPROVED
    )
    assert len(problems) == 1
    assert "plot_id" in problems[0]
    assert unverified == []


@pytest.mark.parametrize("unique_column", [None, "", ["plot_slug"]])
def test_a_home_rule_without_a_usable_key_leaves_the_column_unverified(
    survey_source, unique_column
):
    """Catches a collapse check fed a missing, empty or non-string key: that rule
    gives no evidence either way, so the column is unverified, never refused, and
    the note says why instead of 'Column(s) [None] are not in ...'."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node(unique_column, ["plot_id"]), APPROVED
    )
    assert problems == []
    assert any("plot_id" in n and "unique_column_name" in n for n in unverified)


@pytest.mark.parametrize("source_file", [None, ["plots.csv"]])
def test_a_rule_without_a_usable_source_file_gives_no_evidence(
    survey_source, source_file
):
    """Catches a guard that checks only the key. The caller only passes rules whose
    source_file is an approved file, so this branch is reachable only directly."""
    rule = {**_plot_node("plot_slug", ["plot_id"])["Plot"], "source_file": source_file}
    assert rr._property_failure(rule, "plot_id") == (
        None,
        "the rule has no usable 'source_file'",
    )


def test_a_retained_property_covering_only_part_of_the_identifier_file_is_reported(
    survey_source,
):
    """Case 8: the same shortfall as keying by the column, reached through a retained
    property. Reading keeps plot_id as a property and it survives collapsing (one row
    per reading_id) -- but only for the 2 ids readings.csv holds, the same 2 of
    plots.csv's 4 that keying by it carries. Catches a property path that works out
    its values differently from the key path, and a clause claiming the property
    carries values: it is never read, so all the message can say is 'at most'."""
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "reading_id",
        "properties": ["plot_id"],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert "plot_id" in problems[0]
    assert "retains 'plot_id', so could carry at most 2 of the 4" in problems[0]
    assert "carries 2 of the 4" not in problems[0]


def test_a_key_covering_only_part_of_the_identifier_file_is_reported(survey_source):
    """Case 10: catches the home-file proxy removed naively, so that any node keyed by
    the column passes. Reading is keyed by plot_id but built from readings.csv, which
    holds 2 of the 4 ids plots.csv identifies, so the join can reach half the plots at
    most. The refusal states that shortfall and keeps the properties every refusal
    must have."""
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "plot_id",
        "properties": [],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    text = problems[0]
    assert "carries 2 of the 4" in text
    assert "'PL-2', 'PL-4'" in text
    assert "cannot be built at all" in text
    assert "keying a node built from" in text
    assert "adding a node construction" in text
    assert "drop" not in text.lower()


def _write_detail_with_an_id_plots_lacks(fs):
    """detail.csv identifies rows by plot_id too, but lists PL-7, which plots.csv does
    not: two home files whose values diverge."""
    with fs.open("/src/detail.csv", "w") as handle:
        handle.write("plot_id,notes\nPL-1,a\nPL-2,b\nPL-3,c\nPL-7,z\n")


def test_an_id_the_identifier_file_never_lists_does_not_block_approval(
    survey_source,
):
    """Catches requiring coverage of the files where the column merely repeats. A
    reading points at PL-9, which plots.csv never lists. Neither route out in the
    refusal could conjure that plot -- both build from plots.csv -- so refusing would
    leave the model no move except dropping the relationship."""
    with survey_source.open("/src/readings.csv", "w") as handle:
        handle.write("reading_id,plot_id,tally\nR-1,PL-1,3\nR-2,PL-1,3\nR-3,PL-9,5\n")
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_id", ["canopy"]), APPROVED
    )
    assert problems == []
    assert unverified == []


def test_a_second_identifier_file_with_extra_values_does_not_block_approval(
    survey_source,
):
    """Catches requiring every home file to be covered. detail.csv also identifies
    rows by plot_id and lists PL-7, which the node built from plots.csv does not
    carry. That is the same dangling id a repeating file may hold, and it is treated
    the same way: two files that each happen to have a unique column of the same
    name must not block a plan whose node covers one of them."""
    _write_detail_with_an_id_plots_lacks(survey_source)
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_id", ["canopy"]), ["plots.csv", "detail.csv"]
    )
    assert problems == []
    assert unverified == []


def test_a_retained_property_that_several_nodes_share_does_not_carry_the_column(
    survey_source,
):
    """Catches accepting a retained property because it survives collapsing. Reading
    is keyed per reading and keeps plot_id, and readings.csv holds all four ids --
    but PL-1 sits on two Reading nodes, so a relationship joining on it lands on
    readings, not on one plot."""
    with survey_source.open("/src/readings.csv", "w") as handle:
        handle.write(
            "reading_id,plot_id\nR-1,PL-1\nR-2,PL-1\nR-3,PL-2\nR-4,PL-3\nR-5,PL-4\n"
        )
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "reading_id",
        "properties": ["plot_id"],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert "more than one node" in problems[0]
    assert unverified == []


def test_a_retained_property_on_one_node_per_value_carries_the_column(
    survey_source,
):
    """Catches rejecting every retained property from a file where the column
    repeats. Here it repeats only because R-1 is listed twice: each plot_id still
    sits on exactly one Reading node, so that node is a join target for all four."""
    with survey_source.open("/src/readings.csv", "w") as handle:
        handle.write(
            "reading_id,plot_id\nR-1,PL-1\nR-1,PL-1\nR-2,PL-2\nR-3,PL-3\nR-4,PL-4\n"
        )
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "reading_id",
        "properties": ["plot_id"],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert problems == []
    assert unverified == []


def test_a_retaining_rule_that_could_not_close_the_gap_is_not_read(survey_source):
    """Catches a failed read on a rule that could never change the verdict turning a
    provable refusal into 'unverified'. Reading's key is misspelt, so its pairs
    cannot be read -- but readings.csv holds only 2 of plots.csv's 4 ids, so no
    answer from that read could make Reading carry them all."""
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "READING_ID",
        "properties": ["plot_id"],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert unverified == []


def test_a_failed_retention_is_explained_alongside_a_shortfall(survey_source):
    """Catches dropping one explanation when there are two. Plot keeps plot_id but it
    does not survive collapsing onto plot_label, and Reading is keyed by plot_id but
    holds only 2 of the 4 ids: the refusal must say both."""
    plan = _plot_node("plot_label", ["plot_id"])
    plan["Reading"] = _keyed_by_plot_id("Reading", "readings.csv")
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert "does not survive collapsing" in problems[0]
    assert "carries 2 of the 4" in problems[0]


def test_the_no_carrier_sentence_names_the_files_it_looked_at(survey_source):
    """Catches a blanket 'no node is keyed by it' that the plan contradicts. Ghost
    IS keyed by plot_id, but built from './plots.csv', which is not an approved file,
    so it carries nothing the check can see; the sentence must say where it looked."""
    plan = {"Ghost": _keyed_by_plot_id("Ghost", "./plots.csv")}
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert (
        "no node built from 'plots.csv', 'readings.csv' is keyed by 'plot_id'"
        in problems[0]
    )
    assert "no node is keyed by" not in problems[0]


def test_a_value_quoted_in_the_refusal_cannot_break_its_layout(survey_source):
    """Catches pasting raw cell text into a message the model reads as a list of
    problems, one per line. A cell holding a quote, a newline and '- ' would forge
    a second bullet; a very long cell would be copied whole."""
    long_value = "x" * 300
    with survey_source.open("/src/plots.csv", "w") as handle:
        handle.write(
            "plot_label,plot_id\n"
            "ridge,PL-1\n"
            'ridge,"PL-2\'\n- forged problem"\n'
            f"hollow,PL-3\nhollow,{long_value}\n"
        )
    plan = {"Reading": _keyed_by_plot_id("Reading", "readings.csv")}
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert "\n" not in problems[0]
    assert long_value not in problems[0]


def test_a_failed_value_read_downgrades_a_shortfall_to_unverified(
    survey_source, monkeypatch
):
    """Catches a coverage loop that drops the evidence flag. No node carries plot_id,
    but bad.csv carries the column with a readable header and a value read that
    fails -- it could have been a home file a node covers, so the refusal is withheld
    and the column is reported unverified instead."""
    with survey_source.open("/src/bad.csv", "w") as handle:
        handle.write("plot_id,x\nPL-1,a\n")
    real_read = rr.summarize_column

    def read_or_fail(path, column):
        if path == "bad.csv":
            return None, tool_error("simulated read failure")
        return real_read(path, column)

    monkeypatch.setattr(rr, "summarize_column", read_or_fail)
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), ["plots.csv", "readings.csv", "bad.csv"]
    )
    assert problems == []
    assert any("plot_id" in note and "bad.csv" in note for note in unverified)


def test_a_failed_read_of_the_only_possible_identifier_file_is_unverified(
    survey_source, monkeypatch
):
    """Catches treating 'no home file is short' as 'reachable' when no home file
    could be confirmed at all. plots.csv is where plot_id identifies rows, but its
    value read fails, so no file is known to be a home file and nothing is known to
    be covered. The plan keys Plot by plot_label, so the column may be stranded: it
    must be reported unverified, never passed silently."""
    real_read = rr.summarize_column

    def read_or_fail(path, column):
        if path == "plots.csv":
            return None, tool_error("simulated read failure")
        return real_read(path, column)

    monkeypatch.setattr(rr, "summarize_column", read_or_fail)
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), APPROVED
    )
    assert problems == []
    assert any("plot_id" in note and "plots.csv" in note for note in unverified)


def test_a_node_rule_with_an_unhashable_source_file_does_not_raise(survey_source):
    """Catches looking a rule's source_file up in a dict without checking its type: a
    list is unhashable and would raise inside a plan presentation. The contract is two
    lists for any input, and nothing carries the column here, so it is still refused."""
    plan = {
        "Ghost": {
            "construction_type": "node",
            "source_file": ["plots.csv"],
            "label": "Ghost",
            "unique_column_name": "plot_id",
            "properties": [],
        }
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert isinstance(unverified, list)


def _keyed_by_plot_id(label, source_file):
    return {
        "construction_type": "node",
        "source_file": source_file,
        "label": label,
        "unique_column_name": "plot_id",
        "properties": [],
    }


def test_a_key_overlapping_none_of_the_identifier_files_values_says_so(
    survey_source,
):
    """Catches sending a node that carries the column but overlaps none of the values
    to the 'no node is keyed by it' sentence, which would then be false: Reading IS
    keyed by plot_id, in a format that matches nothing in plots.csv."""
    with survey_source.open("/src/readings.csv", "w") as handle:
        handle.write("reading_id,plot_id,tally\nR-1,X-1,3\nR-2,X-1,3\nR-3,X-2,5\n")
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = _keyed_by_plot_id("Reading", "readings.csv")
    problems, _ = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1
    assert "none of the 4" in problems[0]
    assert "no node is keyed by" not in problems[0]


def test_every_identifier_file_gets_its_own_clause(survey_source):
    """Catches reporting only one home file, or naming one shared closest node for
    both. plots.csv (PL-1..PL-4) is closest to Reading; detail.csv (PL-5..PL-8) is
    closest to Visit. Covering either would do, so the fix clause names both."""
    fs = survey_source
    with fs.open("/src/detail.csv", "w") as handle:
        handle.write("plot_id,notes\nPL-5,a\nPL-6,b\nPL-7,c\nPL-8,d\n")
    with fs.open("/src/visits.csv", "w") as handle:
        handle.write("visit_id,plot_id\nV-1,PL-5\nV-2,PL-5\nV-3,PL-6\nV-4,PL-7\n")
    plan = {
        "Reading": _keyed_by_plot_id("Reading", "readings.csv"),
        "Visit": _keyed_by_plot_id("Visit", "visits.csv"),
    }
    approved = ["plots.csv", "readings.csv", "detail.csv", "visits.csv"]
    problems, unverified = rr.check_reference_columns_are_reachable(plan, approved)
    assert len(problems) == 1
    text = problems[0]
    assert (
        "'Reading' (built from 'readings.csv', keyed by 'plot_id') "
        "carries 2 of the 4 'plot_id' values 'plots.csv' holds"
    ) in text
    assert (
        "'Visit' (built from 'visits.csv', keyed by 'plot_id') "
        "carries 3 of the 4 'plot_id' values 'detail.csv' holds"
    ) in text
    assert "'plots.csv', 'detail.csv'" in text.split("Fix it", 1)[1]


def test_two_witnesses_that_each_hold_part_of_an_identifier_files_values_are_not_combined(
    survey_source,
):
    """Catches judging coverage against the UNION of the nodes' values. Reading holds
    PL-1..PL-3 and Visit holds PL-4, so together they hold everything plots.csv does --
    but a relationship joins to one label, and neither node alone carries all four.
    The split is uneven so the closest node is never a tie."""
    fs = survey_source
    with fs.open("/src/readings.csv", "w") as handle:
        handle.write(
            "reading_id,plot_id,tally\nR-1,PL-1,3\nR-2,PL-2,3\nR-3,PL-3,5\nR-4,PL-1,5\n"
        )
    with fs.open("/src/visits.csv", "w") as handle:
        handle.write("visit_id,plot_id\nV-1,PL-4\nV-2,PL-4\n")
    plan = {
        "Reading": _keyed_by_plot_id("Reading", "readings.csv"),
        "Visit": _keyed_by_plot_id("Visit", "visits.csv"),
    }
    approved = ["plots.csv", "readings.csv", "visits.csv"]
    problems, unverified = rr.check_reference_columns_are_reachable(plan, approved)
    assert len(problems) == 1
    assert "carries 3 of the 4" in problems[0]
    assert "'PL-4'" in problems[0]


def test_incomplete_evidence_downgrades_a_refusal_to_unverified(survey_source):
    """Case 7, the cross-stage evidence leak. plot_id is unique in plots.csv and
    ALSO in an unreadable file the node is built from. A per-stage fail-open
    short-circuits on plots.csv, never credits the unreadable file as a home file,
    and REFUSES A CORRECT PLAN because of a failed read. This is the only test that
    catches that; see TRAP 3."""
    fs = survey_source
    with fs.open("/src/quadrats.csv", "w") as handle:
        handle.write("plot_id,shade\nPL-1,low\nPL-2,high\n")
    plan = {
        "Quadrat": {
            "construction_type": "node",
            "source_file": "quadrats.csv",
            "label": "Quadrat",
            "unique_column_name": "plot_id",
            "properties": ["shade"],
        }
    }
    approved = ["plots.csv", "readings.csv", "quadrats.csv"]
    fs.rm("/src/quadrats.csv")
    problems, unverified = rr.check_reference_columns_are_reachable(plan, approved)
    assert problems == []
    # Two notes, not one: stage 1 reports the unreadable header, and the candidate
    # reports its own downgraded verdict. Pinning a count here would break on the
    # first extra note and says nothing about the behaviour under test.
    assert any("plot_id" in note for note in unverified)


def test_one_unreadable_candidate_does_not_suppress_another(survey_source):
    """Case 9: catches abort-on-first-error. The unreadable file drops out at
    stage 1, so its candidate never actually forms -- but a readable, unrelated
    candidate is still reported rather than swallowed by the earlier failure."""
    fs = survey_source
    with fs.open("/src/sites.csv", "w") as handle:
        handle.write("site_label,site_id\nnorth,ST-1\nnorth,ST-2\n")
    with fs.open("/src/visits.csv", "w") as handle:
        handle.write("visit_id,site_id\nV-1,ST-1\nV-2,ST-1\n")
    plan = _plot_node("plot_label", ["canopy"])
    plan["Site"] = {
        "construction_type": "node",
        "source_file": "sites.csv",
        "label": "Site",
        "unique_column_name": "site_label",
        "properties": [],
    }
    approved = ["plots.csv", "readings.csv", "sites.csv", "visits.csv"]
    fs.rm("/src/visits.csv")
    problems, unverified = rr.check_reference_columns_are_reachable(plan, approved)
    assert len(problems) == 1
    assert "plot_id" in problems[0]
    assert any("visits.csv" in note for note in unverified)


def test_the_report_names_both_routes_out(survey_source):
    """Catches a message that prescribes only re-keying. The check is agnostic
    between re-keying and adding a second node construction, and the model can only
    act on what the message offers."""
    problems, _ = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), APPROVED
    )
    text = problems[0]
    assert "cannot be built at all" in text
    assert "keying a node built from" in text
    assert "adding a node construction" in text
    assert "a label of its own" in text
    assert "drop" not in text.lower()


def test_the_report_has_no_blank_slot_when_every_file_is_a_home_file(survey_source):
    """Catches _report leaving blank slots ('appears in ' / 'joining  to') when the
    column is per-row unique in every file that shares it, so repeating is empty --
    a legitimate 1:1 split with no 'other' file left to name."""
    fs = survey_source
    with fs.open("/src/detail.csv", "w") as handle:
        handle.write("plot_id,notes\nPL-1,a\nPL-2,b\nPL-3,c\nPL-4,d\n")
    problems, _ = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), ["plots.csv", "detail.csv"]
    )
    assert len(problems) == 1
    text = problems[0]
    assert "plot_id" in text
    assert "cannot be built at all" in text
    assert "keying a node built from" in text
    assert "adding a node construction" in text
    assert ", but " in text
    assert ", but  " not in text
    assert "joining  to" not in text
    assert " to  " not in text


def test_an_empty_approved_file_list_produces_nothing(survey_source):
    """Catches a check that reads files it was never given. Every existing approval
    test has no approved_file_list in state and must stay green."""
    assert rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", []), []
    ) == (
        [],
        [],
    )


def test_malformed_rule_field_types_do_not_raise_or_substring_match(survey_source):
    """Case 11: catches skipping the isinstance guards on a rule's fields. An
    unreadable rule's non-string 'unique_column_name' (a list) built into a set
    literal alongside a non-list 'properties' (an int) must not raise
    'unhashable type: list' / 'argument of type int is not iterable', and a
    *string* 'properties' value must not silently substring-match. It is not
    read as absent either: it is unreadable, so a refusal is withheld."""
    plan = {
        "Ghost": {
            "construction_type": "node",
            "source_file": "ghost.csv",
            "label": "Ghost",
            "unique_column_name": ["x"],
            "properties": 5,
        }
    }
    approved = ["plots.csv", "readings.csv", "ghost.csv"]
    problems, unverified = rr.check_reference_columns_are_reachable(plan, approved)
    assert isinstance(problems, list)
    assert isinstance(unverified, list)
    assert len(problems) == 1
    assert "plot_id" in problems[0]

    string_properties_plan = _plot_node("plot_label", "plot_id")
    problems, unverified = rr.check_reference_columns_are_reachable(
        string_properties_plan, APPROVED
    )
    # A substring match would credit the rule with plot_id and say nothing at
    # all; the note is what tells "matched by characters" from "withheld".
    assert problems == []
    assert any(
        "declares 'properties' in a form that could not be read" in n
        for n in unverified
    )


def test_a_read_failure_elsewhere_does_not_block_a_confirmed_reachable_candidate(
    survey_source,
):
    """Catches conflating 'some read failed' with 'this candidate's evidence is
    incomplete'. plot_id is fully confirmed reachable via plot_slug plus a
    surviving property; a wholly unrelated approved file that fails to read
    must add its own note but must not downgrade THIS column's verdict -- a
    failed read withholds evidence, it never manufactures a problem either."""
    fs = survey_source
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write(
            "plot_label,plot_id,plot_slug\n"
            "ridge,PL-1,ridge-a\n"
            "ridge,PL-2,ridge-b\n"
            "hollow,PL-3,hollow-a\n"
        )
    approved = ["plots.csv", "readings.csv", "absent.csv"]
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_slug", ["plot_id"]), approved
    )
    assert problems == []
    assert any("absent.csv" in note for note in unverified)


def test_a_malformed_plan_does_not_raise(survey_source):
    """Catches an implementation that assumes well-formed rules. The contract is two
    lists, always -- a plan presentation must never fail on a bad entry."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        {"Plot": "not-a-dict"}, APPROVED
    )
    assert isinstance(problems, list)
    assert isinstance(unverified, list)


def test_one_file_listed_twice_is_not_two_files(survey_source):
    """Catches counting entries instead of distinct files: a duplicated approved path
    made a single file satisfy the two-file condition and refuse a plan over a column
    nothing else references, in a message naming the same file twice."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), ["plots.csv", "plots.csv"]
    )
    assert problems == []
    assert unverified == []


def test_a_column_repeated_inside_one_header_is_not_two_files(survey_source):
    """Catches recording one entry per header occurrence: a CSV whose header repeats a
    name made that one file look like two, refusing a plan over a column that exists in
    no other file."""
    fs = survey_source
    with fs.open("/src/twin.csv", "w") as handle:
        handle.write("code,code,label\nA,X,alpha\nB,Y,beta\n")
    problems, unverified = rr.check_reference_columns_are_reachable(
        {
            "Twin": {
                "construction_type": "node",
                "source_file": "twin.csv",
                "label": "Twin",
                "unique_column_name": "label",
                "properties": [],
            }
        },
        ["twin.csv"],
    )
    assert problems == []
    assert unverified == []


def test_an_unhashable_property_entry_does_not_raise(survey_source):
    """Catches guarding a rule's 'properties' container but not its elements: a dict
    inside the list is unhashable, and the unreadable-source scan splats that list into
    a set literal -- raising 'unhashable type: dict' inside a plan presentation."""
    plan = {
        "Ghost": {
            "construction_type": "node",
            "source_file": "ghost.csv",
            "label": "Ghost",
            "unique_column_name": "plot_id",
            "properties": [{"name": "plot_id"}, ["plot_id"], "canopy"],
        }
    }
    problems, unverified = rr.check_reference_columns_are_reachable(
        plan, ["plots.csv", "readings.csv", "ghost.csv"]
    )
    assert isinstance(problems, list)
    assert isinstance(unverified, list)
    # 'ghost.csv' is unreadable and could have been plot_id's home file, so the
    # verdict is withheld rather than refused -- the evidence rule, not a crash.
    assert problems == []
    assert any("reachability of 'plot_id' was not verified" in n for n in unverified)


def test_a_non_string_approved_file_entry_is_skipped_and_does_not_raise(survey_source):
    """Catches guarding the approved list's type but not its entries: an unhashable
    entry raises on the dedupe before the read is ever guarded, and a hashable
    non-string one reaches the reader and yields a note about a file nobody named."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_id", ["canopy"]),
        ["plots.csv", ["nested"], {"k": "v"}, 5, None],
    )
    assert problems == []
    assert unverified == []


def test_a_non_iterable_approved_file_list_does_not_raise(survey_source):
    """Catches iterating approved_files without checking its type: the contract is two
    lists for ANY input, and a raise here dies inside a plan presentation."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), 5
    )
    assert problems == []
    assert unverified == []


def test_a_header_only_file_is_not_an_identifier_home(survey_source):
    """Zero rows is not a home: _home_files' at-least-one-row condition is the
    one place uniqueness and identity differ, and it must survive the rewrite.

    Keep this file's vocabulary neutral -- tests/unit/test_generality.py asserts
    the bundled dataset's column names are absent from this module."""
    fs = survey_source
    with fs.open("/src/header_only.csv", "w") as handle:
        handle.write("plot_label,canopy\n")
    homes, evidence_complete, notes, value_sets = rr._home_files(
        "plot_label", ["header_only.csv"]
    )
    assert homes == []
    assert evidence_complete is True
    assert notes == []
    assert value_sets == {"header_only.csv": set()}


def test_a_zero_row_source_withholds_evidence_instead_of_supplying_it(survey_source):
    """A zero-row source has no conflicting group and no value under two keys
    because it has no rows, so both of _property_failure's checks pass
    vacuously and the caller would read (None, None) as 'covered'. It must
    return a message instead, exactly as an unreadable source does -- the
    docstring's rule is that it withholds evidence, never supplies it.

    _property_failure cannot be reached this way through
    check_reference_columns_are_reachable, because _home_files admits a home
    only when row_count is non-zero and no non-empty home set is a subset of
    this file's empty one. That makes this the only place the contract can be
    pinned, and the reason to pin it: the guard must not silently depend on a
    condition living in another function."""
    fs = survey_source
    with fs.open("/src/empty_export.csv", "w") as handle:
        handle.write("plot_label,canopy\n")
    rule = {
        "construction_type": "node",
        "source_file": "empty_export.csv",
        "label": "Plot",
        "unique_column_name": "plot_label",
        "properties": ["plot_label", "canopy"],
    }
    failure, error_message = rr._property_failure(rule, "canopy")
    assert failure is None
    assert error_message == "the source has no data rows"


def test_a_zero_byte_file_passes_the_reachability_check_in_silence(survey_source):
    """Deliberate asymmetry with the file tools, and pre-existing: _columns_by_file
    reads headers through read_csv_header, which returns [] for a zero-byte file
    WITHOUT raising, so the file contributes no column names and never reaches
    _home_files. Pinned so the rewrite does not 'fix' it into a new answer."""
    fs = survey_source
    with fs.open("/src/empty.csv", "w") as handle:
        handle.write("")
    plan = {
        "r1": {
            "construction_type": "node",
            "source_file": "empty.csv",
            "label": "Thing",
            "unique_column_name": "plot_label",
            "properties": ["plot_label"],
        }
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, ["empty.csv"])
    assert problems == []
    assert unverified == []


def test_a_failure_part_way_through_a_read_becomes_a_note_not_a_raise(
    survey_source, monkeypatch
):
    """reference_reachability promises that nothing in it raises. A mid-read
    failure must arrive as evidence_complete=False plus a note.

    Patched by dotted path rather than through an imported module object, so this
    needs no new import in either the test module or the production one."""

    def failing_batches(path, *args, **kwargs):
        yield ["plot_label"], [{"plot_label": "ridge"}]
        raise OSError("source went away")

    monkeypatch.setattr("agentic_kg.tools.file_tools.read_csv_batches", failing_batches)
    homes, evidence_complete, notes, value_sets = rr._home_files(
        "plot_label", ["plots.csv"]
    )
    assert homes == []
    assert evidence_complete is False
    assert len(notes) == 1
    assert "plots.csv" in notes[0]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({}, []),
        ({"properties": None}, []),
        ({"properties": []}, []),
        ({"properties": ["a", "b"]}, ["a", "b"]),
        ({"properties": 5}, None),
        ({"properties": 0}, None),
        ({"properties": "plot_id"}, None),
        ({"properties": ""}, None),
        ({"properties": {"plot_id": 1}}, None),
        ({"properties": {}}, None),
        ({"properties": True}, None),
        ({"properties": [1, "a"]}, None),
        ({"properties": [{"a": 1}, "b"]}, None),
        ({"properties": [["a"]]}, None),
        ({"properties": [None]}, None),
    ],
    ids=repr,
)
def test_declared_properties_tells_unreadable_from_absent(rule, expected):
    """None is 'said something unreadable'; [] is 'declares nothing'. The plan
    consistency check and this module share the predicate, so they cannot
    disagree about which is which."""
    assert rr.declared_properties(rule) == expected


@pytest.mark.parametrize("properties", [5, "plot_id", {"plot_id": 1}, ["canopy", 7]])
def test_an_unreadable_properties_value_withholds_a_refusal(survey_source, properties):
    """Catches reading an unreadable 'properties' as [] -- the same plan with
    ['canopy'] is refused (test_a_key_that_strands_the_referencing_file...), but a
    rule that might retain plot_id in a form nobody could read must not be
    refused on that misreading."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", properties), APPROVED
    )
    assert problems == []
    assert any(
        "reachability of 'plot_id' was not verified" in note
        and "'Plot' declares 'properties' in a form that could not be read" in note
        for note in unverified
    )


def test_an_unreadable_properties_value_on_a_rule_keyed_by_the_column_is_not_in_doubt(
    survey_source,
):
    """A rule keyed by plot_id carries it outright, so what its 'properties' say
    cannot change the verdict and must not withhold one."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_id", 5), APPROVED
    )
    assert problems == []
    assert unverified == []
