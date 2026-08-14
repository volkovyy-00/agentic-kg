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
    assert "plot_id" in problems[0] and "plots.csv" in problems[0]
    assert "readings.csv" in problems[0]
    assert unverified == []


def test_the_key_itself_is_always_reachable(survey_source):
    """Catches a stage 3 that ignores the node key and goes straight to properties."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_id", ["canopy"]), APPROVED
    )
    assert problems == [] and unverified == []


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
    assert problems == [] and unverified == []


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


def test_a_property_on_another_files_node_does_not_confer_reachability(survey_source):
    """Case 8: catches an unrestricted stage 4. Reading is built from readings.csv,
    where plot_id trivially never collapses (one row per reading_id) -- that is a
    fact about readings.csv, not evidence that plots.csv preserved its identifier."""
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "reading_id",
        "properties": ["plot_id"],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1 and "plot_id" in problems[0]


def test_a_coincidental_key_on_an_unrelated_file_does_not_confer_reachability(
    survey_source,
):
    """Case 10: catches an unrestricted stage 3. Reading is keyed by a column that
    happens to be spelled plot_id but belongs to a different file."""
    plan = _plot_node("plot_label", ["canopy"])
    plan["Reading"] = {
        "construction_type": "node",
        "source_file": "readings.csv",
        "label": "Reading",
        "unique_column_name": "plot_id",
        "properties": [],
    }
    problems, unverified = rr.check_reference_columns_are_reachable(plan, APPROVED)
    assert len(problems) == 1 and "plot_id" in problems[0]


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
    assert len(problems) == 1 and "plot_id" in problems[0]
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
    assert "drop" not in text.lower()


def test_the_report_has_no_blank_slot_when_every_file_is_a_home_file(survey_source):
    """Catches _report leaving blank slots ('appears in ' / 'joining  to') when the
    column is per-row unique in every file that shares it, so referencing is empty --
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
    assert ", but " in text and ", but  " not in text
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
    *string* 'properties' value must not silently substring-match instead of
    being treated as absent."""
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
    assert isinstance(problems, list) and isinstance(unverified, list)
    assert len(problems) == 1 and "plot_id" in problems[0]

    string_properties_plan = _plot_node("plot_label", "plot_id")
    problems, unverified = rr.check_reference_columns_are_reachable(
        string_properties_plan, APPROVED
    )
    assert len(problems) == 1 and "plot_id" in problems[0]


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
    assert isinstance(problems, list) and isinstance(unverified, list)


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
    assert isinstance(problems, list) and isinstance(unverified, list)
    # 'ghost.csv' is unreadable and could have been plot_id's home file, so the
    # verdict is withheld rather than refused -- the evidence rule, not a crash.
    assert problems == []
    assert any("reachability of 'plot_id' was not verified" in n for n in unverified)


def test_a_non_iterable_approved_file_list_does_not_raise(survey_source):
    """Catches iterating approved_files without checking its type: the contract is two
    lists for ANY input, and a raise here dies inside a plan presentation."""
    problems, unverified = rr.check_reference_columns_are_reachable(
        _plot_node("plot_label", ["canopy"]), 5
    )
    assert problems == [] and unverified == []
