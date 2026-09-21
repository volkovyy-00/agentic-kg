"""KG-22: a relationship may not join on a node property that holds several
values per node.

Vocabulary-neutral, like test_reference_reachability.py: a field-survey domain.
The bundled example's own reproduction lives in test_construction_plan_tools.py.
These tests call the new module directly where they assert "never raises",
because check_construction_plan_consistency, which find_plan_problems runs first,
raises on some malformed shapes (KG-38).
"""

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools.join_property_check import (
    check_joined_properties_hold_one_value as check,
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


def _node(label, source_file, key, properties):
    return {
        "construction_type": "node",
        "label": label,
        "source_file": source_file,
        "unique_column_name": key,
        "properties": properties,
    }


def _rel(name, from_label, from_column, to_label, to_column):
    return {
        "construction_type": "relationship",
        "relationship_type": name,
        "source_file": "records.csv",
        "from_node_label": from_label,
        "from_node_column": from_column,
        "to_node_label": to_label,
        "to_node_column": to_column,
        "properties": [],
        "property_types": {},
    }


# 'north' holds two species_code values, 'south' one: 1 of 2 nodes conflicts.
WALKS_SEVERAL = (
    "transect_name,species_code,tally\nnorth,SP-1,3\nnorth,SP-2,4\nsouth,SP-1,1\n"
)


def _several_values_plan():
    return {
        "Transect": _node("Transect", "walks.csv", "transect_name", ["species_code"]),
        "Species": _node("Species", "species.csv", "species_code", ["common_name"]),
        "SEEN": _rel("SEEN", "Transect", "species_code", "Species", "species_code"),
    }


# --- the refusal -------------------------------------------------------------


@pytest.fixture
def refusal(source):
    source("walks.csv", WALKS_SEVERAL)
    problems, unverified = check(_several_values_plan())
    assert unverified == []
    assert len(problems) == 1
    return problems[0]


@pytest.mark.parametrize(
    "fragment",
    [
        "'SEEN'",
        "'Transect'",
        "on 'species_code' of",
        "built from 'walks.csv'",
        "keyed by 'transect_name'",
        "more than one value per node",
        "a blank cell counts as a value",
        "in 1 of 2 nodes",
        "adding a node construction from 'walks.csv' keyed by 'species_code'",
        "under a label of its own",
        "joining 'SEEN' on that label's 'species_code'",
    ],
)
def test_the_refusal_is_built_from_the_plans_own_names(refusal, fragment):
    assert fragment in refusal


@pytest.mark.parametrize("word", ["approv", "ready"])
def test_the_refusal_carries_no_approval_framing(refusal, word):
    assert word not in refusal.lower()


def test_the_refusal_never_suggests_dropping_the_relationship(refusal):
    assert "drop" not in refusal.lower()
    assert "remove" not in refusal.lower()


# --- what is not refused -----------------------------------------------------


def test_several_nodes_sharing_one_value_is_not_refused(source):
    source(
        "walks.csv",
        "transect_name,habitat,tally\nnorth,forest,3\nnorth,forest,4\nsouth,forest,1\n",
    )
    plan = {
        "Transect": _node("Transect", "walks.csv", "transect_name", ["habitat"]),
        "Habitat": _node("Habitat", "habitats.csv", "habitat", []),
        "IN": _rel("IN", "Transect", "habitat", "Habitat", "habitat"),
    }
    assert check(plan) == ([], [])


def test_a_join_on_the_nodes_key_is_never_read(source):
    # No source is written at all: a read would come back as a note.
    plan = {
        "Species": _node("Species", "species.csv", "species_code", ["common_name"]),
        "SEEN": _rel("SEEN", "Species", "species_code", "Species", "species_code"),
    }
    assert check(plan) == ([], [])


def test_a_key_that_is_also_listed_as_a_property_is_still_a_key_join(source):
    # No source is written: if the key skip were lost, the read would be a note.
    plan = {
        "Species": _node(
            "Species", "species.csv", "species_code", ["species_code", "common_name"]
        ),
        "SEEN": _rel("SEEN", "Species", "species_code", "Species", "species_code"),
    }
    assert check(plan) == ([], [])


def test_a_join_on_a_column_that_is_not_a_property_is_left_to_the_structural_check(
    source,
):
    plan = _several_values_plan()
    plan["SEEN"]["from_node_column"] = "tally"  # not in Transect's properties
    assert check(plan) == ([], [])


def test_a_join_on_a_node_with_no_rule_is_left_to_the_structural_check(source):
    plan = _several_values_plan()
    del plan["Transect"]
    assert check(plan) == ([], [])


def test_a_node_whose_properties_are_unreadable_gets_no_verdict(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["Transect"]["properties"] = "species_code"
    assert check(plan) == ([], [])


# --- one line per node and property ------------------------------------------


def test_two_relationships_on_one_node_property_give_one_line(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["ALSO_SEEN"] = _rel(
        "ALSO_SEEN", "Transect", "species_code", "Species", "species_code"
    )
    problems, _ = check(plan)
    assert len(problems) == 1
    assert "'SEEN', 'ALSO_SEEN' join on" in problems[0]


def test_two_relationships_give_the_fix_text_once(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["ALSO_SEEN"] = _rel(
        "ALSO_SEEN", "Transect", "species_code", "Species", "species_code"
    )
    problems, _ = check(plan)
    assert problems[0].count("Fix it by") == 1


def test_a_self_join_names_its_relationship_once(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["NEAR"] = _rel("NEAR", "Transect", "species_code", "Transect", "species_code")
    del plan["SEEN"]
    problems, _ = check(plan)
    assert len(problems) == 1
    assert problems[0].count("'NEAR'") == 2  # once naming it, once in the fix


def test_a_relationship_multi_valued_on_both_ends_gives_two_lines_from_first(source):
    source("walks.csv", WALKS_SEVERAL)
    source("samples.csv", "sample_id,species_code\ns1,SP-1\ns1,SP-2\n")
    plan = {
        "Transect": _node("Transect", "walks.csv", "transect_name", ["species_code"]),
        "Sample": _node("Sample", "samples.csv", "sample_id", ["species_code"]),
        "PAIRS": _rel("PAIRS", "Transect", "species_code", "Sample", "species_code"),
    }
    problems, _ = check(plan)
    assert len(problems) == 2
    assert "'Transect'" in problems[0]
    assert "'Sample'" in problems[1]


# --- the node rule is found the way the structural check finds it ------------


def test_the_node_is_found_by_plan_key_even_when_its_label_differs(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["TransectRule"] = plan.pop("Transect")  # its "label" is still "Transect"
    plan["SEEN"]["from_node_label"] = "TransectRule"
    problems, _ = check(plan)
    assert len(problems) == 1


def test_a_relationship_naming_the_label_not_the_plan_key_finds_no_node(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["TransectRule"] = plan.pop("Transect")
    # SEEN still says "Transect": no plan key matches, and the structural check
    # reports that. This check stays silent rather than guessing.
    assert check(plan) == ([], [])


# --- missing cell versus blank cell ------------------------------------------


def test_a_row_with_no_cell_for_the_property_is_not_a_second_value(source):
    source("walks.csv", "transect_name,species_code\nnorth,SP-1\nnorth\nnorth,SP-1\n")
    assert check(_several_values_plan()) == ([], [])


def test_a_present_blank_cell_is_a_second_value(source):
    source("walks.csv", "transect_name,species_code\nnorth,SP-1\nnorth,\n")
    problems, _ = check(_several_values_plan())
    assert len(problems) == 1


# --- evidence: anything unverifiable is a note --------------------------------


def test_an_unreadable_source_is_a_note_not_a_refusal(source):
    problems, unverified = check(_several_values_plan())  # walks.csv never written
    assert problems == []
    assert len(unverified) == 1
    assert "walks.csv" in unverified[0]


@pytest.mark.parametrize("failure", [PermissionError, RuntimeError])
def test_a_failing_existence_check_is_a_note_not_a_raise(source, monkeypatch, failure):
    source("walks.csv", WALKS_SEVERAL)

    def failing(_path):
        raise failure("denied")

    monkeypatch.setattr("agentic_kg.tools.file_tools.source_exists", failing)
    problems, unverified = check(_several_values_plan())
    assert problems == []
    assert len(unverified) == 1
    assert "Error reading CSV file" in unverified[0]


def test_a_property_missing_from_its_source_is_a_note(source):
    source("walks.csv", "transect_name,tally\nnorth,3\n")
    problems, unverified = check(_several_values_plan())
    assert problems == []
    assert len(unverified) == 1


def test_one_unreadable_file_gives_one_note_naming_every_join_it_left_unchecked(
    source,
):
    plan = _several_values_plan()  # walks.csv never written
    plan["Transect"]["properties"] = ["species_code", "tally"]
    plan["COUNTED"] = _rel("COUNTED", "Transect", "tally", "Species", "species_code")
    _, unverified = check(plan)
    assert len(unverified) == 1
    assert "Transect.species_code" in unverified[0]
    assert "Transect.tally" in unverified[0]


def test_a_source_with_no_data_rows_gives_nothing(source):
    source("walks.csv", "transect_name,species_code,tally\n")
    assert check(_several_values_plan()) == ([], [])


def test_a_rule_without_a_usable_source_file_is_a_note(source):
    plan = _several_values_plan()
    plan["Transect"]["source_file"] = None
    problems, unverified = check(plan)
    assert problems == []
    assert len(unverified) == 1
    assert "source_file" in unverified[0]


def test_a_rule_without_a_usable_unique_column_is_a_note(source):
    plan = _several_values_plan()
    plan["Transect"]["unique_column_name"] = ["transect_name"]
    problems, unverified = check(plan)
    assert problems == []
    assert len(unverified) == 1
    assert "unique_column_name" in unverified[0]


# --- never raises ------------------------------------------------------------


@pytest.mark.parametrize(
    "rule_key,field,value",
    [
        ("SEEN", "from_node_label", ["Transect"]),
        ("SEEN", "from_node_label", {"a": 1}),
        ("SEEN", "from_node_label", None),
        ("SEEN", "from_node_column", ["species_code"]),
        ("SEEN", "from_node_column", 5),
        ("Transect", "unique_column_name", ["transect_name"]),
        ("Transect", "unique_column_name", {"a": 1}),
        ("Transect", "unique_column_name", None),
        ("Transect", "source_file", ["walks.csv"]),
        ("Transect", "source_file", None),
        ("Transect", "properties", 5),
        ("Transect", "properties", [["species_code"]]),
    ],
)
def test_a_malformed_rule_gives_no_refusal_and_never_raises(
    source, rule_key, field, value
):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan[rule_key][field] = value
    problems, unverified = check(plan)
    assert problems == []
    assert isinstance(unverified, list)


@pytest.mark.parametrize("field", ["to_node_label", "to_node_column"])
def test_a_malformed_endpoint_does_not_hide_the_other_endpoints_refusal(source, field):
    """Only the malformed endpoint is skipped: the from side still joins on a
    property with several values, so it is still refused, and nothing raises."""
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["SEEN"][field] = {"a": 1}
    problems, _ = check(plan)
    assert len(problems) == 1


def test_a_non_text_label_field_on_a_node_still_refuses_without_raising(source):
    source("walks.csv", WALKS_SEVERAL)
    plan = _several_values_plan()
    plan["Transect"]["label"] = ["Transect"]
    problems, _ = check(plan)
    assert len(problems) == 1


@pytest.mark.parametrize(
    "plan",
    [
        [],
        ["x"],
        "text",
        5,
        None,
        {"a": 1},
        {"a": ["x"]},
        {"Transect": "junk", "SEEN": _rel("SEEN", "Transect", "c", "T", "c")},
    ],
)
def test_a_plan_that_is_not_a_map_of_rules_gives_nothing(plan):
    assert check(plan) == ([], [])
