"""KG-57: proposing a node or relationship checks its properties against the header.

The propose tools checked the key and join columns against the file's header
(KG-50) but never the properties, so a misspelt property ('canpy' for 'canopy')
was accepted, approved and built, and the graph silently lacked it. The
properties now go into the same exact-header check: one read of the header line,
one refusal naming every missing name, key or join columns first.

These tests read real files from an in-memory source and never fake the header
reader, as tests/unit/test_propose_header_check.py does and for its reason: a
fake could pass without reaching the code under test.

Neutral vocabulary (a field survey), except the last section, which proposes the
bundled BOM example's relationships from the real data/bom files.
"""

from pathlib import Path

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.csv_reader import read_csv_header
from agentic_kg.tools.construction_plan_tools import (
    PROPOSED_CONSTRUCTION_PLAN,
    find_plan_problems,
    propose_node_construction,
    propose_node_constructions,
    propose_relationship_construction,
    propose_relationship_constructions,
)

HEADER = ["plotID", "siteID", "canopy"]
BOM = Path(__file__).resolve().parents[2] / "data" / "bom"


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
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write("plotID,siteID,canopy\nPL-1,ridge,open\nPL-2,hollow,closed\n")
    # plotID repeats and the rows disagree on observer: the key-values check
    # (KG-48) refuses a node keyed on plotID that lists observer.
    with fs.open("/src/visits.csv", "w") as handle:
        handle.write("plotID,observer\nPL-1,ann\nPL-1,bob\n")
    # A header name holding a comma and a space, quoted as CSV requires.
    with fs.open("/src/cover.csv", "w") as handle:
        handle.write('plotID,"canopy, cover"\nPL-1,open\n')
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _plan(ctx):
    return ctx.state.get(PROPOSED_CONSTRUCTION_PLAN, {})


def _node(ctx, properties, key="plotID", approved_file="plots.csv", label="Plot"):
    return propose_node_construction(approved_file, label, key, properties, ctx)


def _rel(ctx, properties, from_column="plotID", to_column="siteID", **matched):
    return propose_relationship_construction(
        "plots.csv",
        "ON_SITE",
        "Plot",
        from_column,
        "Site",
        to_column,
        properties,
        ctx,
        **matched,
    )


def _not_in(column, header=HEADER, file="plots.csv"):
    return f"Column '{column}' is not in {file}. Available columns: {header}"


def _several_not_in(columns, header=HEADER, file="plots.csv"):
    return f"Column(s) {columns} are not in {file}. Available columns: {header}"


# --- a property the file lacks is refused -------------------------------------


@pytest.mark.parametrize(
    "prop",
    ["canpy", "Canopy", "canopy ", " canopy", ""],
    ids=["typo", "other-case", "trailing-space", "leading-space", "empty"],
)
def test_a_node_property_that_is_not_exactly_a_header_is_refused(
    survey_source, ctx, prop
):
    result = _node(ctx, ["siteID", prop])
    assert result["status"] == "error"
    assert result["error_message"] == _several_not_in([prop])
    assert "Plot" not in _plan(ctx)


def test_a_relationship_property_the_file_lacks_is_refused(survey_source, ctx):
    result = _rel(ctx, ["canpy"])
    assert result["status"] == "error"
    assert result["error_message"] == _several_not_in(["canpy"])
    assert "ON_SITE" not in _plan(ctx)


# --- one refusal names every missing name, key or join columns first ----------


def test_a_bad_key_and_a_bad_property_are_named_in_one_refusal(survey_source, ctx):
    result = _node(ctx, ["canpy"], key="plotid")
    assert result["error_message"] == _several_not_in(["plotid", "canpy"])


def test_a_bad_join_column_and_a_bad_property_are_named_in_one_refusal(
    survey_source, ctx
):
    result = _rel(ctx, ["canpy"], from_column="plotid")
    assert result["error_message"] == _several_not_in(["plotid", "canpy"])


def test_two_bad_properties_are_named_in_one_refusal_in_the_order_given(
    survey_source, ctx
):
    result = _node(ctx, ["sitID", "canopy", "canpy"])
    assert result["error_message"] == _several_not_in(["sitID", "canpy"])


# --- order against the other checks, and the plan already held ----------------


def test_a_bad_property_is_refused_before_the_key_values_are_read(survey_source, ctx):
    """visits.csv's key repeats with disagreeing rows, which the key-values check
    refuses; the misspelt property must be reported first, from the header alone."""
    result = _node(
        ctx, ["observer", "obsrver"], approved_file="visits.csv", label="Visit"
    )
    assert result["error_message"] == _several_not_in(
        ["obsrver"], header=["plotID", "observer"], file="visits.csv"
    )


def test_a_refused_re_proposal_leaves_the_earlier_rule_unchanged(survey_source, ctx):
    assert _node(ctx, ["canopy"])["status"] == "success"
    result = _node(ctx, ["canpy"])
    assert result["status"] == "error"
    assert _plan(ctx)["Plot"]["properties"] == ["canopy"]


# --- the batch tools ----------------------------------------------------------


def test_a_node_batch_refuses_a_bad_property_and_keeps_the_earlier_entry(
    survey_source, ctx
):
    result = propose_node_constructions(
        [
            {
                "approved_file": "plots.csv",
                "proposed_label": "Plot",
                "unique_column_name": "plotID",
                "proposed_properties": ["canopy"],
            },
            {
                "approved_file": "plots.csv",
                "proposed_label": "Site",
                "unique_column_name": "siteID",
                "proposed_properties": ["canpy"],
            },
        ],
        ctx,
    )
    assert result["status"] == "error"
    assert result["error_message"] == (
        "node construction 1 (Site) failed: " + _several_not_in(["canpy"])
    )
    assert set(_plan(ctx)) == {"Plot"}


def test_a_relationship_batch_refuses_a_bad_property_and_keeps_the_earlier_entry(
    survey_source, ctx
):
    entry = {
        "approved_file": "plots.csv",
        "from_node_label": "Plot",
        "from_node_column": "plotID",
        "to_node_label": "Site",
        "to_node_column": "siteID",
    }
    result = propose_relationship_constructions(
        [
            {
                **entry,
                "proposed_relationship_type": "ON_SITE",
                "proposed_properties": ["canopy"],
            },
            {
                **entry,
                "proposed_relationship_type": "NEAR_SITE",
                "proposed_properties": ["canpy"],
            },
        ],
        ctx,
    )
    assert result["status"] == "error"
    assert result["error_message"] == (
        "relationship construction 1 (NEAR_SITE) failed: " + _several_not_in(["canpy"])
    )
    assert set(_plan(ctx)) == {"ON_SITE"}


# --- what stays accepted ------------------------------------------------------


@pytest.mark.parametrize(
    "properties",
    [[], ["siteID", "canopy"], ["plotID"], ["canopy", "canopy"]],
    ids=["none", "real-columns", "the-key-itself", "listed-twice"],
)
def test_a_node_whose_properties_are_all_columns_is_accepted(
    survey_source, ctx, properties
):
    result = _node(ctx, properties)
    assert result["status"] == "success", result.get("error_message")
    assert _plan(ctx)["Plot"]["properties"] == properties


@pytest.mark.parametrize(
    "properties",
    [["canopy"], ["siteID"]],
    ids=["real-column", "a-join-column"],
)
def test_a_relationship_whose_properties_are_all_columns_is_accepted(
    survey_source, ctx, properties
):
    result = _rel(ctx, properties)
    assert result["status"] == "success", result.get("error_message")


def test_a_header_name_holding_a_comma_is_a_property_like_any_other(survey_source, ctx):
    result = _node(ctx, ["canopy, cover"], approved_file="cover.csv")
    assert result["status"] == "success", result.get("error_message")


def test_matched_node_properties_are_not_checked_against_the_file(survey_source, ctx):
    """from_node_property / to_node_property name node properties, not columns."""
    result = _rel(ctx, ["canopy"], to_node_property="siteCode")
    assert result["status"] == "success", result.get("error_message")


# --- the one-column wording is kept where only one name is asked about --------


@pytest.mark.parametrize(
    "properties", [[], ["plotid"]], ids=["no-properties", "the-bad-key-itself"]
)
def test_a_bad_key_alone_keeps_the_one_column_wording(survey_source, ctx, properties):
    assert _node(ctx, properties, key="plotid")["error_message"] == _not_in("plotid")


# --- an unreadable property list is left to approval -------------------------


@pytest.mark.parametrize("propose", [_node, _rel], ids=["node", "relationship"])
@pytest.mark.parametrize(
    "properties, shown",
    [("canopy", "'canopy'"), (["canopy", None], "['canopy', None]")],
    ids=["one-string", "a-non-text-entry"],
)
def test_an_unreadable_property_list_is_not_checked_name_by_name(
    survey_source, ctx, properties, shown, propose
):
    """Read as names, 'canopy' would be checked letter by letter. The proposal goes
    through, and approval refuses the list with its own message. Each tool has its
    own guard, so both are proposed."""
    result = propose(ctx, properties)
    assert result["status"] == "success", result.get("error_message")
    problems, _ = find_plan_problems(ctx.state)
    assert any(
        f"'properties' must be a list of property names, got {shown}" in problem
        for problem in problems
    ), problems


# --- the bundled BOM example --------------------------------------------------


@pytest.fixture
def bom_source(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", str(BOM))
    reset_settings()
    yield
    reset_settings()


# (file, type, from label, from column, to label, to column, one pinned real property)
BOM_RELATIONSHIPS = [
    (
        "assemblies.csv",
        "IS_PART_OF",
        "Assembly",
        "assembly_id",
        "Product",
        "product_id",
        "quantity",
    ),
    (
        "components.csv",
        "PART_OF",
        "Part",
        "part_id",
        "Assembly",
        "assembly_id",
        "quantity",
    ),
    (
        "part_supplier_mapping.csv",
        "SUPPLIED_BY",
        "Part",
        "part_id",
        "Supplier",
        "supplier_id",
        "lead_time_days",
    ),
]
BOM_IDS = [row[0] for row in BOM_RELATIONSHIPS]


def _propose_bom(ctx, row, properties):
    source_file, rel_type, from_label, from_column, to_label, to_column, _ = row
    return propose_relationship_construction(
        source_file,
        rel_type,
        from_label,
        from_column,
        to_label,
        to_column,
        properties,
        ctx,
    )


@pytest.mark.parametrize("row", BOM_RELATIONSHIPS, ids=BOM_IDS)
def test_a_bundled_relationship_with_every_other_column_is_accepted(
    bom_source, ctx, row
):
    """Guards against over-refusal. Every header name but the join columns, read
    from the file itself, plus one pinned real name, so a header read that came
    back empty or wrong cannot make this pass vacuously."""
    source_file, _, _, from_column, _, to_column, pinned = row
    properties = [
        name
        for name in read_csv_header(source_file)
        if name not in (from_column, to_column)
    ]
    assert pinned in properties
    result = _propose_bom(ctx, row, properties)
    assert result["status"] == "success", result.get("error_message")


@pytest.mark.parametrize("row", BOM_RELATIONSHIPS, ids=BOM_IDS)
def test_a_bundled_relationship_with_a_misspelt_property_is_refused(
    bom_source, ctx, row
):
    """The half that proves the fix: the pinned name with one letter dropped."""
    source_file, rel_type, *_, pinned = row
    misspelt = pinned[:-2] + pinned[-1]
    result = _propose_bom(ctx, row, [pinned, misspelt])
    assert result["status"] == "error"
    assert (
        f"Column(s) ['{misspelt}'] are not in {source_file}." in result["error_message"]
    )
    assert rel_type not in _plan(ctx)
