"""Unit tests for the proposed/approved construction plan state tools.

Regression coverage for a live session in which a targeted revision ("remove
INCLUDED_IN") caused the schema proposal agent to re-derive the whole plan and
overwrite an earlier, user-requested fix: the Assembly node reverted from being
keyed by `assembly_name` to being keyed by the per-row `assembly_id`, while the
ASSEMBLY_OF relationship kept joining on `assembly_name`. That plan was approved
and built, producing 64 wrong Assembly nodes and zero ASSEMBLY_OF edges.

Two layers are covered here:
  * the state layer behaves correctly in isolation (overwrite-by-key semantics),
    so the drift was not a state-management bug; and
  * approval now mechanically refuses the inconsistent plan the drift produced.
"""

import pytest

from agentic_kg.tools import construction_plan_tools as cpt
from agentic_kg.tools.construction_plan_tools import (
    APPROVED_CONSTRUCTION_PLAN,
    PROPOSED_CONSTRUCTION_PLAN,
    approve_proposed_construction_plan,
    check_construction_plan_consistency,
    get_proposed_construction_plan,
    get_proposed_construction_plan_with_approval_check,
    propose_node_construction,
    propose_node_constructions,
    propose_relationship_construction,
    propose_relationship_constructions,
    remove_node_construction,
)


class FakeToolContext:
    def __init__(self, state=None):
        self.state = state or {}


@pytest.fixture
def ctx():
    return FakeToolContext()


@pytest.fixture
def any_column_exists(monkeypatch):
    """Make the propose tools' search_file sanity check always pass."""

    def fake_search_file(file_path, pattern):
        return {"status": "success", "search_results": {"metadata": {"lines_found": 1}}}

    monkeypatch.setattr(cpt, "search_file", fake_search_file)


# --- state layer: overwrite semantics ---------------------------------------


def test_propose_node_twice_same_label_overwrites_unique_column(ctx, any_column_exists):
    propose_node_construction(
        "assemblies.csv",
        "Assembly",
        "assembly_id",
        ["component_name", "quantity", "product_id"],
        ctx,
    )
    propose_node_construction(
        "assemblies.csv",
        "Assembly",
        "assembly_name",
        ["assembly_id", "product_id"],
        ctx,
    )

    plan = get_proposed_construction_plan(ctx)
    assert list(plan) == ["Assembly"], "same label must replace, not accumulate"
    assert plan["Assembly"]["unique_column_name"] == "assembly_name"
    assert plan["Assembly"]["properties"] == ["assembly_id", "product_id"]


def test_propose_node_overwrite_is_total_not_a_merge(ctx, any_column_exists):
    """A re-propose drops fields the previous entry had; nothing is carried over.

    This is why a from-scratch re-derivation silently undoes a targeted fix.
    """
    propose_node_construction(
        "assemblies.csv",
        "Assembly",
        "assembly_name",
        ["assembly_id", "product_id"],
        ctx,
    )
    propose_node_construction(
        "assemblies.csv", "Assembly", "assembly_id", ["component_name"], ctx
    )

    plan = get_proposed_construction_plan(ctx)
    assert plan["Assembly"]["unique_column_name"] == "assembly_id"
    assert plan["Assembly"]["properties"] == ["component_name"]


def test_remove_then_repropose_leaves_only_the_new_entry(ctx, any_column_exists):
    propose_node_construction("assemblies.csv", "Assembly", "assembly_id", [], ctx)
    remove_node_construction("Assembly", ctx)
    assert get_proposed_construction_plan(ctx) == {}

    propose_node_construction("assemblies.csv", "Assembly", "assembly_name", [], ctx)
    plan = get_proposed_construction_plan(ctx)
    assert plan["Assembly"]["unique_column_name"] == "assembly_name"


def test_get_proposed_plan_reflects_state_not_a_snapshot(ctx, any_column_exists):
    propose_node_construction("suppliers.csv", "Supplier", "supplier_id", ["name"], ctx)
    first = get_proposed_construction_plan(ctx)
    propose_node_construction("products.csv", "Product", "product_id", [], ctx)
    assert set(get_proposed_construction_plan(ctx)) == {"Supplier", "Product"}
    assert first is ctx.state[PROPOSED_CONSTRUCTION_PLAN]


# --- batch proposal ---------------------------------------------------------


def _only_these_columns_exist(monkeypatch, known_columns, searched):
    """Stub the propose tools' sanity check so a chosen column is missing.

    Every lookup is appended to `searched`, which is how a test can tell an
    entry was never attempted from an entry that was attempted and rejected.
    """

    def fake_search_file(file_path, pattern):
        searched.append(pattern)
        found = 1 if pattern in known_columns else 0
        return {
            "status": "success",
            "search_results": {"metadata": {"lines_found": found}},
        }

    monkeypatch.setattr(cpt, "search_file", fake_search_file)


def test_propose_node_constructions_adds_every_entry_to_the_plan(
    ctx, any_column_exists
):
    result = propose_node_constructions(
        [
            {
                "approved_file": "products.csv",
                "proposed_label": "Product",
                "unique_column_name": "product_id",
                "proposed_properties": ["product_name"],
            },
            {
                "approved_file": "suppliers.csv",
                "proposed_label": "Supplier",
                "unique_column_name": "supplier_id",
                "proposed_properties": ["name"],
            },
        ],
        ctx,
    )

    assert result["status"] == "success"
    assert len(result[cpt.NODE_CONSTRUCTION]) == 2

    plan = ctx.state[PROPOSED_CONSTRUCTION_PLAN]
    assert set(plan) == {"Product", "Supplier"}
    assert plan["Product"]["unique_column_name"] == "product_id"
    assert plan["Product"]["properties"] == ["product_name"]
    assert plan["Supplier"]["source_file"] == "suppliers.csv"


def test_propose_node_constructions_stops_at_the_first_failing_entry(ctx, monkeypatch):
    """Earlier entries must survive the failure so the agent only has to correct
    the one entry the error names, instead of re-proposing the whole batch."""
    searched = []
    _only_these_columns_exist(
        monkeypatch, {"product_id", "supplier_id", "part_id"}, searched
    )

    result = propose_node_constructions(
        [
            {
                "approved_file": "products.csv",
                "proposed_label": "Product",
                "unique_column_name": "product_id",
                "proposed_properties": [],
            },
            {
                "approved_file": "suppliers.csv",
                "proposed_label": "Supplier",
                "unique_column_name": "supplier_id",
                "proposed_properties": [],
            },
            {
                "approved_file": "assemblies.csv",
                "proposed_label": "Assembly",
                "unique_column_name": "assembly_name",
                "proposed_properties": [],
            },
            {
                "approved_file": "parts.csv",
                "proposed_label": "Part",
                "unique_column_name": "part_id",
                "proposed_properties": [],
            },
        ],
        ctx,
    )

    assert result["status"] == "error"
    assert "2" in result["error_message"], "the error must name the entry's index"
    assert "Assembly" in result["error_message"]
    assert "assembly_name" in result["error_message"]

    plan = ctx.state[PROPOSED_CONSTRUCTION_PLAN]
    assert set(plan) == {"Product", "Supplier"}, (
        "entries before the failure stay in the plan"
    )
    assert "part_id" not in searched, "entries after the failure are never attempted"


def test_propose_relationship_constructions_adds_every_entry_to_the_plan(
    ctx, any_column_exists
):
    result = propose_relationship_constructions(
        [
            {
                "approved_file": "assemblies.csv",
                "proposed_relationship_type": "ASSEMBLY_OF",
                "from_node_label": "Assembly",
                "from_node_column": "assembly_name",
                "to_node_label": "Product",
                "to_node_column": "product_id",
                "proposed_properties": ["quantity"],
            },
            {
                "approved_file": "parts.csv",
                "proposed_relationship_type": "SUPPLIED_BY",
                "from_node_label": "Part",
                "from_node_column": "part_id",
                "to_node_label": "Supplier",
                "to_node_column": "supplier_id",
                "proposed_properties": [],
            },
        ],
        ctx,
    )

    assert result["status"] == "success"
    assert len(result[cpt.RELATIONSHIP_CONSTRUCTION]) == 2

    plan = ctx.state[PROPOSED_CONSTRUCTION_PLAN]
    assert set(plan) == {"ASSEMBLY_OF", "SUPPLIED_BY"}
    assert plan["ASSEMBLY_OF"]["from_node_column"] == "assembly_name"
    assert plan["ASSEMBLY_OF"]["to_node_column"] == "product_id"
    assert plan["ASSEMBLY_OF"]["properties"] == ["quantity"]


def test_propose_relationship_constructions_stops_at_the_first_failing_entry(
    ctx, monkeypatch
):
    searched = []
    _only_these_columns_exist(
        monkeypatch, {"assembly_name", "product_id", "part_id", "supplier_id"}, searched
    )

    result = propose_relationship_constructions(
        [
            {
                "approved_file": "assemblies.csv",
                "proposed_relationship_type": "ASSEMBLY_OF",
                "from_node_label": "Assembly",
                "from_node_column": "assembly_name",
                "to_node_label": "Product",
                "to_node_column": "product_id",
                "proposed_properties": [],
            },
            {
                "approved_file": "parts.csv",
                "proposed_relationship_type": "INCLUDED_IN",
                "from_node_label": "Part",
                "from_node_column": "part_id",
                "to_node_label": "Assembly",
                "to_node_column": "assembly_id",
                "proposed_properties": [],
            },
            {
                "approved_file": "parts.csv",
                "proposed_relationship_type": "SUPPLIED_BY",
                "from_node_label": "Part",
                "from_node_column": "part_id",
                "to_node_label": "Supplier",
                "to_node_column": "supplier_id",
                "proposed_properties": [],
            },
        ],
        ctx,
    )

    assert result["status"] == "error"
    assert "1" in result["error_message"], "the error must name the entry's index"
    assert "INCLUDED_IN" in result["error_message"]
    assert "assembly_id" in result["error_message"]

    plan = ctx.state[PROPOSED_CONSTRUCTION_PLAN]
    assert set(plan) == {"ASSEMBLY_OF"}, "entries before the failure stay in the plan"
    assert "supplier_id" not in searched, (
        "entries after the failure are never attempted"
    )


# --- consistency check ------------------------------------------------------


def _drifted_plan():
    """The exact shape that was approved and built in the failing session."""
    return {
        "Product": {
            "construction_type": "node",
            "source_file": "products.csv",
            "label": "Product",
            "unique_column_name": "product_id",
            "properties": ["product_name"],
        },
        "Assembly": {
            "construction_type": "node",
            "source_file": "assemblies.csv",
            "label": "Assembly",
            "unique_column_name": "assembly_id",
            "properties": ["component_name", "quantity", "product_id"],
        },
        "ASSEMBLY_OF": {
            "construction_type": "relationship",
            "source_file": "assemblies.csv",
            "relationship_type": "ASSEMBLY_OF",
            "from_node_label": "Assembly",
            "from_node_column": "assembly_name",
            "to_node_label": "Product",
            "to_node_column": "product_id",
            "properties": ["quantity"],
        },
    }


def _consistent_plan():
    plan = _drifted_plan()
    plan["Assembly"]["unique_column_name"] = "assembly_name"
    plan["Assembly"]["properties"] = ["assembly_id", "product_id"]
    return plan


def test_consistency_check_flags_join_column_the_node_does_not_carry():
    problems = check_construction_plan_consistency(_drifted_plan())
    assert len(problems) == 1
    assert "ASSEMBLY_OF" in problems[0]
    assert "assembly_name" in problems[0]


def test_consistency_check_accepts_the_plan_the_user_asked_for():
    assert check_construction_plan_consistency(_consistent_plan()) == []


def test_consistency_check_allows_join_on_a_declared_property():
    plan = _consistent_plan()
    # product_id is a property of Assembly, not its key, but it does exist on the node
    plan["ASSEMBLY_OF"]["from_node_column"] = "assembly_name"
    plan["ASSEMBLY_OF"]["to_node_column"] = "product_id"
    assert check_construction_plan_consistency(plan) == []


def test_consistency_check_flags_relationship_to_an_undefined_label():
    plan = _consistent_plan()
    del plan["Assembly"]
    problems = check_construction_plan_consistency(plan)
    assert any("no node construction" in p for p in problems)


def test_consistency_check_ignores_non_dict_and_empty_plans():
    assert check_construction_plan_consistency({}) == []
    assert check_construction_plan_consistency([]) == []


# --- approval gate ----------------------------------------------------------


def test_approve_refuses_the_drifted_plan_and_does_not_record_it(ctx):
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _drifted_plan()
    result = approve_proposed_construction_plan(ctx)

    assert result["status"] == "error"
    assert "ASSEMBLY_OF" in result["error_message"]
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state


def test_approve_records_a_consistent_plan(ctx):
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _consistent_plan()
    result = approve_proposed_construction_plan(ctx)

    assert result["status"] == "success"
    assert ctx.state[APPROVED_CONSTRUCTION_PLAN] == _consistent_plan()


def test_approve_refuses_a_plan_whose_relationship_endpoint_has_no_node(
    ctx, any_column_exists
):
    """A dangling endpoint is refused, not just detected.

    A live session showed a revision presenting a node list that omitted a label a
    relationship still pointed at. Removing a node construction without removing the
    relationships that reference it produces exactly that plan; it must never be
    approvable, since the relationship would build zero edges.
    """
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _consistent_plan()
    remove_node_construction("Assembly", ctx)

    result = approve_proposed_construction_plan(ctx)

    assert result["status"] == "error"
    assert "Assembly" in result["error_message"]
    assert "no node construction" in result["error_message"]
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state


def test_approve_refuses_when_there_is_no_proposed_plan(ctx):
    result = approve_proposed_construction_plan(ctx)
    assert result["status"] == "error"
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state


# --- approval check (dry run) -----------------------------------------------


def test_approval_check_refuses_when_there_is_no_proposed_plan(ctx):
    """Would catch an implementation that returns success on an empty plan --
    the coordinator would then present nothing at all as approvable."""
    result = get_proposed_construction_plan_with_approval_check(ctx)

    assert result["status"] == "error"
    assert "no proposed construction plan" in result["error_message"].lower()


def test_approval_check_reports_the_same_problems_approval_would(ctx):
    """Would catch a check that drifted away from the one approval actually
    runs -- the coordinator would tell the user a plan is fine and then watch
    approve_proposed_construction_plan refuse it."""
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _drifted_plan()

    result = get_proposed_construction_plan_with_approval_check(ctx)

    assert result["status"] == "error"
    for problem in check_construction_plan_consistency(_drifted_plan()):
        assert problem in result["error_message"]


def test_approval_check_returns_the_plan_and_a_directive_when_nothing_blocks(ctx):
    """Would catch a success payload that omits the plan (leaving the
    coordinator to describe it from memory, which its instruction forbids) or
    omits the message (leaving the verdict as an absence again)."""
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _consistent_plan()

    result = get_proposed_construction_plan_with_approval_check(ctx)

    assert result["status"] == "success"
    payload = result["result"]
    assert payload["proposed_construction_plan"] == _consistent_plan()
    assert payload["message"]


def test_approval_check_never_records_an_approval(ctx):
    """Would catch the tool carrying approve_proposed_construction_plan's state
    write across when its body was copied -- every plan the coordinator merely
    looked at would become approved, silently."""
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _consistent_plan()

    get_proposed_construction_plan_with_approval_check(ctx)

    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state


def test_approval_check_blocked_message_prescribes_no_recovery_action(ctx):
    """Would catch the natural-looking addition of "run schema_refinement_loop
    with these as feedback" to the blocked message. That reads as helpful and is
    correct mid-turn -- but on the second 'retry' and on 'stopped:' the
    coordinator's instruction says to stop calling the loop, so the tool result
    would contradict the instruction in the same turn, at exactly the two
    branches this fix exists to clean up. The tool knows only whether approval
    would succeed; only the instruction knows which branch it is in."""
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _drifted_plan()

    message = get_proposed_construction_plan_with_approval_check(ctx)["error_message"]

    assert "schema_refinement_loop" not in message


def test_approval_check_message_names_the_tool_and_forbids_the_unready_claim(ctx):
    """Would catch a message softened into a neutral status line. The reported
    bug is the agent asserting 'not ready for approval' when nothing blocks
    approval; this message is the only place that claim is contradicted."""
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _consistent_plan()

    message = get_proposed_construction_plan_with_approval_check(ctx)["result"][
        "message"
    ]

    assert "approve_proposed_construction_plan" in message
    assert "not ready for approval" in message


# Required values must be present


def test_node_construction_without_a_label_is_refused(ctx, any_column_exists):
    """An absent label used to be stored as a plan entry keyed None with
    "label": None, which check_construction_plan_consistency accepts and which
    only surfaces much later at import time."""
    result = propose_node_constructions(
        [{"approved_file": "products.csv", "unique_column_name": "product_id"}], ctx
    )
    assert result["status"] == "error"
    assert "proposed_label" in result["error_message"]
    assert ctx.state.get(PROPOSED_CONSTRUCTION_PLAN, {}) == {}


def test_node_construction_names_every_missing_value(ctx, any_column_exists):
    result = propose_node_constructions([{"proposed_properties": []}], ctx)
    assert result["status"] == "error"
    for field in ("approved_file", "proposed_label", "unique_column_name"):
        assert field in result["error_message"]


def test_relationship_construction_without_endpoints_is_refused(ctx, any_column_exists):
    result = propose_relationship_constructions(
        [
            {
                "approved_file": "products.csv",
                "proposed_relationship_type": "HAS_PART",
                "from_node_label": "Product",
            }
        ],
        ctx,
    )
    assert result["status"] == "error"
    for field in ("from_node_column", "to_node_label", "to_node_column"):
        assert field in result["error_message"]
    assert ctx.state.get(PROPOSED_CONSTRUCTION_PLAN, {}) == {}


def test_null_properties_are_stored_as_an_empty_list(ctx, any_column_exists):
    """A model may send JSON null rather than omitting the field. That reaches
    Cypher as FOREACH (k IN null | ...), a silent no-op that loads the nodes
    with no properties at all and reports success."""
    propose_node_construction("products.csv", "Product", "product_id", None, ctx)
    assert ctx.state[PROPOSED_CONSTRUCTION_PLAN]["Product"]["properties"] == []

    propose_relationship_construction(
        "products.csv",
        "HAS_PART",
        "Product",
        "product_id",
        "Part",
        "part_id",
        None,
        ctx,
    )
    assert ctx.state[PROPOSED_CONSTRUCTION_PLAN]["HAS_PART"]["properties"] == []


# --- property types ---------------------------------------------------------


def test_node_construction_stores_property_types(ctx, any_column_exists):
    """Without this the model has nowhere to record a type and every property
    reaches the loader as a string -- the defect itself."""
    propose_node_construction(
        "part_supplier_mapping.csv",
        "Part",
        "part_id",
        ["unit_cost", "lead_time_days", "part_name"],
        ctx,
        {"unit_cost": "float", "lead_time_days": "integer"},
    )

    rule = ctx.state[PROPOSED_CONSTRUCTION_PLAN]["Part"]
    assert rule["property_types"] == {"unit_cost": "float", "lead_time_days": "integer"}
    assert rule["properties"] == ["unit_cost", "lead_time_days", "part_name"]


def test_relationship_construction_stores_property_types(ctx, any_column_exists):
    propose_relationship_construction(
        "part_supplier_mapping.csv",
        "SUPPLIED_BY",
        "Part",
        "part_id",
        "Supplier",
        "supplier_id",
        ["unit_cost"],
        ctx,
        {"unit_cost": "float"},
    )

    rule = ctx.state[PROPOSED_CONSTRUCTION_PLAN]["SUPPLIED_BY"]
    assert rule["property_types"] == {"unit_cost": "float"}


def test_batch_node_constructions_carry_property_types(ctx, any_column_exists):
    propose_node_constructions(
        [
            {
                "approved_file": "products.csv",
                "proposed_label": "Product",
                "unique_column_name": "product_id",
                "proposed_properties": ["price"],
                "proposed_property_types": {"price": "float"},
            },
        ],
        ctx,
    )

    assert ctx.state[PROPOSED_CONSTRUCTION_PLAN]["Product"]["property_types"] == {
        "price": "float"
    }


def test_batch_relationship_constructions_carry_property_types(ctx, any_column_exists):
    propose_relationship_constructions(
        [
            {
                "approved_file": "part_supplier_mapping.csv",
                "proposed_relationship_type": "SUPPLIED_BY",
                "from_node_label": "Part",
                "from_node_column": "part_id",
                "to_node_label": "Supplier",
                "to_node_column": "supplier_id",
                "proposed_properties": ["lead_time_days"],
                "proposed_property_types": {"lead_time_days": "integer"},
            },
        ],
        ctx,
    )

    assert ctx.state[PROPOSED_CONSTRUCTION_PLAN]["SUPPLIED_BY"]["property_types"] == {
        "lead_time_days": "integer"
    }


def test_null_property_types_are_stored_as_an_empty_dict(ctx, any_column_exists):
    """A model may send JSON null rather than omitting the field. Stored raw,
    that null would reach the loader as a plan key that reads as 'typed' and
    blow up on .items(); the plan must degrade to plain text properties instead."""
    propose_node_construction(
        "products.csv", "Product", "product_id", ["price"], ctx, None
    )
    propose_relationship_construction(
        "part_supplier_mapping.csv",
        "SUPPLIED_BY",
        "Part",
        "part_id",
        "Supplier",
        "supplier_id",
        ["unit_cost"],
        ctx,
        None,
    )

    assert ctx.state[PROPOSED_CONSTRUCTION_PLAN]["Product"]["property_types"] == {}
    assert ctx.state[PROPOSED_CONSTRUCTION_PLAN]["SUPPLIED_BY"]["property_types"] == {}


# --- type consistency -------------------------------------------------------


def _typed_plan(**overrides):
    """A minimal two-node, one-relationship plan; overrides patch one rule."""
    plan = {
        "Part": {
            "construction_type": "node",
            "source_file": "parts.csv",
            "label": "Part",
            "unique_column_name": "part_id",
            "properties": ["unit_cost", "part_name"],
            "property_types": {"unit_cost": "float"},
        },
        "Supplier": {
            "construction_type": "node",
            "source_file": "suppliers.csv",
            "label": "Supplier",
            "unique_column_name": "supplier_id",
            "properties": ["name"],
            "property_types": {},
        },
        "SUPPLIED_BY": {
            "construction_type": "relationship",
            "source_file": "part_supplier_mapping.csv",
            "relationship_type": "SUPPLIED_BY",
            "from_node_label": "Part",
            "from_node_column": "part_id",
            "to_node_label": "Supplier",
            "to_node_column": "supplier_id",
            "properties": ["lead_time_days"],
            "property_types": {"lead_time_days": "integer"},
        },
    }
    for key, rule in overrides.items():
        plan[key] = rule
    return plan


def test_a_legal_typed_plan_is_consistent():
    assert check_construction_plan_consistency(_typed_plan()) == []


def test_a_type_for_a_property_not_in_the_list_is_refused():
    """A type on a name the loader never reads is a silent no-op: the plan looks
    typed and the graph comes out as strings."""
    plan = _typed_plan()
    plan["Part"]["property_types"] = {"unti_cost": "float"}

    problems = check_construction_plan_consistency(plan)
    assert any("unti_cost" in problem for problem in problems)


def test_typing_the_unique_column_is_refused_even_when_also_a_property():
    """Checked independently of properties-membership: nothing stops a model
    listing the key column as a property too, and the first rule alone would
    then let the identifier be typed."""
    plan = _typed_plan()
    plan["Part"]["properties"] = ["part_id", "part_name"]
    plan["Part"]["property_types"] = {"part_id": "integer"}

    problems = check_construction_plan_consistency(plan)
    assert any("part_id" in problem for problem in problems)


def test_typing_a_column_a_relationship_joins_on_is_refused_and_names_both_exits():
    """import_relationships' MATCH compares the raw CSV string against the stored
    property, and Neo4j does not coerce '5' = 5 -- a typed node side matches
    nothing, silently. The message must name both fixes because the refinement
    loop gets one invocation per turn."""
    plan = _typed_plan()
    plan["Part"]["properties"] = ["part_id", "unit_cost", "part_name"]
    plan["Part"]["property_types"] = {"part_id": "integer"}
    plan["SUPPLIED_BY"]["from_node_column"] = "part_id"

    problems = check_construction_plan_consistency(plan)
    joined = " ".join(problems)
    assert "part_id" in joined
    assert "SUPPLIED_BY" in joined


def test_a_relationship_typing_its_own_join_column_is_refused():
    """Catches the gap where joined_columns is keyed by (node_label, column) but
    looked up as (key, name): for a relationship rule, key is the relationship
    type, so a lookup against a map keyed by node labels never matches. Without
    this check, a relationship that lists its own from_node_column in
    properties and declares a type for it was accepted with zero problems --
    the value gets coerced at build time, the MATCH then compares a number
    against the raw string the node loader stored, and the rule silently
    produces zero relationships."""
    plan = _typed_plan()
    plan["SUPPLIED_BY"]["properties"] = ["lead_time_days", "part_id"]
    plan["SUPPLIED_BY"]["property_types"] = {
        "lead_time_days": "integer",
        "part_id": "integer",
    }

    problems = check_construction_plan_consistency(plan)
    joined = " ".join(problems)
    assert "part_id" in joined
    assert "SUPPLIED_BY" in joined


def test_a_relationship_typing_its_own_join_column_with_an_unresolved_endpoint_label_names_no_target():
    """Catches the implementation that computed
    join_target = (nodes.get(own_node_label) or {}).get("unique_column_name")
    unconditionally: when own_node_label has no matching node construction in
    the plan, that expression is None, and the refusal used to render the
    literal string \"join SUPPLIED_BY on 'None' instead\" -- an unactionable
    second exit that reads as though 'None' were a real column name, breaking
    the two-exits contract the refinement loop depends on for a single-pass
    fix."""
    plan = _typed_plan()
    del plan["Supplier"]
    plan["SUPPLIED_BY"]["properties"] = ["lead_time_days", "supplier_id"]
    plan["SUPPLIED_BY"]["property_types"] = {
        "lead_time_days": "integer",
        "supplier_id": "integer",
    }

    problems = check_construction_plan_consistency(plan)
    joined = " ".join(problems)
    assert "supplier_id" in joined
    assert "'None'" not in joined


def test_a_typed_join_column_is_refused_when_the_rule_key_differs_from_the_label():
    """Catches the gap where joined_columns is keyed by (node_label, column) but
    looked up as (key, name), where key is the plan-dict key of the rule under
    inspection. That lookup only works by coincidence when key == rule['label'],
    which propose_node_construction happens to arrange but nothing in the
    checker guarantees -- this branch's own TYPED_PLAN integration fixture uses
    key 'Part' with label 'TypedPart', so the rule silently did not apply to
    that plan at all."""
    plan = _typed_plan()
    plan["PartKey"] = plan.pop("Part")
    plan["SUPPLIED_BY"]["from_node_column"] = "unit_cost"

    problems = check_construction_plan_consistency(plan)
    joined = " ".join(problems)
    assert "unit_cost" in joined
    assert "SUPPLIED_BY" in joined


def test_an_unknown_type_name_is_refused():
    """'string', 'date' and 'int' are all plausible model output and none of them
    is in the closed set; coerce() would fail every value of such a column."""
    plan = _typed_plan()
    plan["Part"]["property_types"] = {"unit_cost": "decimal"}

    problems = check_construction_plan_consistency(plan)
    assert any("decimal" in problem for problem in problems)


def test_approval_refuses_a_plan_with_an_illegal_type(ctx):
    """The rules are only worth anything if approval enforces them."""
    plan = _typed_plan()
    plan["Part"]["property_types"] = {"unit_cost": "decimal"}
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = plan

    result = approve_proposed_construction_plan(ctx)
    assert result["status"] == "error"
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state
