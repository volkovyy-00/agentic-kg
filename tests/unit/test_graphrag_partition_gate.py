"""Unit tests for the retrieval agent's partition-interpretation disclosure gate.

The gate exists because graphrag_agent_v2's instruction told the model to
silently judge whether a partitioned_by numeric property is a quantity or a
set of kinds, and to say which -- nothing enforced the "say" half (KG-5).
These tests cover the mechanism only: that a real declaration ends up in the
tool-call record. Whether the model's final answer also restates it is not
unit-testable and is verified by hand (see the plan's manual verification
task).
"""

import inspect

from agentic_kg.common.tool_result import is_error, is_success
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent import (
    variants as variants_module,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.agent import (
    graphrag_agent,
    reset_graphrag_handoff_confirmation,
    reset_partition_interpretation_declaration,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.variants import (
    read_neo4j_cypher,
    variants,
)
from agentic_kg.tools.graphrag_partition_tools import (
    NONE_APPLY_SENTINEL,
    PARTITION_INTERPRETATION_DECLARED_KEY,
    declare_partition_interpretation,
)


class FakeToolContext:
    """A tool context carrying .state only -- neither new tool touches .actions."""

    def __init__(self, state=None):
        self.state = dict(state or {})


def _profile(patterns):
    return {"schema": {}, "profile": {"patterns": patterns}}


def _numbers_pattern(property_name="quantity"):
    return {
        "pattern": "Part-[PART_OF]->Assembly",
        "start": "Part",
        "type": "PART_OF",
        "end": "Assembly",
        "partitioned_by": [
            {
                "property": property_name,
                "distribution": [{"value": 1, "count": 5}, {"value": 2, "count": 3}],
                "values_are": "numbers",
                "distribution_covers": "this_pattern",
            }
        ],
    }


def _categories_pattern():
    return {
        "pattern": "Part-[PART_OF]->Assembly",
        "start": "Part",
        "type": "PART_OF",
        "end": "Assembly",
        "partitioned_by": [
            {
                "property": "tier",
                "distribution": [{"value": "a", "count": 5}],
                "values_are": "categories",
                "distribution_covers": "this_pattern",
            }
        ],
    }


def test_declare_with_a_flagged_property_records_the_flag(monkeypatch):
    """Catches a declaration that reports success without writing state,
    which would leave the gated read tool refusing every declared turn."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    context = FakeToolContext()
    result = declare_partition_interpretation(
        "quantity", "treating as a total", context
    )
    assert is_success(result)
    assert context.state[PARTITION_INTERPRETATION_DECLARED_KEY] is True


def test_declare_with_the_none_sentinel_also_records_the_flag(monkeypatch):
    """Catches the 'none apply' escape hatch being rejected -- it would turn
    every false-positive trigger into a dead end instead of a one-line
    reply, since the coarse gate fires whether or not a query actually
    touches the flagged property."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    context = FakeToolContext()
    result = declare_partition_interpretation(
        NONE_APPLY_SENTINEL, "not touching it", context
    )
    assert is_success(result)
    assert context.state[PARTITION_INTERPRETATION_DECLARED_KEY] is True


def test_declare_with_an_unflagged_property_is_refused(monkeypatch):
    """Catches a rubber-stamped declaration: a property the profile does not
    currently flag would open the gate without the model ever having looked
    at what is actually flagged, reopening the disclosure gap this exists
    to close."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    context = FakeToolContext()
    result = declare_partition_interpretation(
        "not_a_real_property", "guessing", context
    )
    assert is_error(result)
    assert PARTITION_INTERPRETATION_DECLARED_KEY not in context.state


def test_declare_refusal_names_the_properties_that_are_flagged(monkeypatch):
    """Catches a generic refusal message that leaves the model guessing what
    a valid argument would have been."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    result = declare_partition_interpretation("bogus", "guessing", FakeToolContext())
    assert "quantity" in result["error_message"]


def _fake_impl_recording_calls(calls):
    def impl(query, params=None):
        calls.append(query)
        return {"status": "success", "query_result": {"records": [], "row_count": 0}}

    return impl


def test_gated_read_runs_the_query_when_no_profile_is_cached(monkeypatch):
    """Catches the gate forcing a cold profile build, or refusing outright,
    when nothing has been profiled yet this session -- fail-open is the
    whole point of using peek_cached_profile over get_cached_profile here."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: None,
    )
    calls = []
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls(calls)
    )
    result = gated("MATCH (n) RETURN sum(n.quantity)", FakeToolContext())
    assert is_success(result)
    assert calls == ["MATCH (n) RETURN sum(n.quantity)"]


def test_gated_read_runs_the_query_when_nothing_is_flagged(monkeypatch):
    """Catches the gate firing on a graph where nothing is partitioned by a
    numeric property -- it must not add friction where this ticket's problem
    cannot occur."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: _profile([_categories_pattern()]),
    )
    calls = []
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls(calls)
    )
    result = gated("MATCH (n) RETURN sum(n.quantity)", FakeToolContext())
    assert is_success(result)
    assert len(calls) == 1


def test_gated_read_runs_a_non_aggregating_query_unconditionally(monkeypatch):
    """Catches the gate firing on a query that names a flagged property but
    never aggregates it -- e.g. a plain lookup -- which would add friction
    beyond what AC1 ('every answer aggregating...') asks for."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    calls = []
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls(calls)
    )
    result = gated("MATCH (n) RETURN n.quantity", FakeToolContext())
    assert is_success(result)
    assert len(calls) == 1


def test_gated_read_refuses_an_undeclared_aggregation_over_a_flagged_graph(
    monkeypatch,
):
    """Catches the gate being absent: an aggregating query, on a graph with a
    flagged numeric property, must be refused before a declaration is made --
    this is the core behaviour KG-5 asks for."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    calls = []
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls(calls)
    )
    result = gated("MATCH (n) RETURN sum(n.quantity)", FakeToolContext())
    assert is_error(result)
    assert "declare_partition_interpretation" in result["error_message"]
    assert calls == []


def test_gated_read_refusal_names_the_flagged_property(monkeypatch):
    """Catches a generic refusal message that leaves the model guessing which
    property it needs to declare."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls([])
    )
    result = gated("MATCH (n) RETURN sum(n.quantity)", FakeToolContext())
    assert "quantity" in result["error_message"]


def test_gated_read_detects_uppercase_aggregate_keywords(monkeypatch):
    """Catches a case-sensitive regex: Cypher's SUM/COUNT/etc. are not
    case-sensitive, and a model writing 'SUM(quantity)' must not silently
    bypass the gate."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls([])
    )
    result = gated("MATCH (n) RETURN SUM(n.quantity)", FakeToolContext())
    assert is_error(result)


def test_gated_read_runs_once_a_declaration_is_recorded(monkeypatch):
    """Catches a gate that keeps refusing even after a real declaration --
    the model would be unable to ever complete the query it was asked for."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    calls = []
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls(calls)
    )
    context = FakeToolContext({PARTITION_INTERPRETATION_DECLARED_KEY: True})
    result = gated("MATCH (n) RETURN sum(n.quantity)", context)
    assert is_success(result)
    assert len(calls) == 1


def test_gated_wrapper_calls_the_injected_implementation_exactly_once(monkeypatch):
    """Regression test for the closure-shadowing trap: if the inner closure
    ever called the bare name 'read_neo4j_cypher' instead of the bound
    default argument 'read_neo4j_cypher_impl', it would recurse into itself
    instead of the injected fake, and this test would hang or raise
    RecursionError instead of passing."""
    monkeypatch.setattr(
        variants_module,
        "peek_cached_profile",
        lambda: None,
    )
    calls = []
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls(calls)
    )
    gated("MATCH (n) RETURN n", FakeToolContext())
    assert len(calls) == 1


def test_gated_wrapper_presents_to_the_model_as_read_neo4j_cypher():
    """Catches a future refactor of make_gated_read_neo4j_cypher that renames
    its inner closure -- ADK derives the model-facing tool name from
    __name__, and step 2/3's instruction text names 'read_neo4j_cypher' by
    that literal name."""
    gated = variants_module.make_gated_read_neo4j_cypher(
        read_neo4j_cypher_impl=_fake_impl_recording_calls([])
    )
    assert gated.__name__ == "read_neo4j_cypher"


def test_v1_keeps_the_unwrapped_read_tool_by_identity():
    """Catches the gate leaking into v1's tools list -- v1 is the ungated A/B
    baseline and must not gain handoff-style mechanics, the same reasoning
    that keeps v1's 'finished' ungated."""
    tools = variants["graphrag_agent_v1"]["tools"]
    assert any(tool is read_neo4j_cypher for tool in tools)


def test_v2_holds_the_gated_read_tool_and_the_declare_tool():
    """Catches an instruction that tells the model to call
    'declare_partition_interpretation' when the tool was never added to the
    variant's tools list -- or a v2 tools list that still holds the bare,
    unwrapped read tool instead of the gated one."""
    tools = variants["graphrag_agent_v2"]["tools"]
    assert declare_partition_interpretation in tools
    assert read_neo4j_cypher not in tools
    assert any(getattr(tool, "__name__", None) == "read_neo4j_cypher" for tool in tools)


class FakeCallbackContext:
    """A callback context carrying state only -- callbacks never touch .actions."""

    def __init__(self, state=None):
        self.state = dict(state or {})


def test_reset_clears_a_previous_declaration():
    """Catches a reset that only initialises a missing key -- a declaration
    from an earlier turn would otherwise let the gate stay open on a later
    turn that never actually declared anything."""
    context = FakeCallbackContext({PARTITION_INTERPRETATION_DECLARED_KEY: True})
    reset_partition_interpretation_declaration(context)
    assert context.state[PARTITION_INTERPRETATION_DECLARED_KEY] is False


def test_reset_is_wired_onto_the_graphrag_agent():
    """Catches the callback being defined but never attached, which leaves
    the flag sticky for the whole session after the first declaration."""
    callbacks = graphrag_agent.canonical_before_agent_callbacks
    assert reset_partition_interpretation_declaration in callbacks


def test_reset_parameter_is_named_callback_context():
    """Catches a rename. ADK invokes these callbacks by keyword
    (base_agent.py:385-387), so a different parameter name fails at request
    time with a TypeError rather than at import."""
    parameters = list(
        inspect.signature(reset_partition_interpretation_declaration).parameters
    )
    assert parameters == ["callback_context"]


def test_both_resets_are_wired_together():
    """Catches one reset replacing the other in the before_agent_callback
    list instead of joining it -- the handoff-confirmation gate would
    silently stop being reset every turn."""
    callbacks = graphrag_agent.canonical_before_agent_callbacks
    assert reset_graphrag_handoff_confirmation in callbacks
    assert reset_partition_interpretation_declaration in callbacks
