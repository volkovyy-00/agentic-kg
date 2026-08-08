"""Wiring pins for the schema proposal/critic pair.

Prompt text is not otherwise covered by anything: these assert the two facts
whose absence would silently disable the feature -- the evidence tool not being
reachable, and the revision paragraph not naming property_types.
"""
import pytest

from agentic_kg.common.value_types import ALLOWED_TYPES
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.variants import (
    variants,
)
from agentic_kg.tools.construction_plan_tools import (
    propose_node_construction,
    propose_relationship_construction,
)
from agentic_kg.tools.file_tools import column_type_hint, column_type_hints


def test_both_agents_can_call_the_type_hint_tool():
    """The proposal agent needs it to declare types; the critic needs it to
    challenge one. A critic without it can only object from the column name."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        assert column_type_hint in variants[name]["tools"], name


def test_the_proposal_agent_can_batch_type_hints():
    assert column_type_hints in variants["schema_proposal_agent_v1"]["tools"]


def test_the_revision_paragraph_names_property_types():
    """A propose call replaces the whole entry. A re-proposal that restates
    properties but forgets property_types silently reverts every declared type to
    text, and the plan looks identical -- the critic sees only the current
    snapshot and cannot detect it."""
    instruction = variants["schema_proposal_agent_v1"]["instruction"]
    assert "property_types" in instruction
    assert "restate every field" in instruction


def test_both_instructions_carry_the_property_type_rules():
    """The rules block is shared so the two cannot drift; if the subsection were
    added to only one, the critic could reject plans built to a different rule."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        assert "column_type_hint" in variants[name]["instruction"], name


@pytest.mark.parametrize("fn", [propose_node_construction, propose_relationship_construction])
def test_proposed_property_types_stays_optional_in_the_declaration(fn):
    """Pins TRAP 6: ADK 1.10.0 validates that a parameter's default is an
    instance of its annotation while building the FunctionDeclaration, and
    isinstance(None, dict) is False. A regression from
    'proposed_property_types: Optional[dict] = None' back to 'dict = None'
    raises ValueError at toolset construction and takes down the ENTIRE
    schema-proposal phase, not just this field -- and nothing else on this
    branch would catch it, since that failure only happens at runtime when
    the toolset is built. Building the declaration here reproduces that
    construction step as a unit test."""
    from google.adk.tools.function_tool import FunctionTool

    declared = FunctionTool(fn)._get_declaration()
    props = (declared.parameters.properties or {}) if declared.parameters else {}
    required = declared.parameters.required or [] if declared.parameters else []

    assert "proposed_property_types" in props
    assert "proposed_property_types" not in required


@pytest.mark.parametrize("fn", [propose_node_construction, propose_relationship_construction])
def test_every_allowed_type_is_named_in_the_tool_description(fn):
    """The closed set lives in value_types.ALLOWED_TYPES, but the model only
    ever learns it from prose -- these docstrings are the tool descriptions ADK
    sends. Adding a fourth type (dates are the named candidate) to the constant
    without touching the text leaves the model told it is illegal, and the
    consistency check would accept a type the model never proposes."""
    for allowed in ALLOWED_TYPES:
        assert allowed in fn.__doc__, allowed


def test_every_allowed_type_is_named_in_the_validation_rules():
    """Same staleness, one layer up: the shared rules block tells both agents
    which types exist, so a new entry in ALLOWED_TYPES that never reaches this
    prompt is a type the proposal agent will not use and the critic will
    reject."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        instruction = variants[name]["instruction"]
        for allowed in ALLOWED_TYPES:
            assert allowed in instruction, f"{name}: {allowed}"
