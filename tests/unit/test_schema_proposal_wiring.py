"""Wiring pins for the schema proposal/critic pair.

Prompt text is not otherwise covered by anything: these assert the two facts
whose absence would silently disable the feature -- the evidence tool not being
reachable, and the revision paragraph not naming property_types.
"""
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.variants import (
    variants,
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
