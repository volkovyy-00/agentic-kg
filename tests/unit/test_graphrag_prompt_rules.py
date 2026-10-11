"""Pins for the GraphRAG agent's rule about temporal properties.

The agent learns how to filter a typed date only from its prompt, and the traps
(a literal of the wrong type returns null and drops rows without an error) are
invisible in a result set, so the text is pinned.
"""

from agentic_kg.common.graph_profile import (
    _NEO4J_TRIM_MISSES,
    NUMERIC_CAST,
    NUMERIC_CAST_GAP_CHECK,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.variants import (
    variants,
)


def _instruction() -> str:
    return " ".join(variants["graphrag_agent_v2"]["instruction"].split())


def test_rule_6_says_temporal_properties_need_no_cast():
    text = _instruction()
    assert "DATE, DATE_TIME and LOCAL_DATE_TIME properties need no cast" in text
    assert "'numeric_like' reads 'no', or 'unknown' when the profile is sampled" in text


def test_rule_6_names_the_literal_that_matches_each_type():
    text = _instruction()
    assert "date('2025-03-04') for DATE" in text
    assert "datetime('2025-03-04T10:00:00Z') for DATE_TIME" in text
    assert "localdatetime('2025-03-04T10:00:00') for LOCAL_DATE_TIME" in text


def test_rule_6_warns_that_a_mismatched_literal_returns_null_not_an_error():
    text = _instruction()
    assert "returns null rather than an error and silently drops the rows" in text


def test_rule_6_says_zoned_and_local_values_cannot_be_compared():
    text = _instruction()
    assert (
        "A DATE_TIME value and a LOCAL_DATE_TIME value cannot be compared with each other"
        in text
    )


def _raw_instruction() -> str:
    """Not normalised: str.split() splits on the very characters these tests look for."""
    return variants["graphrag_agent_v2"]["instruction"]


def test_the_instruction_carries_no_brace_for_adk_to_substitute():
    raw = _raw_instruction()
    assert "{" not in raw
    assert "}" not in raw


def test_the_instruction_writes_whitespace_as_escapes_never_raw():
    raw = _raw_instruction()
    assert "\\u00A0" in raw
    for char in _NEO4J_TRIM_MISSES:
        assert char not in raw


def test_rule_6_quotes_the_generated_cast_and_gap_check_verbatim():
    raw = _raw_instruction()
    assert NUMERIC_CAST in raw
    assert NUMERIC_CAST_GAP_CHECK in raw


def test_rule_6_says_what_the_cast_does_and_when_to_check_it():
    text = _instruction()
    for phrase in (
        "every non-blank value the profile lists converts to a number",
        "reads brackets as a minus sign",
        "turns a blank into null",
        "Never cast such a property with a bare toFloat",
        "the profile listed only some of its values",
    ):
        assert phrase in text, phrase
    assert "a plain cast works" not in text
