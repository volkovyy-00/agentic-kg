"""TurnFlag: the key and plumbing every turn-scoped gate shares."""

from agentic_kg.common.turn_flags import TurnFlag
from agentic_kg.tools.construction_handoff_tools import HANDOFF_CONFIRMED
from agentic_kg.tools.graphrag_handoff_tools import GRAPHRAG_HANDOFF_CONFIRMED
from agentic_kg.tools.graphrag_partition_tools import (
    PARTITION_INTERPRETATION_DECLARED,
)


def test_set_writes_the_literal_true():
    state = {}
    TurnFlag("k").set(state)
    assert state["k"] is True


def test_reset_writes_the_literal_false():
    """test_construction_handoff_gate.py asserts `is False`, so a falsy None
    or 0 would break it."""
    state = {"k": True}
    TurnFlag("k").reset(state)
    assert state["k"] is False


def test_is_set_reads_missing_as_unset():
    assert TurnFlag("k").is_set({}) is False


def test_is_set_after_set_and_after_reset():
    flag, state = TurnFlag("k"), {}
    flag.set(state)
    assert flag.is_set(state) is True
    flag.reset(state)
    assert flag.is_set(state) is False


def test_the_existing_gates_keep_their_key_strings():
    """Session state written by an earlier version must still be read."""
    assert HANDOFF_CONFIRMED.key == "construction_handoff_confirmed"
    assert GRAPHRAG_HANDOFF_CONFIRMED.key == "graphrag_handoff_confirmed"
    assert PARTITION_INTERPRETATION_DECLARED.key == "partition_interpretation_declared"
