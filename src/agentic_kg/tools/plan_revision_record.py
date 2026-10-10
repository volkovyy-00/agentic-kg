"""The record that the user came back from construction to change the plan.

An ordinary session key, not `temp:` (which lasts one invocation, i.e. one user
message) and not a turn flag (cleared every turn): the clear question is asked
in one turn and answered in a later one. The way back writes it as pending; the
construction agent's check tool moves it to asked or answered; a successful
build sets it back to None. A failed build leaves it answered, so a retry
neither asks again nor erases again: the question is asked once per revision.

This module imports nothing from agentic_kg.tools. The build
(kg_construction_tools), the plan tools and the revision tools all import it,
and any import back from one of them would be a cycle.
"""

from typing import Any, Optional

from agentic_kg.common.session_state import SessionState

PLAN_REVISION_KEY = "plan_revision"

PENDING = "pending"
ASKED = "asked"
ANSWERED = "answered"

# What the build and the construction agent's constraint tool refuse with while
# the clear question is unsettled. Names the tool to call, or the agent loops.
REVISION_REFUSAL = (
    "The user came back to change the plan, and whether to clear the database "
    "before the rebuild is not settled yet. Call 'check_database_before_rebuild' "
    "and follow its result before creating constraints or building."
)


def revision(state: SessionState) -> Optional[dict[str, Any]]:
    """The current record, or None when no revision is pending."""
    value = state.get(PLAN_REVISION_KEY)
    return value if isinstance(value, dict) and value.get("status") else None


def mark_pending(state: SessionState) -> None:
    """The user went back; overwrites whatever the record held."""
    state[PLAN_REVISION_KEY] = {"status": PENDING}


def mark_asked(state: SessionState, invocation_id: str, contents: dict) -> None:
    """The question was put to the user in this invocation, about these contents."""
    state[PLAN_REVISION_KEY] = {
        "status": ASKED,
        "asked_in": invocation_id,
        "contents": contents,
    }


def mark_answered(state: SessionState, cleared: Optional[bool]) -> None:
    """Settled: cleared, kept, or (None) there was nothing to clear."""
    state[PLAN_REVISION_KEY] = {"status": ANSWERED, "cleared": cleared}


def clear_revision(state: SessionState) -> None:
    """The rebuild succeeded; no revision is pending."""
    state[PLAN_REVISION_KEY] = None


def revision_refusal(state: SessionState) -> Optional[str]:
    """REVISION_REFUSAL while the question is pending or asked, else None."""
    record = revision(state)
    if record is not None and record["status"] in (PENDING, ASKED):
        return REVISION_REFUSAL
    return None
