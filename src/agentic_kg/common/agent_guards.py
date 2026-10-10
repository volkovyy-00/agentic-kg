"""The callbacks every LlmAgent carries, as Agent(...) keyword args.

Spread into every constructor -- `**agent_guard_callbacks(gated=...)` -- so no
agent gets part of a set; tests/unit/test_agent_wiring_guards.py fails if an
agent in any root lacks them.

Every agent gets the rejected-call cap (rejected_call_cap.py), which stops a
model stuck on calls ADK rejects. end_turn_past_cap is first among the
before-model callbacks: when it answers, ADK skips the ones after it.

A gated agent also gets the transfer guard (adk_transfer.py): the strip
removes the injected transfer tool, drop_foreign_context removes the worked
example of it from history, and the refusal answers a call made anyway --
ending the turn when that reply also spoke to the user, which is what the
after-model recorder is for.

An agent that needs its own callback in one of these slots extends this
helper rather than wiring lists; a duplicate keyword in the Agent(...) call
fails at import.
"""

from typing import Any

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.adk_transfer import (
    record_hidden_transfer_reply_text,
    refuse_transfer_to_agent,
    strip_transfer_to_agent,
)
from agentic_kg.common.rejected_call_cap import (
    end_turn_past_cap,
    mark_tool_outcome,
    mark_unknown_tool,
)


def agent_guard_callbacks(gated: bool) -> dict[str, Any]:
    """The callbacks an agent needs; `gated` adds the transfer guard."""
    if not gated:
        return {
            "before_model_callback": [end_turn_past_cap],
            "after_tool_callback": mark_tool_outcome,
            "on_tool_error_callback": mark_unknown_tool,
        }
    return {
        "before_model_callback": [
            end_turn_past_cap,
            drop_foreign_context,
            strip_transfer_to_agent,
        ],
        "after_model_callback": record_hidden_transfer_reply_text,
        "before_tool_callback": refuse_transfer_to_agent,
        "after_tool_callback": mark_tool_outcome,
        "on_tool_error_callback": mark_unknown_tool,
    }
