# src/agentic_kg/common/adk_context.py
"""Context hygiene for agents that must not inherit other agents' claims.

ADK shows one agent the output of the others by rewriting each foreign event
into a user-role message (`_present_other_agent_message`, google/adk/flows/llm_flows/
contents.py). That is useful for an agent summarising a colleague's work and
actively harmful for one whose job is to report what the database says: a
warning another agent emitted hours earlier arrives wearing the user's role and
reads as ground truth.

Detection has to key on the sentinel text, not the role. By the time a
before_model_callback sees llm_request.contents, the converter has already set
both `role` and `author` to 'user', so a foreign event and a real human turn
are indistinguishable by role -- filtering on role would silently discard
everything the user actually typed. The sentinel is the first part the
converter creates, before its parts loop, and _get_contents deep-copies one
Content per event without merging adjacent ones, so it reliably sits at index
0. (The converter drops the event entirely when nothing but the sentinel would
remain, which leaves nothing here to filter.)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Must match google.adk.flows.llm_flows.contents._present_other_agent_message.
# tests/unit/test_adk_context.py drives ADK's real converter to detect drift.
FOREIGN_CONTEXT_SENTINEL = "For context:"


def _is_foreign(content: Any) -> bool:
    parts = getattr(content, "parts", None)
    if not parts:
        return False
    return getattr(parts[0], "text", None) == FOREIGN_CONTEXT_SENTINEL


def drop_foreign_context(callback_context: Any, llm_request: Any) -> Optional[None]:
    """Remove other agents' output from the request, in place.

    The parameter NAMES are load-bearing: ADK invokes this purely by keyword,
    as callback(callback_context=..., llm_request=...) (_handle_before_model_callback
    in base_llm_flow.py).
    Renaming either one fails at request time with a TypeError, not at import.

    Returns None so ADK proceeds with the (now filtered) request; a non-None
    return would short-circuit the model call entirely.
    """
    contents = getattr(llm_request, "contents", None)
    if not contents:
        return None

    kept = [c for c in contents if not _is_foreign(c)]
    dropped = len(contents) - len(kept)

    if dropped and not kept:
        # Filtering everything would hand the model an empty `contents`, which
        # most backends reject outright -- an unhandled exception mid-turn,
        # the failure mode send_query and send_read_query go out of their way
        # to avoid. Leaving the request untouched is the lesser harm: the model
        # sees context it should not have, rather than the turn dying.
        #
        # Not reachable through the coordinator today, where a real user turn
        # always survives the filter. This is a guard, not a code path in use.
        logger.warning(
            "Every message looked like foreign context; leaving the request "
            "unfiltered rather than sending an empty one"
        )
        return None

    if dropped:
        logger.debug("Dropped %d foreign-context message(s) before model call", dropped)
        llm_request.contents = kept
    return None
