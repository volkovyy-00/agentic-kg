# src/agentic_kg/common/adk_context.py
"""Context hygiene for agents that must not inherit other agents' claims.

ADK shows one agent the output of the others by rewriting each foreign event
into a user-role message (`_present_other_agent_message`, google/adk/flows/
llm_flows/_fencing.py): a fixed preamble part, then each of the other agent's
text parts fenced between quote markers. That is useful for an agent
summarising a colleague's work and actively harmful for one whose job is to
report what the database says: a warning another agent emitted hours earlier
arrives wearing the user's role and reads as ground truth. The fence tells the
model the quoted text is data, not instructions -- it does not stop the model
believing it.

Detection has to key on the preamble, not the role. By the time a
before_model_callback sees llm_request.contents, the converter has already set
the role to 'user', so a foreign event and a real human turn are
indistinguishable by role -- filtering on role would silently discard
everything the user actually typed. The preamble is the first part the
converter creates, and _get_contents copies one Content per event without
merging adjacent ones, so it reliably sits at index 0. The match is exact on
the whole preamble, so a user message that merely begins "For context:" is
kept. (The converter drops the event entirely when nothing but the preamble
would remain, which leaves nothing here to filter.)
"""

import logging
from typing import Any, Optional

# A private module, but ADK's own conformance tests import the constant from
# here, and the google-adk pin is one minor window. Imported, not copied, so
# ADK's wording can never drift from what we match; if a bump moves or renames
# it, this import fails loudly instead of the filter going quietly inert.
from google.adk.flows.llm_flows._fencing import OTHER_AGENT_CONTEXT_PREAMBLE

logger = logging.getLogger(__name__)

# The whole preamble, not its first words: a user message may start "For
# context:" too. tests/unit/test_adk_context.py drives ADK's real converter to
# check it still leads every foreign event.
FOREIGN_CONTEXT_SENTINEL: str = OTHER_AGENT_CONTEXT_PREAMBLE


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
