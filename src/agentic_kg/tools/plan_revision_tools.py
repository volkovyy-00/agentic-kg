"""The clear question the construction agent asks before rebuilding a revised plan.

After a way back (construction_handoff_tools / return_to_plan), the database
may still hold the previous build. Removing just that build is not possible --
builds merge into existing nodes -- so "clear" erases the whole database, and
the user sees what it holds first. The question is asked only after a way back,
only when the database is not empty, and once per revision; the record that
tracks it is plan_revision_record.

The answer must come in a later turn than the question: both answer tools
refuse inside the invocation (one user message) that asked it. That proves a
turn passed; it does not prove the user said yes -- the existing gates share
that limit. A specific required word ("erase") was considered and rejected: it
would refuse a natural "yes" in other wording or another language and is still
only a text match.
"""

from google.adk.tools import ToolContext

from agentic_kg.common.tool_result import ToolResult, is_error, tool_error, tool_success
from agentic_kg.tools.cypher_tools import (
    database_contents,
    describe_contents,
    is_empty,
    reset_neo4j_data,
)
from agentic_kg.tools.kg_construction_tools import NOT_APPROVED_MESSAGE, approved_plan
from agentic_kg.tools.plan_revision_record import (
    ANSWERED,
    ASKED,
    mark_answered,
    mark_asked,
    revision,
)

NO_REBUILD_PENDING = (
    "No rebuild is pending: no revised plan is waiting to be built. Carry on with "
    "whatever the user asked."
)
REBUILD_OWED = (
    "A revised plan is approved and not yet built. Run steps 1 to 6 against "
    "it; any earlier build in this conversation is the previous version."
)
OTHER_REQUEST = (
    "If the user's latest message asks for something else, such as changing the "
    "plan again, do that instead."
)
SAME_TURN_REFUSAL = (
    "The clear question was asked in this same turn, so the user has not answered "
    "it yet: nothing was recorded. End your reply with the question and wait for "
    "their answer."
)
NOT_ASKED_REFUSAL = (
    "There is no open question about clearing the database, so nothing was "
    "recorded. Call 'check_database_before_rebuild' first."
)


def _question(contents: dict, asked_earlier: bool) -> str:
    listing = describe_contents(contents)
    if asked_earlier:
        return (
            "The question whether to clear the database was put to the user in an "
            "earlier turn. If the user's latest message answers it, call "
            "'clear_database_for_rebuild' for yes or 'keep_database_for_rebuild' "
            "for no. " + OTHER_REQUEST + " If it does not answer, otherwise ask again: "
            "list what the database holds:\n"
            + listing
            + "\nSay that clearing erases the whole database, including anything this "
            "program did not build, and that keeping it rebuilds on top of what is "
            "there, and end your reply with the question."
        )
    return (
        "Ask the user now, and end your reply with the question. List what the "
        "database holds:\n" + listing + "\nSay that clearing erases the whole "
        "database, including anything this program did not build, and that "
        "keeping it rebuilds on top of what is there. Do not call "
        "'clear_database_for_rebuild' or 'keep_database_for_rebuild' in this reply."
    )


def check_database_before_rebuild(tool_context: ToolContext) -> ToolResult:
    """Check whether a revised plan is waiting to be rebuilt, and whether the user
    must first be asked about clearing the database.

    Call this at the start of every turn, before anything else, and follow what
    its result says.
    """
    state = tool_context.state
    record = revision(state)
    if record is None:
        return tool_success("rebuild", NO_REBUILD_PENDING)
    if approved_plan(state) is None:
        return tool_error(NOT_APPROVED_MESSAGE)
    if record["status"] == ANSWERED:
        return tool_success("rebuild", REBUILD_OWED + " " + OTHER_REQUEST)
    if record["status"] == ASKED:
        earlier = record["asked_in"] != tool_context.invocation_id
        return tool_success("question", _question(record["contents"], earlier))

    read = database_contents()
    if is_error(read):
        return read
    contents = read["contents"]
    if is_empty(contents):
        mark_answered(state, cleared=None)
        return tool_success("rebuild", "The database is empty. " + REBUILD_OWED)
    mark_asked(state, tool_context.invocation_id, contents)
    return tool_success("question", _question(contents, asked_earlier=False))


def _answer_refusal(tool_context: ToolContext) -> ToolResult | None:
    state = tool_context.state
    record = revision(state)
    if record is not None and approved_plan(state) is None:
        return tool_error(NOT_APPROVED_MESSAGE)
    if record is None or record["status"] != ASKED:
        return tool_error(NOT_ASKED_REFUSAL)
    if record["asked_in"] == tool_context.invocation_id:
        return tool_error(SAME_TURN_REFUSAL)
    return None


def clear_database_for_rebuild(tool_context: ToolContext) -> ToolResult:
    """Erase the whole database before rebuilding the revised plan.

    Call this only when the user, in their own words in this turn, answered yes
    to clearing the database. Refuses in the turn the question was asked.
    """
    refusal = _answer_refusal(tool_context)
    if refusal is not None:
        return refusal
    erased = reset_neo4j_data()
    if is_error(erased):
        return erased
    mark_answered(tool_context.state, cleared=True)
    return tool_success("cleared", "The database was cleared. " + REBUILD_OWED)


def keep_database_for_rebuild(tool_context: ToolContext) -> ToolResult:
    """Keep what the database holds and rebuild the revised plan on top of it.

    Call this only when the user, in their own words in this turn, answered no
    to clearing the database. Refuses in the turn the question was asked.
    """
    refusal = _answer_refusal(tool_context)
    if refusal is not None:
        return refusal
    read = database_contents()
    if is_error(read):
        return read
    mark_answered(tool_context.state, cleared=False)
    contents = read["contents"]
    plan = approved_plan(tool_context.state) or {}
    built_labels = {
        r.get("label") for r in plan.values() if r.get("construction_type") == "node"
    }
    built_types = {
        r.get("relationship_type")
        for r in plan.values()
        if r.get("construction_type") == "relationship"
    }
    return tool_success(
        "kept",
        {
            "labels_not_built": {
                k: v for k, v in contents["labels"].items() if k not in built_labels
            },
            "relationship_types_not_built": {
                k: v
                for k, v in contents["relationship_types"].items()
                if k not in built_types
            },
            "message": (
                "Tell the user which labels and relationship types are in the database but "
                "not built by the new plan, with their counts, and that the node and "
                "relationship counts reported after the build include data that was "
                "already there. " + REBUILD_OWED
            ),
        },
    )
