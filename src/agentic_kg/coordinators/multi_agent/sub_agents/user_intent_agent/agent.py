from google.adk.agents import Agent

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.adk_transfer import strip_transfer_to_agent
from agentic_kg.common.llm_catalog import LlmKind, get_llm

# variants are pairs of instructions with tools
from .variants import variants

AGENT_NAME = "user_intent_agent_v2"

# Whether this variant is the GATED one. Named once rather than compared at
# the use site so that flipping which variant is selected is a single edit
# that cannot go half-applied: v1 must lose the strip along with the gate,
# since its ungated exit is the only one it has.
IS_GATED_VARIANT = AGENT_NAME == "user_intent_agent_v2"

user_intent_agent = Agent(
    name=AGENT_NAME,
    model=get_llm(LlmKind.conversational),
    description="Knowledge graph use case ideation.",
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    # Two MODEL callbacks, and deliberately NO before_agent_callback. ADK
    # injects its own 'transfer_to_agent' tool, plus an instruction
    # advertising it, into any LlmAgent with a parent or peers, and it does
    # not consult the approval gate in variants.py. That tool is the exit
    # the agent actually took in the reported session
    # (docs/backlog/user-goal-approval-never-recorded.md): it asked its
    # clarifying questions and transferred in the same reply, so the user's
    # agreement was heard by the coordinator, which has no approval tool.
    # strip_transfer_to_agent removes the declaration.
    #
    # drop_foreign_context removes the matching example. This agent is
    # entered BY the coordinator's own transfer_to_agent call, which ADK
    # rewrites into a "For context: [kg_construction_agent_v1] called tool
    # `transfer_to_agent`..." turn that then sits in this agent's history
    # for the whole interview -- a worked example of the exact tool name
    # and argument shape, on every turn of the stickiest phase there is.
    # Removing the declaration and leaving the example is half a fix: the
    # model copies it, the strip has popped it from tools_dict, and ADK
    # raises ValueError mid-turn -- a dead turn with no response and no
    # spinner, which is not a failure the model can recover from. Same
    # pairing, same reason, as graph_construction_agent/agent.py.
    #
    # Still NO before_agent_callback: graphrag_agent/agent.py carries one
    # because it gates on a per-turn boolean that must be reset; this gate
    # compares two durable state keys and has no flag to reset.
    #
    # Deliberately NOT disallow_transfer_to_parent: that flag would also
    # close the door, and would also stop Runner._find_agent_to_run
    # (runners.py:474-489) from returning this agent for the user's second
    # message, so every mid-interview reply would be re-arbitrated by the
    # coordinator. An interview is multi-turn by nature. See adk_transfer.py.
    before_model_callback=(
        [drop_foreign_context, strip_transfer_to_agent] if IS_GATED_VARIANT else None
    ),
)

root_agent = user_intent_agent
