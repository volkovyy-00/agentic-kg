from google.adk.agents import Agent

from agentic_kg.common.agent_guards import agent_guard_callbacks
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
    # ADK gives this agent its own 'transfer_to_agent', which does not consult
    # the approval gate in variants.py. It is the exit the agent actually took
    # in the reported session (docs/backlog/user-goal-approval-never-recorded.md):
    # it asked its clarifying questions and transferred in the same reply, so
    # the user's agreement was heard by the coordinator, which has no approval
    # tool. agent_guard_callbacks removes the tool and the worked example of
    # it in this agent's history, which matters most here: the interview is
    # the stickiest phase. The mechanism, and why disallow_transfer_to_parent
    # is NOT used, are in common/adk_transfer.py, once. Here it
    # would send every mid-interview reply back to the coordinator.
    #
    # Deliberately NO before_agent_callback: graphrag_agent/agent.py carries
    # one because it gates on a per-turn boolean that must be reset; this gate
    # compares two durable state keys and has no flag to reset.
    **agent_guard_callbacks(gated=IS_GATED_VARIANT),
)

root_agent = user_intent_agent
