from google.adk.agents import Agent

from agentic_kg.common.adk_transfer import strip_transfer_to_agent
from agentic_kg.common.llm_catalog import get_llm, LlmKind

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
        # ONE callback. ADK injects its own 'transfer_to_agent' tool, plus an
        # instruction advertising it, into any LlmAgent with a parent or peers,
        # and it does not consult the approval gate in variants.py. That tool
        # is the exit the agent actually took in the reported session
        # (docs/backlog/user-goal-approval-never-recorded.md): it asked its
        # clarifying questions and transferred in the same reply, so the user's
        # agreement was heard by the coordinator, which has no approval tool.
        #
        # Deliberately NOT a list, and deliberately NOT accompanied by a
        # before_agent_callback. graphrag_agent/agent.py carries two of each
        # because it gates on a per-turn boolean that must be reset; this gate
        # compares two durable state keys and has no flag to reset. Copying
        # that shape here would reintroduce exactly the machinery this design
        # avoids. drop_foreign_context is likewise absent: this agent answers
        # nothing from graph state.
        #
        # Deliberately NOT disallow_transfer_to_parent: that flag would also
        # close the door, and would also stop Runner._find_agent_to_run
        # (runners.py:474-489) from returning this agent for the user's second
        # message, so every mid-interview reply would be re-arbitrated by the
        # coordinator. An interview is multi-turn by nature. See adk_transfer.py.
        before_model_callback=strip_transfer_to_agent if IS_GATED_VARIANT else None,
    )

root_agent = user_intent_agent
