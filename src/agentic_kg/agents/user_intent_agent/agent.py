from google.adk.agents import Agent

from agentic_kg.common.agent_guards import agent_guard_callbacks
from agentic_kg.common.llm_catalog import LlmKind, get_llm

# variants are pairs of instructions with tools
from .variants import variants

AGENT_NAME = "user_intent_agent_v1"


def build_user_intent_agent() -> Agent:
    return Agent(
        name=AGENT_NAME,
        # Explicit, so ADK 2.x never assigns one: see tests/unit/test_agent_modes.py.
        mode="chat",
        model=get_llm(LlmKind.reasoning),
        description="Knowledge graph use case ideation.",
        instruction=variants[AGENT_NAME]["instruction"],
        tools=variants[AGENT_NAME]["tools"],
        **agent_guard_callbacks(gated=False),
    )


# For compatibility with ADK CLI (`adk run path/to/agent_folder` expects a
# module-level `root_agent`). Importing this module directly will construct the
# agent, but the package no longer imports this by default, so general imports
# remain side-effect free.
root_agent = build_user_intent_agent()
