from google.adk.agents import Agent

from agentic_kg.common.agent_guards import agent_guard_callbacks
from agentic_kg.common.llm_catalog import LlmKind, get_llm

from .variants import variants

AGENT_NAME = "file_suggestion_agent_v1"
file_suggestion_agent = Agent(
    name=AGENT_NAME,
    description="Helps the user select files to import.",
    model=get_llm(LlmKind.conversational),
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    **agent_guard_callbacks(gated=False),
)

root_agent = file_suggestion_agent
