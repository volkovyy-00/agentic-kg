from google.adk.agents import Agent

from agentic_kg.common.agent_names import SINGLE_AGENT_COORDINATOR
from agentic_kg.common.config import validate_env
from agentic_kg.common.llm_catalog import LlmKind, get_llm

# `adk web` discovers both coordinators, so single_agent needs its own
# validate_env() call -- it does not import multi_agent, and without this it
# starts with no key configured and fails opaquely at the first LLM call.
# Validated before importing cypher_agent, which constructs a LiteLlm
# instance at import time.
validate_env()

from agentic_kg.agents.cypher_agent.agent import cypher_agent

# variants are pairs of instructions with tools
from .variants import variants

AGENT_NAME = SINGLE_AGENT_COORDINATOR
single_agent_agent = Agent(
    name=AGENT_NAME,
    model=get_llm(LlmKind.conversational),
    description="Knowledge graph construction using Neo4j and cypher.",  # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],  # Make the tool available to this agent
    sub_agents=[cypher_agent],
)

root_agent = single_agent_agent
