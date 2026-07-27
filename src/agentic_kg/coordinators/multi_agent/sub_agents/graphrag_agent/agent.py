from google.adk.agents import Agent


from agentic_kg.common.llm_catalog import get_llm, LlmKind

from .variants import variants

AGENT_NAME = "graphrag_agent_v1"
graphrag_agent = Agent(
    name=AGENT_NAME,
    model=get_llm(LlmKind.conversational),
    description="Information retrieval from a knowledge graph using a range of query tools.", # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"], 
)

root_agent = graphrag_agent