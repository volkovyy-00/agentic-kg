from google.adk.agents import Agent


from agentic_kg.common.llm_catalog import get_llm, LlmKind

from .variants import variants

AGENT_NAME = "graphrag_agent_v2"
graphrag_agent = Agent(
    name=AGENT_NAME,
    # Stays on the conversational tier deliberately: the experiment is whether
    # better information alone fixes the framing errors. Changing information
    # and model together would make the result uninterpretable.
    model=get_llm(LlmKind.conversational),
    description="Information retrieval from a knowledge graph using a range of query tools.", # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    # .get() so v1, which has no callback, stays a no-op: the field defaults to
    # None (llm_agent.py:225) and canonical_before_model_callbacks returns []
    # on falsy (390-391).
    before_model_callback=variants[AGENT_NAME].get("before_model_callback"),
)

root_agent = graphrag_agent
