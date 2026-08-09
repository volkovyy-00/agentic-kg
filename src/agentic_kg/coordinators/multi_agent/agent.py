from google.adk.agents import LlmAgent

from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR
from agentic_kg.common.config import validate_env
from agentic_kg.common.llm_catalog import LlmKind, get_llm
from agentic_kg.tools.cypher_tools import get_physical_schema, neo4j_is_ready
from agentic_kg.tools.file_tools import get_source_location

# Validated before importing sub_agents: each sub-agent constructs a LiteLlm
# instance at import time, so a missing/placeholder key should fail here,
# first, rather than after every sub-agent has already been built.
validate_env()

from .sub_agents import (
    file_suggestion_agent,
    graph_construction_agent,
    graphrag_agent,
    schema_proposal_agent,
    user_intent_agent,
)

full_workflow_agent = LlmAgent(
    name=MULTI_AGENT_COORDINATOR,
    description="""Knowledge graph construction using Neo4j.""",
    model=get_llm(LlmKind.conversational),
    instruction="""You are an expert in knowledge graph construction using Neo4j.
        Your primary goal is to guide the user through the process of knowledge graph construction.

        The user may want to check the setup before proceeding. Use tools for:
        - checking that the Neo4j database is ready using the 'neo4j_is_ready' tool
        - finding where source files are read from with the 'get_source_location' tool
        - checking whether the database is empty with 'get_physical_schema' tool

        Delegate to sub-agents to perform the work. Follow this sequence of agents:
        1. user_intent_agent -- start here to determine the user goal for kind of graph and description
        2. file_suggestion_agent -- requires approved user goals to make suggestions about what files to use
        3. schema_proposal_agent -- requires approved file suggestions to propose a graph schema with construction rules
        4. graph_construction_agent -- requires an approved graph schema design
        5. graphrag_agent -- used to interact with the knowledge graph.only available if 'get_physical_schema' tool shows that a graph exists
        """,
    sub_agents=[
        user_intent_agent,
        file_suggestion_agent,
        schema_proposal_agent,
        graph_construction_agent,
        graphrag_agent,
    ],
    tools=[get_physical_schema, get_source_location, neo4j_is_ready],
)

root_agent = full_workflow_agent
