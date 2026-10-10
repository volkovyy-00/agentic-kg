"""Agent names shared across agent modules: coordinators, and peers that hand off to each other.

Sub-agents are constructed at import time, before their coordinator exists, so
they cannot discover their parent's name at runtime without reaching into ADK
private attributes. Holding the names here breaks that cycle.

This module must import nothing from the package, or the cycle returns. It
lives in common/ rather than under either coordinator because both trees need
it and agents/ must not depend on coordinators/.
"""

# coordinators/multi_agent/agent.py
MULTI_AGENT_COORDINATOR = "kg_construction_agent_v1"

# coordinators/single_agent/agent.py -- the parent of agents/cypher_agent
SINGLE_AGENT_COORDINATOR = "single_agent_agent_v1"

# coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py -- the
# construction agent's way back targets it.
SCHEMA_PROPOSAL_COORDINATOR = "schema_proposal_agent_coordinator"

# coordinators/multi_agent/sub_agents/graph_construction_agent/agent.py -- the
# schema stage hands a re-approved revision straight to it. Here, not imported
# from that module: it would import the schema stage back, a cycle.
GRAPH_CONSTRUCTION_AGENT = "graph_construction_agent_v1"
