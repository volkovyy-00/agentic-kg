"""Coordinator agent names, shared with the sub-agents that report to them.

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
