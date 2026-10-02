"""A third root agent, outside both coordinators.

It builds `root_agent` from the standalone agents/user_intent_agent package,
not from the multi-agent workflow's user_intent_agent. The documented
`adk web src/agentic_kg/coordinators/` does not load it. The test suite's
agent-tree discovery does, through the `root_agent =` line below.
"""

from .agents.user_intent_agent.agent import build_user_intent_agent

root_agent = build_user_intent_agent()
