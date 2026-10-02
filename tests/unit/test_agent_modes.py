"""Every agent's ADK 'mode' is fixed at import, so ADK never picks one (KG-25).

google-adk 2.x writes a mode onto any LlmAgent whose mode is None, and keeps it
on the object: run as a workflow node -- which happens to a parent agent when a
sub-agent's 'finished' hands control back -- it becomes 'single_turn', and its
include_contents becomes 'none', so it stops seeing the conversation. Because
the agents are module-level singletons, that leaked from one test into the
next: tests make a sub-agent the Runner's root, so the coordinator above it was
never given a mode. (Under adk web the Runner sets a root's unset mode to
'chat' before every run, so the product was barely exposed.) An explicit
mode="chat" is never overwritten.

The rule, keyed on the agent's parent:
- no parent (a root) or an LlmAgent parent: mode == "chat";
- any other parent (today the refinement loop's two children, under a
  LoopAgent, which runs them directly and never as nodes): mode is None.

The tree comes from agent_tree.py, shared with test_agent_wiring_guards.py.
"""

from agent_tree import all_llm_agents, root_agents
from google.adk.agents import LlmAgent


def test_discovery_finds_every_coordinator():
    """Guards the discovery itself: if it found nothing, the rule below would
    pass while checking nothing."""
    names = {agent.name for agent in root_agents()}
    assert {
        "kg_construction_agent_v1",
        "single_agent_agent_v1",
        "user_intent_agent_v1",
    } <= names


def test_every_agent_that_adk_could_assign_a_mode_has_one():
    wrong = []
    for agent in all_llm_agents():
        parent = agent.parent_agent
        expected = "chat" if parent is None or isinstance(parent, LlmAgent) else None
        if agent.mode != expected:
            wrong.append(f"{agent.name}: mode={agent.mode!r}, expected {expected!r}")
    assert not wrong, "\n".join(wrong)
