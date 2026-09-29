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

Roots are found by reading files, not by walking packages: coordinators/ has
no __init__.py, so pkgutil would never reach the two coordinators.
"""

import importlib
import re
from pathlib import Path

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools.agent_tool import AgentTool

import agentic_kg

_PACKAGE_DIR = Path(agentic_kg.__file__).parent
_ROOT_ASSIGNMENT = re.compile(r"^root_agent\s*=", re.MULTILINE)


def _root_agents() -> list[BaseAgent]:
    roots = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        if not _ROOT_ASSIGNMENT.search(path.read_text()):
            continue
        relative = path.relative_to(_PACKAGE_DIR.parent).with_suffix("")
        module = importlib.import_module(".".join(relative.parts))
        roots.append(module.root_agent)
    return roots


def _children(agent: BaseAgent) -> list[BaseAgent]:
    wrapped = [
        tool.agent
        for tool in getattr(agent, "tools", [])
        if isinstance(tool, AgentTool)
    ]
    return [*agent.sub_agents, *wrapped]


def _all_llm_agents() -> list[LlmAgent]:
    seen: dict[int, BaseAgent] = {}
    pending = _root_agents()
    while pending:
        agent = pending.pop()
        if id(agent) not in seen:
            seen[id(agent)] = agent
            pending.extend(_children(agent))
    return [agent for agent in seen.values() if isinstance(agent, LlmAgent)]


def test_discovery_finds_every_coordinator():
    """Guards the discovery itself: if it found nothing, the rule below would
    pass while checking nothing."""
    names = {agent.name for agent in _root_agents()}
    assert {
        "kg_construction_agent_v1",
        "single_agent_agent_v1",
        "user_intent_agent_v1",
    } <= names


def test_every_agent_that_adk_could_assign_a_mode_has_one():
    wrong = []
    for agent in _all_llm_agents():
        parent = agent.parent_agent
        expected = "chat" if parent is None or isinstance(parent, LlmAgent) else None
        if agent.mode != expected:
            wrong.append(f"{agent.name}: mode={agent.mode!r}, expected {expected!r}")
    assert not wrong, "\n".join(wrong)
