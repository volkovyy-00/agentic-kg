"""Every agent the package's root agents build, for tests that check them all.

Shared by test_agent_modes.py (each agent's ADK mode) and
test_agent_wiring_guards.py (output_key and the transfer-guard callbacks). A
helper module like fakes.py, so no test file imports another.

Roots are found by reading files, not by walking packages: coordinators/ has
no __init__.py, so pkgutil would never reach the two coordinators. The walk is
cached: the agents are module-level singletons, so one walk serves every test.
"""

import importlib
import re
from functools import cache
from pathlib import Path

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools.agent_tool import AgentTool

import agentic_kg

PACKAGE_DIR = Path(agentic_kg.__file__).parent
# Plain or annotated (`root_agent: LlmAgent = ...`); never an attribute
# assignment or a comparison of an imported root_agent.
ROOT_ASSIGNMENT = re.compile(r"^root_agent\s*(?::[^=\n]+)?=(?!=)", re.MULTILINE)


@cache
def root_files() -> list[Path]:
    """Every module under the package that assigns a top-level root_agent."""
    return [
        path
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        if ROOT_ASSIGNMENT.search(path.read_text())
    ]


@cache
def root_agents() -> list[BaseAgent]:
    roots = []
    for path in root_files():
        relative = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
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


@cache
def all_llm_agents() -> list[LlmAgent]:
    seen: dict[int, BaseAgent] = {}
    pending = list(root_agents())  # the cached list is never consumed
    while pending:
        agent = pending.pop()
        if id(agent) not in seen:
            seen[id(agent)] = agent
            pending.extend(_children(agent))
    return [agent for agent in seen.values() if isinstance(agent, LlmAgent)]


@cache
def user_facing_llm_agents() -> list[LlmAgent]:
    """Every LlmAgent a user can talk to: the walk through sub_agents only.

    An agent reached only through an AgentTool (the refinement loop's
    proposer and critic) runs inside another agent's tool call, and its
    output reaches the user only through that agent.
    """
    seen: dict[int, BaseAgent] = {}
    pending = list(root_agents())
    while pending:
        agent = pending.pop()
        if id(agent) not in seen:
            seen[id(agent)] = agent
            pending.extend(agent.sub_agents)
    return [agent for agent in seen.values() if isinstance(agent, LlmAgent)]
