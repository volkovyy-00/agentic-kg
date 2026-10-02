"""Every agent the package's root agents build, for tests that check them all.

Shared by test_agent_modes.py (each agent's ADK mode) and
test_agent_wiring_guards.py (output_key and the transfer-guard callbacks). A
helper module like fakes.py, so no test file imports another.

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


def root_agents() -> list[BaseAgent]:
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


def all_llm_agents() -> list[LlmAgent]:
    seen: dict[int, BaseAgent] = {}
    pending = root_agents()
    while pending:
        agent = pending.pop()
        if id(agent) not in seen:
            seen[id(agent)] = agent
            pending.extend(_children(agent))
    return [agent for agent in seen.values() if isinstance(agent, LlmAgent)]
