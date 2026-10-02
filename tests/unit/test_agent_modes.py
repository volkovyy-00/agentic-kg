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

from agent_tree import (
    PACKAGE_DIR,
    ROOT_ASSIGNMENT,
    all_llm_agents,
    root_agents,
    root_files,
)
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


def test_discovery_reaches_every_coordinator_directory():
    """Guards the discovery's pattern: a coordinator whose agent.py exposes its
    root in a form the pattern misses would be skipped by every tree-walking
    guard, which would then pass without checking it."""
    expected = sorted((PACKAGE_DIR / "coordinators").glob("*/agent.py"))
    assert expected
    missing = [path for path in expected if path not in root_files()]
    assert not missing, f"no root_agent found in: {missing}"


def test_the_root_pattern_matches_root_assignments_only():
    """root_agents() imports every module the pattern matches and reads its
    root_agent, so a module that only uses an imported one must not match."""
    lines = {
        "root_agent = agent": True,
        "root_agent: LlmAgent = agent": True,
        'root_agent.name = "x"': False,
        "root_agent != other": False,
        "root_agent <= other": False,
        "root_agent == other": False,
        "    root_agent = agent": False,
    }
    wrong = [
        line
        for line, is_root in lines.items()
        if bool(ROOT_ASSIGNMENT.search(line)) != is_root
    ]
    assert not wrong, f"misread as (not) a root assignment: {wrong}"
