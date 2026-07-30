# Schema Refinement Loop Latency Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a bug where `schema_refinement_loop`'s retry round silently discards the critic's
feedback, and add a code-enforced cap of one `schema_refinement_loop` invocation per user turn, so
worst-case per-turn latency in the schema-proposal stage is bounded and predictable.

**Architecture:** Both fixes live in one file, `schema_proposal_agent/agent.py`. A single
`before_agent_callback` moves from `schema_proposal_agent` (where it wrongly fires every loop
iteration) to `refinement_loop`, the `LoopAgent` itself (where it fires exactly once per
invocation) — this callback resets `feedback` for a fresh invocation and, using ADK's
`before_agent_callback` short-circuit contract, refuses a second invocation within the same turn.
A second, small callback on the coordinator (`root_agent`) resets that turn's budget to zero so
every fresh user message gets to try once. No new files for production code; two new test files.

**Tech Stack:** Python 3.12, `uv`, Google ADK 1.10.0 (pinned, see Global Constraints), pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-schema-refinement-loop-latency-design.md` —
authoritative, reviewed twice against ADK source, approved. Read it before starting.

## Global Constraints

- **Do not touch the batching, `reasoning_effort`, or critic-model questions** — all three were
  investigated in the spec and explicitly rejected/deferred. This plan is scope-limited to the two
  approved fixes.
- **Do not enable the existing commented-out `# before_agent_callback=initialize_schema_and_construction_plan`
  line on `refinement_loop`** (`agent.py:98` as of this writing) — leave it exactly as a comment.
  Mention it in the PR description so a reviewer doesn't wonder if this change was meant to touch it.
- **`_VALIDATION_RULES` in `variants.py` is shared between `schema_proposal_agent_v1` and
  `schema_critic_agent_v1` — do not edit it.** Out of scope for this change entirely.
- Run unit tests with `uv run pytest -q` (must stay green, no Docker needed for anything in this
  plan). Do not pipe through `tail` — `$?` would report `tail`'s exit code, not pytest's:
  `uv run pytest -q > /tmp/out.log 2>&1; echo "exit: $?"; cat /tmp/out.log`.
- Work on branch `reduce-schema-refinement-latency` (already checked out). Commit after every task.
- **Reassigning `.model`/`.before_agent_callback` on an already-constructed agent object is safe** —
  `BaseAgent`'s `model_config` sets neither `frozen=True` nor `validate_assignment`, so post-construction
  attribute assignment is both unblocked and unvalidated (a wrong-typed value would be accepted
  silently, not rejected — don't rely on pydantic to catch a mistake here). Used in Task 2's tests;
  directly verified against this exact installed ADK version (`1.10.0`) before this plan was written,
  no need to re-verify it. If `google-adk` is ever bumped past `<2` (currently far behind upstream,
  which is at `1.36.2`+), re-verify this assumption — it was not checked against any other version.

---

## File Structure

**Modified files**

| Path | Change |
|---|---|
| `src/agentic_kg/coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py` | Replace `initialize_feedback` with `prepare_refinement_loop_invocation` (moved to `refinement_loop`'s `before_agent_callback`) and add `reset_schema_refinement_turn_budget` (on `root_agent`'s `before_agent_callback`); add one instruction bullet for the `stopped:` case. |

**New files**

| Path | Responsibility |
|---|---|
| `tests/unit/test_schema_refinement_loop_callbacks.py` | Unit tests for the two new callback functions: feedback preservation, short-circuit behavior, `stopped:` message shape, and that the callbacks are wired to the right agent objects. |
| `tests/unit/test_schema_refinement_loop_turn_cap.py` | The one empirical, Runner-driven test: two sequential user turns through a real `InMemoryRunner`, with all three agents' models replaced by a scripted fake, proving the turn-budget reset happens once per incoming user message — the one fact source-reading alone couldn't settle (see spec's *Verification* section). |

---

## Task 1: Fix the feedback-clobbering bug and add the code-enforced turn cap

**Why first:** this is the actual production code change; everything else is testing it.

**Files:**
- Modify: `src/agentic_kg/coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py`
- Test: `tests/unit/test_schema_refinement_loop_callbacks.py` (new)

**Interfaces:**
- Consumes: nothing new — `types.Content`/`types.Part` from `google.genai.types` (already imported
  in `agent.py` as `types`), `Optional` from `typing`.
- Produces (for Task 2 and Task 3 to import):
  - `prepare_refinement_loop_invocation(callback_context) -> Optional[types.Content]` — module-level
    function in `agent.py`.
  - `reset_schema_refinement_turn_budget(callback_context) -> None` — module-level function in
    `agent.py`.
  - `refinement_loop.before_agent_callback` set to `prepare_refinement_loop_invocation`.
  - `root_agent.before_agent_callback` set to `reset_schema_refinement_turn_budget`.
  - State key `schema_refinement_calls_this_turn` (int), alongside the existing `feedback` (str) key.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_schema_refinement_loop_callbacks.py`:

```python
"""prepare_refinement_loop_invocation must reset 'feedback' exactly once per
schema_refinement_loop invocation (never mid-loop), and must enforce at most
one such invocation per user turn.

Regression test for a bug found via live smoke-testing on 2026-07-29: the
reset used to live on schema_proposal_agent's own before_agent_callback,
which ADK's LoopAgent re-invokes on every iteration -- so the loop's retry
round always saw an empty <feedback> block and silently re-derived the
schema from scratch instead of acting on the critic's actual objections.
See docs/superpowers/specs/2026-07-29-schema-refinement-loop-latency-design.md.
"""
from types import SimpleNamespace

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    prepare_refinement_loop_invocation,
    reset_schema_refinement_turn_budget,
    refinement_loop,
    schema_proposal_agent,
    root_agent,
)


def _ctx(state):
    return SimpleNamespace(state=state)


def test_first_invocation_this_turn_resets_feedback_and_proceeds():
    state = {
        "schema_refinement_calls_this_turn": 0,
        "feedback": "stale text from a previous turn",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    assert result is None
    assert state["feedback"] == ""
    assert state["schema_refinement_calls_this_turn"] == 1


def test_second_invocation_this_turn_short_circuits_without_touching_feedback():
    state = {
        "schema_refinement_calls_this_turn": 1,
        "feedback": "retry\n- bad join key",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    assert result is not None
    # Untouched: the short-circuit message below quotes this value verbatim.
    assert state["feedback"] == "retry\n- bad join key"
    assert state["schema_refinement_calls_this_turn"] == 2


def test_short_circuit_message_quotes_feedback_and_cannot_be_misrouted():
    state = {
        "schema_refinement_calls_this_turn": 1,
        "feedback": "retry\n- bad join key",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    text = "\n".join(p.text for p in result.parts if p.text)
    assert "retry\n- bad join key" in text
    # The coordinator routes on whether the tool result BEGINS WITH 'retry';
    # this message must not collide with that check.
    assert not text.strip().lower().startswith("retry")
    assert text.strip().lower().startswith("stopped:")


def test_schema_proposal_agent_no_longer_resets_feedback_itself():
    """The exact bug: schema_proposal_agent used to carry its own
    before_agent_callback, which LoopAgent fires on every iteration. It must
    carry none now -- the reset lives one level up, on refinement_loop,
    which fires only once per invocation."""
    assert schema_proposal_agent.before_agent_callback is None


def test_refinement_loop_carries_the_new_callback():
    assert refinement_loop.before_agent_callback is prepare_refinement_loop_invocation


def test_reset_schema_refinement_turn_budget_zeroes_the_counter():
    state = {"schema_refinement_calls_this_turn": 5}
    reset_schema_refinement_turn_budget(_ctx(state))
    assert state["schema_refinement_calls_this_turn"] == 0


def test_coordinator_carries_the_reset_callback():
    assert root_agent.before_agent_callback is reset_schema_refinement_turn_budget
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_schema_refinement_loop_callbacks.py -v`

Expected: `ImportError: cannot import name 'prepare_refinement_loop_invocation' from ...` at collection
time — `agent.py` does not yet define `prepare_refinement_loop_invocation` or
`reset_schema_refinement_turn_budget`.

- [ ] **Step 3: Implement the fix**

In `src/agentic_kg/coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py`:

Change line 7 from:
```python
from typing import AsyncGenerator
```
to:
```python
from typing import AsyncGenerator, Optional
```

Replace the existing block (current lines 23-29):
```python
# initialize context for schema_proposal_agent with blank feedback, which may get filled later by the schema_critic_agent
def initialize_feedback(callback_context: CallbackContext) -> None:
    callback_context.state["feedback"] = ""

def initialize_schema_and_construction_plan(callback_context: CallbackContext) -> None:
    callback_context.state["proposed_schema"] = ""
    callback_context.state["proposed_construction_plan"] = []
```
with:
```python
def prepare_refinement_loop_invocation(callback_context: CallbackContext) -> Optional[types.Content]:
    """Runs once per schema_refinement_loop invocation (never mid-loop, since
    this is attached to refinement_loop itself, not to schema_proposal_agent
    -- a LoopAgent's own before_agent_callback fires once per call to
    LoopAgent.run_async, while each sub-agent's before_agent_callback would
    refire on every internal iteration).

    Resets 'feedback' for a fresh invocation, and enforces at most one
    invocation of this loop per user turn: increment the counter before
    checking it, and leave 'feedback' untouched on the short-circuited path
    so the returned message can quote the critic's actual last verdict.
    """
    calls = callback_context.state.get("schema_refinement_calls_this_turn", 0) + 1
    callback_context.state["schema_refinement_calls_this_turn"] = calls
    if calls > 1:
        last_feedback = callback_context.state.get("feedback", "")
        return types.Content(
            role="model",
            parts=[types.Part(text=(
                "stopped: schema_refinement_loop already ran once this turn "
                f"(last verdict: {last_feedback}). Do not call it again this "
                "turn -- call get_proposed_construction_plan and present that "
                "plan together with the verdict above, and let the user decide."
            ))],
        )
    callback_context.state["feedback"] = ""
    return None


def reset_schema_refinement_turn_budget(callback_context: CallbackContext) -> None:
    """Runs once per incoming user turn, on the coordinator itself, giving
    every fresh turn a new one-invocation budget for schema_refinement_loop."""
    callback_context.state["schema_refinement_calls_this_turn"] = 0


def initialize_schema_and_construction_plan(callback_context: CallbackContext) -> None:
    callback_context.state["proposed_schema"] = ""
    callback_context.state["proposed_construction_plan"] = []
```

Remove the `before_agent_callback=initialize_feedback` line from `schema_proposal_agent`'s
constructor (current lines 32-39):
```python
AGENT_NAME = "schema_proposal_agent_v1"
schema_proposal_agent = LlmAgent(
    name=AGENT_NAME,
    description="Proposes a knowledge graph schema based on the user goal and approved file list",
    model=get_llm(LlmKind.reasoning),
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"], 
    before_agent_callback=initialize_feedback
)
```
becomes:
```python
AGENT_NAME = "schema_proposal_agent_v1"
schema_proposal_agent = LlmAgent(
    name=AGENT_NAME,
    description="Proposes a knowledge graph schema based on the user goal and approved file list",
    model=get_llm(LlmKind.reasoning),
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"], 
)
```

Add the callback to `refinement_loop` (current lines 93-99):
```python
refinement_loop = LoopAgent(
    name="schema_refinement_loop",
    description="Analyzes approved files to propose a graph construction plan based on user intent and feedback",
    max_iterations=2,
    sub_agents=[schema_proposal_agent, schema_critic_agent, CheckStatusAndEscalate(name="StopChecker")],
    # before_agent_callback=initialize_schema_and_construction_plan
)
```
becomes:
```python
refinement_loop = LoopAgent(
    name="schema_refinement_loop",
    description="Analyzes approved files to propose a graph construction plan based on user intent and feedback",
    max_iterations=2,
    sub_agents=[schema_proposal_agent, schema_critic_agent, CheckStatusAndEscalate(name="StopChecker")],
    before_agent_callback=prepare_refinement_loop_invocation,
    # before_agent_callback=initialize_schema_and_construction_plan
)
```
(the commented-out line stays exactly as it was — out of scope, see Global Constraints).

Add the reset callback to `root_agent`, and one new instruction bullet for the `stopped:` case
(current lines 101-107, inside the larger instruction string that continues below):
```python
root_agent = LlmAgent(
    name="schema_proposal_agent_coordinator",
    model=get_llm(LlmKind.reasoning),
    instruction="""
    You are a coordinator for the graph construction plan process. Use tools to propose a schema to the user.
    If the user disapproves, use the tools to refine the schema and ask the user to approve again.
    When the schema approval has been recorded, use the 'finished' tool.
```
becomes:
```python
root_agent = LlmAgent(
    name="schema_proposal_agent_coordinator",
    model=get_llm(LlmKind.reasoning),
    before_agent_callback=reset_schema_refinement_turn_budget,
    instruction="""
    You are a coordinator for the graph construction plan process. Use tools to propose a schema to the user.
    If the user disapproves, use the tools to refine the schema and ask the user to approve again.
    When the schema approval has been recorded, use the 'finished' tool.
```

Add a new bullet immediately after the existing "If the verdict the loop returns begins with
'retry'..." bullet (current lines 130-136), so the rule list reads:
```python
    - If the verdict the loop returns begins with 'retry', the critic found problems that are still in
      the plan: call 'schema_refinement_loop' again, passing that retry feedback, instead of presenting a
      plan with known problems for approval. Do this at most once for a given problem. If the loop
      returns 'retry' a second time, stop calling it: some objections cannot be fixed by changing the
      schema, because they are properties of the data. Call 'get_proposed_construction_plan', show the
      user that plan together with the critic's remaining objections, and let them decide whether to
      approve it as it stands.
    - If the verdict the loop returns begins with 'stopped:', 'schema_refinement_loop' has already run once
      this turn and refused to run again -- do not call it again this turn no matter what. Call
      'get_proposed_construction_plan' and present that plan together with the verdict's last-known
      feedback (quoted in the 'stopped:' message), and let the user decide whether to approve it or ask
      for another change, which will run in a fresh turn with a new budget.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_schema_refinement_loop_callbacks.py -v`

Expected: 7 passed.

- [ ] **Step 5: Run the full unit suite to check for regressions**

Run: `uv run pytest -q > /tmp/schema_loop_task1.log 2>&1; echo "exit: $?"; cat /tmp/schema_loop_task1.log`

Expected: exit 0, all prior tests still pass (the existing `tests/unit/test_schema_refinement_loop.py`
`StopChecker` tests are untouched by this change and must still pass).

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py \
        tests/unit/test_schema_refinement_loop_callbacks.py
git commit -m "fix: stop schema_refinement_loop from clobbering feedback mid-loop, cap at one invocation per turn"
```

---

## Task 2: Empirical two-turn test proving the turn-budget reset

**Why this task exists:** the spec identifies one fact that source-reading ADK internals cannot
settle — whether `schema_proposal_agent_coordinator`'s `before_agent_callback` really fires exactly
once per incoming user message under ADK's scheduling, as opposed to some other number of times.
This task closes that gap empirically, using ADK's own `InMemoryRunner` test harness with the real
production agents, and only their models replaced by a scripted fake — not a hand-built
`InvocationContext`, which is fragile and untested ADK-internal plumbing to hand-roll ourselves.

**Files:**
- Test: `tests/unit/test_schema_refinement_loop_turn_cap.py` (new)

**Interfaces:**
- Consumes: `root_agent`, `schema_proposal_agent`, `schema_critic_agent` from
  `schema_proposal_agent/agent.py` (Task 1's fixed versions); `google.adk.models.base_llm.BaseLlm`,
  `google.adk.models.llm_response.LlmResponse`, `google.adk.runners.InMemoryRunner`,
  `google.genai.types` (all already present in the installed `google-adk` dependency, no new
  packages needed).
- Produces: nothing consumed by a later task — this is the last test task.

- [ ] **Step 1: Write the test**

Create `tests/unit/test_schema_refinement_loop_turn_cap.py`:

```python
"""Empirical proof that the coordinator's before_agent_callback -- which
resets the schema_refinement_loop turn budget -- fires exactly once per
incoming user message, not once per some other unit of ADK scheduling.

This is deliberately a Runner-driven test rather than a hand-built
InvocationContext: "how many times does ADK call run_async on the
coordinator per user turn" is a framework scheduling behavior, and the only
way to answer it without trusting an assumption is to actually run it
through ADK's own Runner. The three agents' models are replaced with a
scripted fake so the test is fast and deterministic -- it is not testing
whether gpt-5 decides to call the tool, only whether the turn-budget
mechanism holds when it does.
"""
import asyncio

import pytest
from pydantic import Field
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    root_agent,
    schema_proposal_agent,
    schema_critic_agent,
)


class ScriptedLlm(BaseLlm):
    """Replays a fixed script of LlmResponses, one per call; holds on the
    last one if more calls arrive than scripted responses."""

    responses: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _tool_call_response(request_text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(
            name="schema_refinement_loop", args={"request": request_text},
        )),
    ]))


def test_each_user_turn_gets_its_own_one_call_budget():
    async def run():
        # schema_proposal_agent_v1 and schema_critic_agent_v1 only need to
        # end their turn immediately (plain text, no tool calls) so the one
        # allowed schema_refinement_loop invocation per turn completes fast.
        schema_proposal_agent.model = ScriptedLlm(
            model="scripted", responses=[_text_response("a minimal schema proposal")],
        )
        schema_critic_agent.model = ScriptedLlm(
            model="scripted", responses=[_text_response("valid")],
        )
        # Coordinator: turn 1 tries to call schema_refinement_loop TWICE in a
        # row (reproducing the exact live-observed misbehavior this fix
        # guards against), then finalizes. Turn 2 tries once, then finalizes.
        root_agent.model = ScriptedLlm(model="scripted", responses=[
            _tool_call_response("propose an initial schema"),
            _tool_call_response("the user asked for another change"),
            _text_response("turn 1 final response"),
            _tool_call_response("the user asked for yet another change"),
            _text_response("turn 2 final response"),
        ])

        runner = InMemoryRunner(agent=root_agent, app_name="turn_cap_test")
        session = await runner.session_service.create_session(
            app_name="turn_cap_test", user_id="u1",
        )

        turn_1_events = [
            event async for event in runner.run_async(
                user_id="u1", session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please propose a schema")],
                ),
            )
        ]
        # Captured here, before turn 2 runs a second real round and bumps
        # these counts again -- this is the actual proof (a call-count spy,
        # not a text-content coincidence) that the short-circuited second
        # schema_refinement_loop call in turn 1 never invoked either
        # sub-agent's model at all.
        calls_after_turn_1 = (
            schema_proposal_agent.model.call_count,
            schema_critic_agent.model.call_count,
        )

        turn_2_events = [
            event async for event in runner.run_async(
                user_id="u1", session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please change it again")],
                ),
            )
        ]
        calls_after_turn_2 = (
            schema_proposal_agent.model.call_count,
            schema_critic_agent.model.call_count,
        )
        return turn_1_events, turn_2_events, calls_after_turn_1, calls_after_turn_2

    turn_1_events, turn_2_events, calls_after_turn_1, calls_after_turn_2 = asyncio.run(run())

    def _function_response_texts(events):
        texts = []
        for event in events:
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if part.function_response and part.function_response.response:
                    texts.append(str(part.function_response.response.get("result", "")))
        return texts

    turn_1_results = _function_response_texts(turn_1_events)
    turn_2_results = _function_response_texts(turn_2_events)

    # Turn 1: two schema_refinement_loop calls were made. The first
    # succeeds (real verdict), the second is short-circuited.
    assert len(turn_1_results) == 2
    assert turn_1_results[0] == "valid"
    assert turn_1_results[1].startswith("stopped:")

    # The spy: each sub-agent's model was called exactly once during turn 1,
    # even though schema_refinement_loop was invoked twice -- proving the
    # second invocation's before_agent_callback short-circuit genuinely
    # skipped running schema_proposal_agent_v1 and schema_critic_agent_v1,
    # not just that it happened to return matching text.
    assert calls_after_turn_1 == (1, 1)

    # Turn 2: a fresh message resets the budget, so its one call succeeds
    # again (a second real round) rather than being short-circuited.
    assert len(turn_2_results) == 1
    assert turn_2_results[0] == "valid"
    assert calls_after_turn_2 == (2, 2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_schema_refinement_loop_turn_cap.py -v`

Expected: FAILS on `assert turn_1_results[1].startswith("stopped:")` (or a similar assertion) —
without Task 1's fix, this would be a second real `"valid"`/`"retry"` verdict from an actual second
loop run, not a short-circuit. If Task 1 was completed first, this should already PASS; if so,
treat this step as confirmation rather than a red-first step, and move on.

- [ ] **Step 3: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_schema_refinement_loop_turn_cap.py -v`

Expected: PASS (1 passed). If Task 1 is already implemented correctly, this should pass without
further code changes — this task is pure test-writing, not a new production-code change.

- [ ] **Step 4: Run the full unit suite one more time**

Run: `uv run pytest -q > /tmp/schema_loop_task2.log 2>&1; echo "exit: $?"; cat /tmp/schema_loop_task2.log`

Expected: exit 0, all tests pass, including both new test files from Task 1 and Task 2.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_schema_refinement_loop_turn_cap.py
git commit -m "test: empirically verify schema_refinement_loop's turn budget resets once per user message"
```

---

## Task 3: Final regression pass and PR notes

**Why this task exists:** confirm nothing else in the five-stage coordinator pipeline was disturbed,
and leave an accurate paper trail for review (this branch also carries an unrelated CLAUDE.md
documentation commit from earlier in the session, already committed).

**Files:** none modified — verification and PR-description only.

**Interfaces:** none.

- [ ] **Step 1: Run the complete unit test suite from a clean state**

```bash
uv run pytest -q > /tmp/schema_loop_final.log 2>&1; echo "exit: $?"; cat /tmp/schema_loop_final.log
```

Expected: exit 0. Note the total test count in the log for the PR description.

- [ ] **Step 2: Review the diff against the spec's stated scope**

```bash
git diff main --stat
```

Expected: only `schema_proposal_agent/agent.py` (production) plus the two new test files, plus the
earlier `CLAUDE.md` doc commit and the spec/plan doc commits already on this branch. If anything
else shows up, stop and check it against `docs/superpowers/specs/2026-07-29-schema-refinement-loop-latency-design.md`'s
*Scope* section before proceeding — this plan's Global Constraints list what must NOT be touched.

- [ ] **Step 3: Draft PR notes (for the human to use when opening the PR)**

Write down, for use in the PR description, these two call-outs from the spec that a reviewer would
otherwise have to rediscover themselves:
- The existing commented-out `# before_agent_callback=initialize_schema_and_construction_plan` on
  `refinement_loop` predates this change, is unrelated to either fix, and was deliberately left
  untouched.
- Batching verification tool calls, testing `reasoning_effort="minimal"`, and moving the critic to
  a cheaper model were all investigated during design and explicitly rejected/deferred — see the
  spec's *Why this exists* and *Scope* sections for the reasoning, so a reviewer doesn't re-propose
  them as obvious follow-ups without that context.

No commit for this step — it's a note for whoever opens the PR, not a file change.
