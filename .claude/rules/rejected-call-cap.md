---
paths:
  - "src/agentic_kg/common/rejected_call_cap.py"
  - "src/agentic_kg/common/agent_guards.py"
  - "src/agentic_kg/common/adk_transfer.py"
  - "src/agentic_kg/common/tool_result.py"
  - "src/agentic_kg/tools/**"
  - "src/agentic_kg/coordinators/**"
  - "src/agentic_kg/agents/**"
  - "src/agentic_kg/agent.py"
---
# Rejected-call cap

- Every LlmAgent spreads `**agent_guard_callbacks(gated=...)`; never wire its callbacks by hand.
  `tests/unit/test_agent_wiring_guards.py` fails on an agent in any root that lacks them.
- `end_turn_past_cap` stays first among an agent's before-model callbacks: when it answers, ADK skips the
  rest, which keeps a capped critic's stop text out of `record_critic_verdict`.
- Only the hidden-transfer refusal ends a turn on a reply that also spoke to the user; unknown names and
  ADK's rejections only count toward the cap.
- A tool never returns a bare `{"error": ...}`: `is_adk_rejection` reads that shape as ADK rejecting the
  call. Return `tool_error(...)`.
- A composite that runs agents in steps reads `was_stopped` rather than trusting a stopped step's text, as
  `CheckStatusAndEscalate` does.
- The cap relies on ADK internals (the bare `BaseTool` stand-in for an unknown name, the rejection's shape,
  a before-model answer skipping the after-model callbacks); after a `google-adk` bump, read
  `tests/unit/test_rejected_call_cap_flow.py`'s results first.
