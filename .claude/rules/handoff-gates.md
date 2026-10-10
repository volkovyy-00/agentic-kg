---
paths:
  - "src/agentic_kg/coordinators/multi_agent/**"
  - "src/agentic_kg/common/adk_transfer.py"
  - "src/agentic_kg/common/agent_guards.py"
  - "src/agentic_kg/common/adk_context.py"
  - "src/agentic_kg/tools/adk_tools.py"
  - "src/agentic_kg/tools/*_handoff_tools.py"
  - "src/agentic_kg/tools/graphrag_partition_tools.py"
  - "src/agentic_kg/tools/user_goal_tools.py"
---
# Handoff gates and the transfer guard

- Gates come in two shapes: the turn-scoped flag/reset/confirm shape (construction exit, retrieval
  exit, the partition tool gate) and the durable-state equality check (intent exit). Never factor the
  intent gate into the flag shape.
- A new flag/reset/confirm gate takes its key and plumbing from `TurnFlag`; its confirm tool, gated tool,
  reset function, docstrings and refusal text stay hand-written for that gate.
- A gated agent takes every transfer-related callback from `**agent_guard_callbacks(gated=...)`;
  if it needs its own callback of one of those kinds, extend the helper rather than wiring lists.
- Leave `disallow_transfer_to_parent` and `disallow_transfer_to_peers` unset; a `make_finished`
  target must be the agent's parent or a peer.
- Write each gated `finished` docstring for its own handoff; never share one between gates.
- Refuse a call or end a gated turn with a reply, never by raising; set `skip_summarization` only on
  a reply that also spoke to the user.
