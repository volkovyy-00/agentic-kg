---
name: debug-adk-web
description: Use when debugging the running agentic-kg app in the adk web dev UI — a turn that shows no response and no spinner, a red error event or a one-line snackbar in the chat, a hung tool call, a suspected swallowed exception, or reasoning-model calls that stopped working.
---

# Debugging a live `adk web` session

From the browser alone, a hung tool call, a routing bug and a swallowed exception look identical.
Check, cheapest first:

1. **Session events:** `GET /apps/{app}/users/{user}/sessions/{id}`. A frozen event count means
   nothing happened. `{app}` is the directory under `src/agentic_kg/coordinators/` (`multi_agent`),
   while the coordinator's events are authored by its registered name, `kg_construction_agent_v1`.
2. **Trace:** the undocumented `GET /dev/apps/{app}/debug/trace/session/{id}`. Spans carry
   `start_time` / `end_time`. A model call that raised still gets a `call_llm` span, but with no
   attributes at all (ADK sets them only per response), and the span JSON has no status field.
3. **Server log:** the `adk web` process's own output is the only place a swallowed exception surfaces.

An exception that escapes the run is not swallowed: on google-adk 2.10, `/run_sse` sends it to the
browser, which shows a red error event in the chat plus a one-line snackbar.

**Trace shape.** On google-adk 2.10 a `call_llm` span covers only the model call. The tool calls it
asked for (`execute_tool <name>`) and any agent transfer are traced beside it, as siblings under the
agent's `invoke_agent <name>` span, not inside it: look for a hung tool there.

**Reasoning-model calls failing.** A 402 from OpenRouter shows as the red event and snackbar, and the
`call_llm` span has no attributes. Check the OpenRouter balance and the server log before assuming a
code regression; why the output-token cap exists is at `_LLM_MAX_TOKENS` in
`src/agentic_kg/common/llm_catalog.py`.

Never reload the tab while a turn is genuinely streaming.
