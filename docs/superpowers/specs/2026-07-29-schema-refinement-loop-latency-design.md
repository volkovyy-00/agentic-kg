# Schema refinement loop: fixing a feedback bug and capping worst-case latency

**Date:** 2026-07-29
**Status:** Approved (design reviewed twice against ADK source before write-up — see *Verification*)
**Branch:** `reduce-schema-refinement-latency`

## Why this exists

Live smoke-testing on 2026-07-29 (sessions `ce85bbb3` and `60f846dc`) showed
`schema_refinement_loop` — the `LoopAgent` wrapping `schema_proposal_agent_v1` →
`schema_critic_agent_v1` inside `schema_proposal_agent_coordinator` — taking 450-620s (7.5-10.3
min) per invocation, and in one turn the coordinator chained two such invocations back-to-back
(no user-visible response in between) for roughly 990s combined. The user is demoing this system
to an interviewer this Friday and needs live per-turn latency to be predictable and short enough
not to look broken.

Two independent problems were found and confirmed against ADK's actual source
(`google/adk/agents/base_agent.py`, `loop_agent.py`, `agent_tool.py` in this project's venv), not
assumed from behavior alone:

1. **A real bug, not just slowness.** `schema_proposal_agent_v1`'s instruction embeds
   `<feedback>{feedback}</feedback>` (`schema_proposal_agent/variants.py:123-126`), populated from
   `schema_critic_agent_v1`'s `output_key="feedback"`. But `initialize_feedback` — the callback
   that seeds `state["feedback"] = ""` — is attached as `schema_proposal_agent`'s own
   `before_agent_callback` (`schema_proposal_agent/agent.py:38`). `BaseAgent.run_async`
   (`base_agent.py:218-224`) runs `before_agent_callback` on *every* call to that agent object, and
   `LoopAgent._run_async_impl` calls each sub-agent's `run_async` once per iteration — so on the
   loop's second iteration (the retry round), `schema_proposal_agent_v1` sees `feedback=""`,
   clobbered immediately before its own turn, instead of the critic's actual verdict from iteration
   1. The "retry" round is not incorporating feedback at all; it is blindly re-deriving the schema
   from scratch. This is a likely contributor to round 2 costing nearly as much as round 1 in the
   observed traces (92-230s vs. 230s).
2. **No enforced bound on total rounds.** The `LoopAgent` itself caps at `max_iterations=2`, but
   `schema_proposal_agent_coordinator`'s instructions (`agent.py:130-136`) additionally *permit*
   calling the whole `schema_refinement_loop` tool a second time if the critic's verdict is
   `retry` again — a prose rule only, unenforced in code. Worst case observed: 4 full
   proposal+critique rounds for one user turn.

A third candidate cause — verification tool calls (`column_stats`/`join_preview`/`collapse_check`)
issued serially rather than batched — was investigated and **rejected**: those tools have a real
data dependency (the agent's own instructions, `variants.py:216-218`, mandate
`column_stats` → `collapse_check` → `join_preview` in that order for a given relationship), and the
cheap tool calls being serialized total only ~39s of a 230s round — the expensive reasoning calls
(65% of the time) are untouched by batching them. Two more candidates were checked and found
already handled or too risky: `reasoning_effort` is already `"low"` and confirmed forwarded to
`openai/gpt-5` via OpenRouter (verified against the installed `litellm` package's provider
mapping); moving `schema_critic_agent_v1` to a cheaper model was rejected because it embeds the
identical `_VALIDATION_RULES` block the proposer uses (`variants.py:30-113`, shared between both),
so a weaker critic risks more false `retry` verdicts, each costing a full extra round — net slower,
not faster.

## Scope

**In**
- Fix the feedback-clobbering bug: move the reset from `schema_proposal_agent`'s
  `before_agent_callback` to `refinement_loop`'s (the `LoopAgent` itself).
- Enforce, in code, that the coordinator can trigger at most one `schema_refinement_loop`
  invocation per user turn (i.e. at most `max_iterations=2` rounds total, never 3 or 4).
- Tests demonstrating both fixes, including one empirical (Runner-driven, two-turn) test closing
  the one assumption source-reading alone couldn't settle (see *Verification*).

**Out (deferred, not part of this change)**
- Batching `column_stats`/`join_preview`/`collapse_check` — rejected above, low yield, real
  ordering dependency.
- Testing `reasoning_effort="minimal"` — plausible, unverified against OpenRouter's actual
  behavior for `gpt-5`; a follow-up experiment, not a code change bundled with a bug fix.
- Moving `schema_critic_agent_v1` to a cheaper model — rejected above.
- Investigating whether `max_tokens=8192` truncates a `gpt-5` response (`finish_reason="length"`)
  in a way indistinguishable from a slow-but-complete call — a real open question, but a separate
  investigation with its own evidence-gathering, not bundled here.
- The existing commented-out `# before_agent_callback=initialize_schema_and_construction_plan` on
  `refinement_loop` (`agent.py:98`) is left untouched. It predates this change and is unrelated to
  either fix; call it out in the PR description so a reviewer doesn't wonder whether this change
  was meant to also enable it.

## Design

### Fix 1 — move the feedback reset to the loop's own callback

Remove `before_agent_callback=initialize_feedback` from `schema_proposal_agent`
(`schema_proposal_agent/agent.py:38`). The reset still needs to happen once per fresh invocation
of the loop (so a brand-new request doesn't inherit a stale prior verdict as if it were live
feedback) — but it must not fire again between iteration 1 and iteration 2 of the *same*
invocation. `LoopAgent`'s own `before_agent_callback` fires exactly once per call to
`refinement_loop.run_async(...)` (i.e. once per coordinator tool-call), which is the correct
scope. `schema_proposal_agent` has exactly one call site — as `refinement_loop`'s first sub-agent
(`agent.py:97`) — so nothing else depends on it resetting itself.

### Fix 2 — hard cap via before_agent_callback short-circuit

ADK's `before_agent_callback` contract (`base_agent.py:392-404`): if the callback returns non-empty
`Content`, ADK sets `ctx.end_invocation = True` and returns a synthetic `Event` built from that
content; `run_async` (`base_agent.py:218-224`) returns immediately after, without ever calling
`_run_async_impl` — sub-agent LLM calls are skipped entirely, not just deprioritized.
`AgentTool.run_async` (`agent_tool.py:143-155`) takes the last event's content text as the tool's
return value to the coordinator, and since `end_invocation=True` stops the run right after that
synthetic event, it *is* the last event — so the short-circuit text reliably becomes what the
coordinator sees.

Both fixes share one callback on `refinement_loop`, in this exact order (increment before check,
so the counter is always correct; reset `feedback` only on the call that actually proceeds, so a
short-circuited call doesn't erase the verdict its own message quotes):

```python
def prepare_refinement_loop_invocation(callback_context: CallbackContext) -> Optional[types.Content]:
    calls = callback_context.state.get("schema_refinement_calls_this_turn", 0) + 1
    callback_context.state["schema_refinement_calls_this_turn"] = calls
    if calls > 1:
        last_feedback = callback_context.state.get("feedback", "")
        return types.Content(
            role="model",
            parts=[types.Part(text=(
                f"stopped: schema_refinement_loop already ran once this turn "
                f"(last verdict: {last_feedback}). Do not call it again this turn — call "
                f"get_proposed_construction_plan and present that plan together with the "
                f"verdict above, and let the user decide."
            ))],
        )
    callback_context.state["feedback"] = ""
    return None
```

`role="model"` is set on the returned `Content` for consistency with `StopChecker`'s synthetic
events two functions away in the same file (`agent.py:83`), which sets it deliberately with a
comment explaining why a short-circuit event needs real content — no functional requirement was
found that mandates it, but there's no reason to diverge from an established local pattern.

The counter (`schema_refinement_calls_this_turn`) is reset to `0` by a `before_agent_callback` on
`schema_proposal_agent_coordinator` (`root_agent`) itself, so each new user turn gets a fresh
one-call budget.

**Routing safety:** the coordinator's instructions (`agent.py:120-133`) only special-case text
*beginning with* `'retry'`; the `stopped:` message doesn't collide, and this design adds one
explicit instruction bullet naming the `stopped:` case rather than relying on implicit fallthrough
— implicit prompt-text fallthrough is exactly the kind of thing that silently breaks when the
instruction block is edited later.

### State keys introduced

- `schema_refinement_calls_this_turn` (int, coordinator-scoped) — new.
- `feedback` (str) — existing key, reset location moves from `schema_proposal_agent` to
  `refinement_loop`; no schema change.

## Testing

1. Unit test: the moved callback preserves `feedback` across two loop iterations (regression test
   for the exact bug found — assert iteration 2's rendered instruction contains the critic's
   iteration-1 verdict, not an empty string).
2. Unit test: a second same-turn invocation of `refinement_loop` short-circuits — assert no
   sub-agent (`schema_proposal_agent_v1`, `schema_critic_agent_v1`) is invoked, via a mock/spy, not
   by measuring elapsed time.
3. Unit test: the `stopped:` message does not begin with `retry`, so it cannot be misrouted by the
   coordinator's existing routing rule.
4. Runner-level test (same style as the existing `StopChecker` / `test_schema_refinement_loop.py`
   tests, which already prefer direct behavioral assertions over trusting docstrings): drive two
   sequential user turns through an actual session and assert turn 1 allows exactly one loop
   invocation, and turn 2 (a fresh message) resets the budget and allows exactly one more. This is
   the one part of the design that source-reading alone couldn't settle — "how many times does ADK
   call `run_async` on the coordinator per user turn" is a framework scheduling behavior, not
   something reading `base_agent.py` in isolation proves — so it gets an empirical test rather than
   a comment citing source.

## Verification

This design was reviewed twice, independently, against the actual installed ADK source (not
against docstrings or assumed behavior) before being written up:

- Confirmed `LoopAgent._run_async_impl` re-invokes each sub-agent's `run_async` (and therefore its
  `before_agent_callback`) once per iteration, while the `LoopAgent`'s own callback fires once per
  outer invocation — the exact mechanism Fix 1 depends on.
- Confirmed the `end_invocation` short-circuit path in `base_agent.py:392-404` and that
  `AgentTool.run_async` surfaces the synthetic event's text as the tool's return value — the exact
  mechanism Fix 2 depends on.
- Confirmed `AgentTool.run_async` forwards `tool_context.state.to_dict()` into the inner session
  and forwards `state_delta` back out (`agent_tool.py:140-148`), so a counter written inside the
  callback persists back into the coordinator's real session across repeated tool calls — not just
  within one isolated `AgentTool` run.
- Grepped every `state[...]` write in `src/agentic_kg` to confirm
  `schema_refinement_calls_this_turn` doesn't collide with any existing key.
- Confirmed `types.Content(parts=[...])` defaults `role=None`, which is why the design explicitly
  sets `role="model"`.

The one gap neither review could close by reading source — whether
`schema_proposal_agent_coordinator`'s `before_agent_callback` fires exactly once per incoming user
message under ADK's agent-transfer model — is deliberately covered by testing item 4 above rather
than asserted.
