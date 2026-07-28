from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext

from google.adk.tools import agent_tool

from typing import AsyncGenerator
from google.adk.events import Event, EventActions
from google.genai import types

from agentic_kg.common.llm_catalog import get_llm, LlmKind
from agentic_kg.tools.construction_plan_tools import (
    get_proposed_construction_plan, 
    approve_proposed_construction_plan,
)
from agentic_kg.tools.adk_tools import make_finished
from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR

finished = make_finished(MULTI_AGENT_COORDINATOR)

from .variants import variants

# initialize context for schema_proposal_agent with blank feedback, which may get filled later by the schema_critic_agent
def initialize_feedback(callback_context: CallbackContext) -> None:
    callback_context.state["feedback"] = ""

def initialize_schema_and_construction_plan(callback_context: CallbackContext) -> None:
    callback_context.state["proposed_schema"] = ""
    callback_context.state["proposed_construction_plan"] = []

AGENT_NAME = "schema_proposal_agent_v1"
schema_proposal_agent = LlmAgent(
    name=AGENT_NAME,
    description="Proposes a knowledge graph schema based on the user goal and approved file list",
    model=get_llm(LlmKind.reasoning),
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"], 
    before_agent_callback=initialize_feedback
)
    
CRITIC_NAME = "schema_critic_agent_v1"
schema_critic_agent = LlmAgent(
    name=CRITIC_NAME,
    description="Criticizes the proposed construction plan for relevance and correctness.",
    model=get_llm(LlmKind.reasoning),
    instruction=variants[CRITIC_NAME]["instruction"],
    tools=variants[CRITIC_NAME]["tools"], 
    output_key="feedback"
)

class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        feedback = ctx.session.state.get("feedback", "valid")
        # Only the leading *word* decides the route: the critic may append a
        # "Warnings:" section to a 'valid' verdict for data-quality issues that
        # no schema change can fix, and that must not restart the loop. A prefix
        # match is not enough — "Validation failed: ..." and "Valid identifiers
        # are missing on Part" both start with "valid" but mean retry, so the
        # first whitespace-delimited token is compared exactly (minus trailing
        # punctuation).
        text = str(feedback).strip()
        first_token = text.split(maxsplit=1)[0].strip(":.,").lower() if text else ""
        should_stop = first_token == "valid"
        # This event's text is what the coordinator's AgentTool call returns:
        # AgentTool takes the text of the *last* event of the wrapped agent's
        # run, and StopChecker always runs last in the loop. When this event
        # carried no content the tool returned "", which coordinator models
        # read as "the tool failed / returned no results" and invented
        # fallbacks instead of reading the plan. Always surface the critic's
        # verdict here so the tool result is never empty.
        # The coordinator routes on whether this result *begins* with 'retry',
        # so the verdict has to come first: any preamble would make that test
        # false for every result the loop can return. An absent verdict means
        # the critic did not answer, not that it found problems, so point the
        # coordinator at the plan rather than at another blind re-run.
        summary = text if text else (
            "retry: the critic produced no verdict. Call "
            "'get_proposed_construction_plan' and judge the plan yourself rather "
            "than running the loop again on no feedback."
        )
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=summary)]),
            actions=EventActions(escalate=should_stop),
        )

refinement_loop = LoopAgent(
    name="schema_refinement_loop",
    description="Analyzes approved files to propose a graph construction plan based on user intent and feedback",
    max_iterations=2,
    sub_agents=[schema_proposal_agent, schema_critic_agent, CheckStatusAndEscalate(name="StopChecker")],
    # before_agent_callback=initialize_schema_and_construction_plan
)

root_agent = LlmAgent(
    name="schema_proposal_agent_coordinator",
    model=get_llm(LlmKind.reasoning),
    instruction="""
    You are a coordinator for the graph construction plan process. Use tools to propose a schema to the user.
    If the user disapproves, use the tools to refine the schema and ask the user to approve again.
    When the schema approval has been recorded, use the 'finished' tool.

    You cannot change the construction plan yourself. Only the 'schema_refinement_loop' tool can, and the
    only way to learn what the plan currently says is the 'get_proposed_construction_plan' tool. What you
    remember from an earlier turn is not evidence: the refinement loop can change entries you did not ask
    about, so a plan you described one turn ago may no longer be what is stored.

    Rules for presenting a plan — these prevent the user approving a schema that will never be built:
    - Never describe a plan, a change to a plan, or a "revised"/"updated" schema from memory or from your
      own reasoning about what should have happened.
    - Immediately before every message in which you show the user a schema, call
      'get_proposed_construction_plan'. Present that call's actual returned data, in that same turn, and
      reproduce each construction's fields exactly as returned — source_file, label or relationship_type,
      unique_column_name, from/to node labels and columns, and properties. If your description and the
      tool result differ on any field, the tool result is correct and yours is wrong.
    - After calling 'schema_refinement_loop', do not assume the requested change was made, and do not
      report it as made. The loop returns only the critic's final verdict ('valid' or 'retry' plus
      feedback), never the plan itself and never raw tool output such as column statistics — a verdict
      is not evidence of what the plan says. Call 'get_proposed_construction_plan' and compare
      the result against what the user asked for. If the change is missing, or if something the user
      previously approved has changed back or otherwise drifted, say so plainly and run the loop again
      with feedback naming both the requested change and the regression — do not present the plan as if
      it were correct.
    - If the verdict the loop returns begins with 'retry', the critic found problems that are still in
      the plan: call 'schema_refinement_loop' again, passing that retry feedback, instead of presenting a
      plan with known problems for approval. Do this at most once for a given problem. If the loop
      returns 'retry' a second time, stop calling it: some objections cannot be fixed by changing the
      schema, because they are properties of the data. Call 'get_proposed_construction_plan', show the
      user that plan together with the critic's remaining objections, and let them decide whether to
      approve it as it stands.

    Guidance for tool use:
    - Use the 'schema_refinement_loop' tool to produce or update a construction plan.
    - Use the 'get_proposed_construction_plan' tool to read the current construction rules for
      transforming approved files into the schema
    - Present the proposed construction plan to the user for approval, following the rules above
    - If they disapprove, pass their feedback to the 'schema_refinement_loop' tool and go back to step 1
    - If the user approves, use the 'approve_proposed_construction_plan' tool to record the approval.
      This tool refuses plans whose relationships join on columns their nodes do not carry, or whose
      relationships reference a node label that has no node construction in the plan. If it returns
      an error, the schema is NOT approved: report the error to the user verbatim, run
      'schema_refinement_loop' with that error as feedback, and present the corrected plan for approval
      again. Never tell the user the schema is approved unless that tool returned success.
    - Finally, use the 'finished' tool to signal that schema proposal is complete and construction can begin
    """,
    tools=[agent_tool.AgentTool(refinement_loop), 
        get_proposed_construction_plan, 
        approve_proposed_construction_plan,
        finished
    ]
)