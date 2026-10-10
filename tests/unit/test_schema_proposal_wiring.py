"""Wiring pins for the schema proposal/critic pair.

Prompt text is not otherwise covered by anything: these assert the two facts
whose absence would silently disable the feature -- the evidence tool not being
reachable, and the revision paragraph not naming property_types.
"""

import asyncio
import inspect
import re
from types import SimpleNamespace

import pytest
from google.adk.utils.instructions_utils import inject_session_state

from agentic_kg.common.value_types import ALLOWED_TYPES
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent import (
    agent as agent_module,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent import (
    variants as variants_module,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    CheckStatusAndEscalate,
    prepare_refinement_loop_invocation,
    root_agent,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.variants import (
    variants,
)
from agentic_kg.tools import adk_tools, construction_plan_tools
from agentic_kg.tools.construction_plan_tools import (
    propose_node_construction,
    propose_relationship_construction,
)
from agentic_kg.tools.file_tools import column_type_hint, column_type_hints


def test_both_agents_can_call_the_type_hint_tool():
    """The proposal agent needs it to declare types; the critic needs it to
    challenge one. A critic without it can only object from the column name."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        assert column_type_hint in variants[name]["tools"], name


def test_the_proposal_agent_can_batch_type_hints():
    assert column_type_hints in variants["schema_proposal_agent_v1"]["tools"]


@pytest.mark.parametrize(
    "agent", ("schema_proposal_agent_v1", "schema_critic_agent_v1")
)
def test_every_tool_an_instruction_names_is_a_tool_that_agent_has(agent):
    """An instruction advertising a tool the agent was not given fails quietly:
    the model follows the advertised path, the name is not in tools_dict, and
    google-adk 2.10 answers with its not-found reply, which invites a retry -- a
    wasted model call every time the model follows the advertisement, not a
    loud error at startup. (That reply is build_tool_not_found_response; the
    refuse_transfer_to_agent docstring in common/adk_transfer.py describes it.)

    It happened here: _VALIDATION_RULES is shared text embedded in BOTH agents
    and offers 'column_type_hints', while only the proposal agent held it. Shared
    prompt text is exactly where this hides, because the tool lists are not.

    Only names that are real tools somewhere in the module are checked, so
    ordinary quoted words in the prompts ('retry', 'valid') are not mistaken for
    tool references.
    """
    instruction = variants[agent]["instruction"]
    wired = {tool.__name__ for tool in variants[agent]["tools"]}
    known_tools = _tool_names_in(vars(variants_module))

    named = {
        match
        for match in re.findall(r"'([a-z_][a-z0-9_]*)'", instruction)
        if match in known_tools
    }
    missing = named - wired
    assert not missing, (
        f"{agent}'s instruction names {sorted(missing)}, which it cannot call. "
        f"Either wire the tool in or stop advertising it."
    )


def test_the_revision_paragraph_names_property_types():
    """A propose call replaces the whole entry. A re-proposal that restates
    properties but forgets property_types silently reverts every declared type to
    text, and the plan looks identical -- the critic sees only the current
    snapshot and cannot detect it."""
    instruction = variants["schema_proposal_agent_v1"]["instruction"]
    assert "property_types" in instruction
    assert "restate every field" in instruction


def test_both_instructions_carry_the_property_type_rules():
    """The rules block is shared so the two cannot drift; if the subsection were
    added to only one, the critic could reject plans built to a different rule."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        assert "column_type_hint" in variants[name]["instruction"], name


@pytest.mark.parametrize(
    "fn", [propose_node_construction, propose_relationship_construction]
)
def test_proposed_property_types_stays_optional_in_the_declaration(fn):
    """Pins TRAP 6: proposed_property_types must stay an OPTIONAL parameter in
    the FunctionDeclaration the model sees, or every propose call without it
    would be rejected.

    History: google-adk 1.10.0 validated that a parameter's default is an
    instance of its annotation, so 'dict = None' raised ValueError at toolset
    construction and took down the entire schema-proposal phase. Later releases
    stopped raising, and google-adk 2.x declares tools as a JSON schema
    (parameters_json_schema; declaration.parameters is None), where the field is
    anyOf[object, null] with default null and not required. The test guards that
    declaration contract, not a crash. Keep the 'Optional[dict] = None'
    annotation: it is the honest type. Building the declaration here reproduces
    that construction step as a unit test."""
    from google.adk.tools.function_tool import FunctionTool

    schema = FunctionTool(fn)._get_declaration().parameters_json_schema or {}
    props = schema.get("properties", {})
    required = schema.get("required", [])

    assert "proposed_property_types" in props
    assert "proposed_property_types" not in required


def test_the_matched_properties_stay_optional_in_the_declaration():
    """Same contract as proposed_property_types: a propose call that leaves the
    new fields out must not be rejected by the declaration."""
    from google.adk.tools.function_tool import FunctionTool

    schema = (
        FunctionTool(propose_relationship_construction)
        ._get_declaration()
        .parameters_json_schema
        or {}
    )
    for name in ("from_node_property", "to_node_property"):
        assert name in schema.get("properties", {}), name
        assert name not in schema.get("required", []), name


def names_type(text: str, name: str) -> bool:
    """True when `name` appears as a whole quoted word ('date' or "date").

    A bare substring test passes for 'date' when only 'datetime' is written, so
    a prompt that forgot the plain date type would still look complete.
    """
    return re.search(rf"""(["']){re.escape(name)}\1""", text) is not None


def test_the_type_name_match_does_not_take_date_from_datetime():
    assert not names_type("one of 'datetime' or 'localdatetime'", "date")
    assert names_type('"integer", "date", "datetime"', "date")
    assert names_type("'date', 'datetime'", "datetime")


@pytest.mark.parametrize(
    "fn", [propose_node_construction, propose_relationship_construction]
)
def test_every_allowed_type_is_named_in_the_tool_description(fn):
    """The closed set lives in value_types.ALLOWED_TYPES, but the model only
    ever learns it from prose -- these docstrings are the tool descriptions ADK
    sends. Any new entry in ALLOWED_TYPES must be named, as a whole quoted
    word, in both descriptions, or the model is never told it may propose it."""
    for allowed in ALLOWED_TYPES:
        assert names_type(fn.__doc__, allowed), allowed


@pytest.mark.parametrize(
    "fn", [propose_node_construction, propose_relationship_construction]
)
def test_the_name_character_rule_is_stated_in_the_tool_description(fn):
    """The propose tools refuse a label or relationship type that is not a plain
    identifier (KG-44), and accept any header of the file as a key, join column
    or matched property (KG-51). Unless the description states both rules, the
    model learns them only from a refusal and spends turns retrying variants such
    as 'Order-ID', or renames a column the file spells 'Order ID'. It must also
    know that keywords are fine, so it does not rename a correct 'Order'."""
    text = " ".join(fn.__doc__.split())
    assert "a letter or underscore followed by letters, digits or underscores" in text
    assert "Cypher keywords" in text
    assert "do not rename or reformat it" in text


def test_the_key_rule_is_stated_in_the_node_tool_description():
    """The propose tool refuses a key with a blank value, and a repeating key whose
    rows disagree on a listed property (KG-48). The description is where the model
    learns that before it proposes, rather than from the refusal alone. The rule
    sits above Args:, whose lines test_the_args_section_names_exactly_the_real_
    parameters reads as parameter names."""
    text = " ".join(propose_node_construction.__doc__.split())
    assert "must have a value in every row" in text
    assert "agree on every property you list for the node" in text
    assert "leave those properties off the node" in text
    assert text.index("must have a value in every row") < text.index("Args:")


def test_every_allowed_type_is_named_in_the_validation_rules():
    """Same staleness, one layer up: the shared rules block tells both agents
    which types exist, so a new entry in ALLOWED_TYPES that never reaches this
    prompt is a type the proposal agent will not use and the critic will
    reject."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        instruction = variants[name]["instruction"]
        for allowed in ALLOWED_TYPES:
            assert names_type(instruction, allowed), f"{name}: {allowed}"


@pytest.mark.parametrize(
    "fn", [propose_node_construction, propose_relationship_construction]
)
def test_the_args_section_names_exactly_the_real_parameters(fn):
    """These docstrings are the tool descriptions ADK sends to the model, so an
    Args entry for a parameter that does not exist is an instruction to fill in
    a field the tool cannot accept -- and a real parameter left undocumented is
    one the model has no guidance for.

    propose_relationship_construction documented 'unique_column_name', which it
    has never had, while omitting 'proposed_properties', which it requires. The
    invented one mattered most here: the branch's whole safety rule is that join
    columns stay text, and naming a node-only field on the relationship tool
    points the model at the wrong columns to protect.
    """
    args_block = fn.__doc__.split("Args:")[1].split("Returns:")[0]
    documented = set(re.findall(r"^\s+(\w+):", args_block, re.MULTILINE))
    actual = set(inspect.signature(fn).parameters) - {"tool_context"}

    assert documented - actual == set(), (
        f"documented but not parameters: {documented - actual}"
    )
    assert actual - documented == set(), (
        f"parameters but not documented: {actual - documented}"
    )


class _FakeCallbackContext:
    def __init__(self):
        self.state = {}


class _FakeSession:
    def __init__(self, state):
        self.state = state


class _FakeInvocationContext:
    def __init__(self, state):
        self.session = _FakeSession(state)


def _stopped_verdict_text():
    """The text prepare_refinement_loop_invocation returns when the loop is
    called a second time in one turn. First call returns None and arms the
    counter; the second short-circuits."""
    ctx = _FakeCallbackContext()
    assert prepare_refinement_loop_invocation(ctx) is None
    content = prepare_refinement_loop_invocation(ctx)
    return content.parts[0].text


def _empty_verdict_text():
    """The text CheckStatusAndEscalate yields when the critic returned no
    verdict at all."""
    checker = CheckStatusAndEscalate(name="StopChecker")
    ctx = _FakeInvocationContext({"feedback": ""})

    async def collect():
        return [event async for event in checker._run_async_impl(ctx)]

    events = asyncio.run(collect())
    return events[-1].content.parts[0].text


def _plan_problem_composite_text():
    """The text CheckStatusAndEscalate yields when a mechanical plan check
    found problems. Driven through the real stop-check, like the helpers above,
    so a change in how the composite is produced is caught here too."""
    checker = CheckStatusAndEscalate(name="StopChecker")
    ctx = _FakeInvocationContext(
        {
            "feedback": "valid",
            "proposed_construction_plan": {
                "REFERS_TO": {
                    "construction_type": "relationship",
                    "relationship_type": "REFERS_TO",
                    "source_file": "a.csv",
                    "from_node_label": "Missing",
                    "from_node_column": "a_id",
                    "to_node_label": "AlsoMissing",
                    "to_node_column": "b_id",
                }
            },
            "approved_file_list": [],
        }
    )

    async def collect():
        return [event async for event in checker._run_async_impl(ctx)]

    events = asyncio.run(collect())
    return events[-1].content.parts[0].text


def _plan(join_column):
    """A minimal two-node plan whose relationship joins on `join_column`.
    'assembly_name' is not a column Assembly carries, so that spelling drives
    every refusal branch here; 'assembly_id' is, so it drives the success
    branch."""
    return {
        "Product": {
            "construction_type": "node",
            "source_file": "products.csv",
            "label": "Product",
            "unique_column_name": "product_id",
            "properties": ["product_name"],
        },
        "Assembly": {
            "construction_type": "node",
            "source_file": "assemblies.csv",
            "label": "Assembly",
            "unique_column_name": "assembly_id",
            "properties": ["component_name", "quantity", "product_id"],
        },
        "ASSEMBLY_OF": {
            "construction_type": "relationship",
            "source_file": "assemblies.csv",
            "relationship_type": "ASSEMBLY_OF",
            "from_node_label": "Assembly",
            "from_node_column": join_column,
            "to_node_label": "Product",
            "to_node_column": "product_id",
            "properties": ["quantity"],
        },
    }


def _refusal_message_text():
    """The text approve_proposed_construction_plan returns when the plan is
    internally inconsistent. approve_proposed_construction_plan is held by
    exactly one agent -- this coordinator -- so this refusal string lands in
    the coordinator's context and, if it names a tool the coordinator does not
    hold, is the same dead-turn failure this file's other checks exist to
    prevent."""
    ctx = _FakeCallbackContext()
    ctx.state[construction_plan_tools.PROPOSED_CONSTRUCTION_PLAN] = _plan(
        "assembly_name"
    )
    result = construction_plan_tools.approve_proposed_construction_plan(ctx)
    return result["error_message"]


def _approval_check_texts():
    """Both messages the coordinator's own plan-reading tool can return. They
    name 'approve_proposed_construction_plan' and land in the coordinator's
    context exactly as the refusal above does, so the same rename that would
    break the strings in agent.py would break these -- and this file's stated
    job is catching precisely that."""
    read = construction_plan_tools.get_proposed_construction_plan_with_approval_check
    ctx = _FakeCallbackContext()

    ctx.state[construction_plan_tools.PROPOSED_CONSTRUCTION_PLAN] = _plan(
        "assembly_name"
    )
    blocked = read(ctx)

    ctx.state[construction_plan_tools.PROPOSED_CONSTRUCTION_PLAN] = _plan("assembly_id")
    allowed = read(ctx)

    assert blocked["status"] == "error"
    assert allowed["status"] == "success"
    return [blocked["error_message"], allowed["result"]["message"]]


def _tool_names_in(namespace):
    return {
        name
        for name, value in namespace.items()
        if callable(value)
        and not isinstance(value, type)
        and getattr(value, "__module__", "").startswith("agentic_kg.tools.")
        and not name.startswith("_")
    }


def test_every_tool_the_coordinator_names_is_a_tool_the_coordinator_has():
    """The coordinator's instruction and tools are inline attributes on
    root_agent, a shape test_every_tool_an_instruction_names_is_a_tool_that_agent_has
    cannot reach -- it only scans the variants dict. So a rename that updates
    the instruction but forgets one of the two generated tool-result strings
    ships silently, and the coordinator then receives tool output naming a tool
    that is not in tools_dict: google-adk 2.10 answers a call to it with a
    not-found error, a wasted model call each time the coordinator follows it.

    Two details are load-bearing, and getting either wrong makes this test pass
    on the failure it exists to catch:

    The pattern matches UNQUOTED names. The generated string in
    prepare_refinement_loop_invocation says "call
    get_proposed_construction_plan_with_approval_check and present" with no
    quotes, so the quoted-only pattern used by the sibling test above sees one
    of the two sites and misses the other.

    known_tools comes from the tools modules as well as this agent's namespace.
    Sourcing it from vars(agent_module) alone -- as the sibling test does, which
    is safe there only because variants.py imports every name it uses -- makes
    the check blind to a RENAME: once the old name is no longer imported here, a
    forgotten mention of it is not recognised as a tool name at all.
    """
    texts = [
        root_agent.instruction,
        _stopped_verdict_text(),
        _empty_verdict_text(),
        _plan_problem_composite_text(),
        _refusal_message_text(),
        *_approval_check_texts(),
    ]
    wired = {getattr(tool, "name", None) or tool.__name__ for tool in root_agent.tools}
    known_tools = (
        _tool_names_in(vars(agent_module))
        | _tool_names_in(vars(construction_plan_tools))
        | _tool_names_in(vars(adk_tools))
    )

    named = {
        match
        for text in texts
        for match in re.findall(r"[a-z_][a-z0-9_]*", text)
        if match in known_tools
    }
    missing = named - wired
    assert not missing, (
        f"the coordinator names {sorted(missing)}, which it cannot call. "
        f"Either wire the tool in or stop advertising it."
    )


def _render_proposal_instruction(state):
    """Render the proposal instruction the way ADK does, against a bare state."""
    ctx = SimpleNamespace(
        _invocation_context=SimpleNamespace(
            session=SimpleNamespace(state=state), artifact_service=None
        )
    )
    template = variants["schema_proposal_agent_v1"]["instruction"]
    return asyncio.run(inject_session_state(template, ctx))


def test_the_proposal_instruction_renders_without_a_kind():
    """The placeholder is optional: a state with no 'feedback_kind' yet must
    render, not raise KeyError and kill the turn."""
    rendered = _render_proposal_instruction({"feedback": ""})
    assert "Kind of feedback: \n" in rendered


def test_the_proposal_instruction_renders_the_kind():
    rendered = _render_proposal_instruction(
        {"feedback": "x", "feedback_kind": "critic"}
    )
    assert "Kind of feedback: critic" in rendered


def test_the_kind_gloss_carries_no_approval_framing():
    """PR #20: the proposal step's context holds no approval or readiness
    framing. Scoped to the lines this ticket added."""
    template = variants["schema_proposal_agent_v1"]["instruction"]
    start = template.index("Kind of feedback:")
    gloss = template[start : template.index("no feedback this round", start)]
    assert "approv" not in gloss.lower()
    assert "ready" not in gloss.lower()


def _instruction_words():
    """root_agent's instruction with whitespace collapsed, so a phrase may
    wrap across source lines."""
    return " ".join(root_agent.instruction.split())


@pytest.mark.parametrize(
    "phrase",
    [
        "the critic's final verdict",
        "the critic found problems that are still in the plan",
        "together with the critic's remaining objections",
    ],
)
def test_the_coordinator_no_longer_calls_every_verdict_the_critics(phrase):
    """KG-30: the three sentences that attributed a retry to the critic
    whichever writer filled the slot."""
    assert phrase not in _instruction_words()


def test_the_coordinator_splits_a_second_retry_on_the_read_tool():
    """KG-30 AC1/AC4. The coordinator never sees session state, so its signal
    is the read tool's status: no plan, mechanical problems, or success. The
    no-plan branch uses the read tool's own wording."""
    words = _instruction_words()
    no_plan = "error saying there is no proposed construction plan"
    mechanical = "cannot be approved as it stands"
    critic = "let the user decide whether to approve it as it stands"
    assert no_plan in words
    assert mechanical in words
    assert critic in words
    assert "no proposed construction plan" in (
        construction_plan_tools.NO_PROPOSED_PLAN_MESSAGE.lower()
    )
    # The mechanical branch comes first, so it is read before the critic one.
    assert words.index(mechanical) < words.index("properties of the data")


def test_the_coordinator_reads_the_stopped_kind():
    words = _instruction_words()
    assert "calls its verdict a mechanical check finding" in words


# --- KG-45 ---------------------------------------------------------------------

BOTH = ("schema_proposal_agent_v1", "schema_critic_agent_v1")


def _flat(name):
    """The instruction with whitespace collapsed, so a pinned phrase may wrap."""
    return " ".join(variants[name]["instruction"].split())


@pytest.mark.parametrize("name", BOTH)
def test_both_instructions_explain_the_matched_property(name):
    instruction = _flat(name)
    assert "'from_node_property'" in instruction
    assert "'to_node_property'" in instruction
    assert "never the relationship file's column" in instruction, "collapse_check"
    assert "all three together" in instruction, "a swap moves the properties too"


def test_the_revision_paragraph_names_the_matched_properties():
    instruction = _flat("schema_proposal_agent_v1")
    assert "a re-proposal that omits them" in instruction
    assert "set it to the new identifier" in instruction


def test_the_proposer_says_how_to_write_a_reference_under_another_name():
    instruction = _flat("schema_proposal_agent_v1")
    assert "holds another node's key under a different name" in instruction


def test_the_proposers_steps_keep_a_matched_property_untyped():
    """The numbered step the model follows says it, not only the rules block."""
    instruction = _flat("schema_proposal_agent_v1")
    step = instruction[instruction.index("8. Never declare a type") :]
    step = step[: step.index("9. ")]
    assert "any relationship end is matched on" in step
    assert "joins on or is matched on a typed property" in step


def test_the_critic_rejects_a_reference_matched_under_its_own_name():
    instruction = _flat("schema_critic_agent_v1")
    assert "links nodes sharing the value" in instruction


@pytest.mark.parametrize("name", BOTH)
@pytest.mark.parametrize("word", ["employee", "manager", "reportsto", "northwind"])
def test_no_instruction_borrows_one_datasets_vocabulary(name, word):
    assert word not in variants[name]["instruction"].lower()


def test_the_critic_instruction_renders_against_a_bare_state():
    """A literal {word} would be read as a state key and raise KeyError,
    killing the critic's turn. The critic has no placeholders, so rendering
    must hand back the template unchanged."""
    ctx = SimpleNamespace(
        _invocation_context=SimpleNamespace(
            session=SimpleNamespace(state={}), artifact_service=None
        )
    )
    template = variants["schema_critic_agent_v1"]["instruction"]
    assert asyncio.run(inject_session_state(template, ctx)) == template


def test_the_allowed_types_are_the_six_named_ones():
    assert ALLOWED_TYPES == (
        "integer",
        "float",
        "boolean",
        "date",
        "datetime",
        "localdatetime",
    )


def test_the_rules_name_the_hint_shape_for_each_temporal_type():
    """The proposer must declare the kind the hint reports and never choose
    between zoned and local from a column name."""
    for name in ("schema_proposal_agent_v1", "schema_critic_agent_v1"):
        instruction = variants[name]["instruction"]
        for shape in ("date_like", "datetime_like", "localdatetime_like"):
            assert f"'{shape}'" in instruction, f"{name}: {shape}"
        assert "never one chosen from the column name" in " ".join(
            instruction.split()
        ), name


def test_the_critic_checks_date_columns_but_never_for_keys_and_joins():
    """A key, join column or matched property must stay text: approval refuses a
    type on one. Without the exception a date column used as a join would send
    the critic and the proposer in a loop."""
    text = " ".join(variants["schema_critic_agent_v1"]["instruction"].split())
    assert (
        "Is a column that looks like a date or timestamp left without a declared "
        "type?" in text
    )
    assert "unless the column is a node's unique identifier" in text
    assert "those must stay text" in text


def test_the_critic_never_pushes_a_temporal_type_that_would_clear_real_values():
    """A 60% date / 40% zoned column is suggested 'date', and declaring it clears
    every timestamp with only a warning. The proposer is told to read
    'example_unconvertible' first; the critic must not then overrule it."""
    text = " ".join(variants["schema_critic_agent_v1"]["instruction"].split())
    assert "'localdatetime' with an 'unconvertible_count' of 0" in text
    assert "leaving that column as text is the proposer's call" in text
