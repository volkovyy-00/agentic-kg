from agentic_kg.tools.adk_tools import make_finished


class FakeActions:
    def __init__(self):
        self.escalate = False
        self.transfer_to_agent = None


class FakeToolContext:
    def __init__(self):
        self.actions = FakeActions()


def test_finished_transfers_to_the_bound_parent():
    finished = make_finished("kg_construction_agent_v1")
    context = FakeToolContext()
    finished(context)
    assert context.actions.transfer_to_agent == "kg_construction_agent_v1"


def test_finished_sets_escalate():
    finished = make_finished("anything")
    context = FakeToolContext()
    finished(context)
    assert context.actions.escalate is True


def test_finished_takes_no_arguments_beyond_context():
    """A zero-argument tool is more reliable than one requiring the model to
    reproduce an agent name, which is why this is not ADK's transfer_to_agent."""
    import inspect
    finished = make_finished("x")
    parameters = list(inspect.signature(finished).parameters)
    assert parameters == ["tool_context"]


def test_tool_is_still_named_finished():
    assert make_finished("x").__name__ == "finished"


def test_no_private_attribute_access():
    """The old implementation reached into tool_context._invocation_context."""
    import inspect
    source = inspect.getsource(make_finished)
    assert "_invocation_context" not in source
