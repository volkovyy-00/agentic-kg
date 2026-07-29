from typing import Any, Callable, Dict

from google.adk.tools import ToolContext


def make_finished(parent_agent_name: str) -> Callable[[ToolContext], Dict[str, Any]]:
    """Build a zero-argument 'finished' tool bound to a parent agent's name.

    ADK offers a public transfer_to_agent(agent_name, tool_context), but it
    requires the model to reproduce the target name as an argument. A
    zero-argument tool is categorically more reliable, especially on smaller
    models. Binding the name at construction avoids both the argument and the
    private-attribute lookup the previous implementation used.
    """

    def finished(tool_context: ToolContext) -> Dict[str, Any]:
        """Finish the current phase and hand control back to the coordinator."""
        # escalate is currently inert: ADK only reads it for control flow
        # inside a LoopAgent, and no caller of make_finished runs inside one.
        # Kept deliberately so a future phase added inside a loop behaves
        # correctly without anyone having to remember to add it back.
        tool_context.actions.escalate = True
        tool_context.actions.transfer_to_agent = parent_agent_name
        return {}

    return finished
