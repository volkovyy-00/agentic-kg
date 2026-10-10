"""The result shape a new ADK tool in this package returns.

A tool returns tool_success(key, value) -- {"status": "success", key: value}
-- or tool_error(message) -- {"status": "error", "error_message": message}.
Callers read a result through is_success, is_error, get_or_else, get_or_raise,
map_result and map_error rather than indexing the dict. Do not invent another
dict shape for a new tool. Some existing tools predate this and return a bare
value -- 'finished' returns {} on success (only a gated wrapper's refusal is a
tool_error), and get_proposed_construction_plan / get_approved_construction_plan
return the plan itself -- so do not assume every result in the tree is a
ToolResult, and do not reshape those without checking the instructions that
read them.
"""

from typing import Any, Callable, Dict, Mapping

# A plain dict, not a TypedDict union: tool_success() stores the payload under a
# caller-chosen key ("records", "files", ...), which no TypedDict can describe.
# (google-adk 2.10 can declare a Union-of-TypedDicts return, so that is not the
# reason.)
ToolResult = Dict[str, Any]


def tool_success(key: str, result: Any) -> ToolResult:
    """Create a successful result containing the given value.

    Args:
        key: the key to store the result under
        result: The successful result value

    Returns:
        ToolResult: success dict with the result under the given key
    """
    return {"status": "success", key: result}


def tool_error(message: str) -> ToolResult:
    """Create an error result with the given message.

    Args:
        message: The error message
        error_type: Optional exception type to use (defaults to ValueError)

    Returns:
        ToolResult: error dict
    """
    return {
        "status": "error",
        "error_message": str(message) if message is not None else "Unknown error",
    }


def is_success(result: ToolResult) -> bool:
    return result["status"] == "success"


def is_error(result: ToolResult) -> bool:
    # Tolerant, unlike is_success: ADK's after-tool callbacks see whatever a
    # tool returned, including dicts with no "status" (make_finished's {}).
    return isinstance(result, Mapping) and result.get("status") == "error"


def _payload_key(result: Mapping[str, Any]) -> str:
    """Return the key holding the payload of a success result.

    Prefers "result" when present; otherwise requires exactly one
    non-"status" key, since that's the only key tool_success() sets.
    """
    if "result" in result:
        return "result"
    keys = [k for k in result if k != "status"]
    if len(keys) != 1:
        raise ValueError(
            f"Ambiguous or missing payload key in success result: {result!r}"
        )
    return keys[0]


def map_result(result: ToolResult, f: Callable[[Any], Any]) -> ToolResult:
    if not is_success(result):
        return result
    key = _payload_key(result)
    return tool_success(key, f(result[key]))


def map_error(result: ToolResult, f: Callable[[str], Any]) -> ToolResult:
    return (
        {"status": "error", "error_message": f(result["error_message"])}
        if is_error(result)
        else result
    )


def get_or_else(result: ToolResult, default: Any) -> Any:
    return result[_payload_key(result)] if is_success(result) else default


def get_or_raise(result: ToolResult) -> Any:
    if is_success(result):
        return result[_payload_key(result)]
    elif is_error(result):
        raise Exception(result["error_message"])
