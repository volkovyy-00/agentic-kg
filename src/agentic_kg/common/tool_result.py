"""The result shape a new ADK tool in this package returns.

A tool returns tool_success(key, value) -- {"status": "success", key: value} --
or tool_error(message) -- {"status": "error", "error_message": message}.
Either may also carry warnings=[...]: a top-level "warnings" list of strings,
present only when the list is non-empty. The name "warnings" is reserved for
that list in every tool's result; never use it as a payload key.
Callers read a result through is_success, is_error, get_or_else, get_or_raise,
map_result and map_error rather than indexing the dict. Do not invent another
dict shape for a new tool. Never return a bare {"error": ...}: the
rejected-call cap (rejected_call_cap.is_adk_rejection) reads a result whose
only key is "error", holding a string, as ADK rejecting the call. Some existing
tools predate this and return a bare value -- 'finished' returns {} on success
(only a gated wrapper's refusal is a tool_error),
get_proposed_construction_plan returns the plan itself, and
get_approved_construction_plan returns the plan itself or a tool_error when
none is approved -- so do not assume every result in the tree is a ToolResult,
and do not reshape those without checking the instructions that read them.
"""

from typing import Any, Callable, Dict, Mapping

# A plain dict, not a TypedDict union: tool_success() stores the payload under a
# caller-chosen key ("records", "files", ...), which no TypedDict can describe.
# (google-adk 2.10 can declare a Union-of-TypedDicts return, so that is not the
# reason.)
ToolResult = Dict[str, Any]


def tool_success(
    key: str, result: Any, *, warnings: list[str] | None = None
) -> ToolResult:
    """Create a successful result containing the given value.

    Args:
        key: the key to store the result under; never "warnings"
        result: The successful result value
        warnings: warning strings, stored under "warnings" only when non-empty

    Returns:
        ToolResult: success dict with the result under the given key
    """
    return _with_warnings({"status": "success", key: result}, warnings)


def tool_error(message: str, *, warnings: list[str] | None = None) -> ToolResult:
    """Create an error result with the given message.

    Args:
        message: The error message
        warnings: warning strings, stored under "warnings" only when non-empty

    Returns:
        ToolResult: error dict
    """
    return _with_warnings(
        {
            "status": "error",
            "error_message": str(message) if message is not None else "Unknown error",
        },
        warnings,
    )


def _with_warnings(result: ToolResult, warnings: list[str] | None) -> ToolResult:
    # Only when non-empty: a build with nothing to warn about returns the same
    # result it always did, and no other tool's result gains the key.
    if warnings:
        result["warnings"] = warnings
    return result


def is_success(result: ToolResult) -> bool:
    return result["status"] == "success"


def is_error(result: ToolResult) -> bool:
    # Tolerant, unlike is_success: ADK's after-tool callbacks see whatever a
    # tool returned, including dicts with no "status" (make_finished's {}).
    return isinstance(result, Mapping) and result.get("status") == "error"


def _payload_key(result: Mapping[str, Any]) -> str:
    """Return the key holding the payload of a success result.

    Prefers "result" when present; otherwise requires exactly one key other
    than "status" and "warnings", since tool_success() sets only those.
    """
    if "result" in result:
        return "result"
    keys = [k for k in result if k not in ("status", "warnings")]
    if len(keys) != 1:
        raise ValueError(
            f"Ambiguous or missing payload key in success result: {result!r}"
        )
    return keys[0]


def map_result(result: ToolResult, f: Callable[[Any], Any]) -> ToolResult:
    if not is_success(result):
        return result
    key = _payload_key(result)
    return tool_success(key, f(result[key]), warnings=result.get("warnings"))


def map_error(result: ToolResult, f: Callable[[str], Any]) -> ToolResult:
    return (
        tool_error(f(result["error_message"]), warnings=result.get("warnings"))
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
