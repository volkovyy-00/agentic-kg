"""No module in the package builds a dict with a literal "error" key.

rejected_call_cap.is_adk_rejection reads a tool result whose only key is
"error", holding a string, as ADK's own rejection of the call. A tool that
returned that shape itself would count as stuck, and three in a row would end
the turn. Tools return tool_error() (status plus error_message).

Catches three spellings: a dict literal, a dict(error=...) call and a
subscript store (d["error"] = ...). Reads such as result["status"] == "error"
are not stores and pass. Misses keys built at run time -- a plan keyed by a
model-chosen label is one, which is why is_adk_rejection also requires a
single string-valued key -- and results from third-party code.
"""

import ast
from pathlib import Path

import pytest

import agentic_kg

PACKAGE_DIR = Path(agentic_kg.__file__).parent


def _error_key_lines(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            lines += [
                node.lineno
                for key in node.keys
                if isinstance(key, ast.Constant) and key.value == "error"
            ]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            lines += [
                node.lineno for keyword in node.keywords if keyword.arg == "error"
            ]
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "error"
        ):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    "source",
    [
        'result = {"error": "no"}',
        'result = dict(error="no")',
        'result["error"] = "no"',
        'result = {"status": "error", "error": "no"}',
    ],
)
def test_the_scan_finds_each_spelling(source):
    assert _error_key_lines(source) == [1]


@pytest.mark.parametrize(
    "source",
    [
        'failed = result["status"] == "error"',
        'message = result["error"]',
        'result = {"status": "error", "error_message": "no"}',
        "result = {key: 1}",
        "result = {**other}",
    ],
)
def test_the_scan_ignores_reads_and_other_keys(source):
    assert _error_key_lines(source) == []


def test_no_module_in_the_package_builds_a_bare_error_result():
    offenders = [
        f"{path.relative_to(PACKAGE_DIR.parent)}:{line}"
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for line in _error_key_lines(path.read_text())
    ]
    assert not offenders, "\n".join(
        [
            *offenders,
            "Fix: return tool_error(...) (agentic_kg.common.tool_result). The "
            "rejected-call cap reads a lone 'error' key as ADK rejecting the call.",
        ]
    )
