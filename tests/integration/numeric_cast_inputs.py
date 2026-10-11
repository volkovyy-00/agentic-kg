"""The values that pin graph_profile.NUMERIC_CAST to coerce in Neo4j.

Not a test module. test_numeric_cast_integration.py imports it, and the pre-merge Aura check (a one-off
script in the private notes repo) loads it by path, so both run exactly the same inputs.
"""

import importlib.util
from pathlib import Path

from agentic_kg.common.value_types import CURRENCY_SYMBOLS

_SIGNS = ("", "-", "+")
_SHAPES = ("42", "1,234.50", "0.5")
# Each place a value can carry whitespace: both ends, between sign and symbol, between symbol and digits,
# inside brackets.
_POSITIONS = (
    "{w}{s}{c}{d}{w}",
    "{s}{w}{c}{d}",
    "{s}{c}{w}{d}",
    "{c}{w}{s}{d}",
    "{c}{s}{w}{d}",
    "{w}({w}{c}{w}{d}{w}){w}",
    "{w}{d}{w}",
)


def _adversarial_values() -> list:
    """The converter's adversarial corpus, loaded by path so this module needs no sys.path setup."""
    path = (
        Path(__file__).resolve().parents[1] / "unit" / "test_value_types_invariants.py"
    )
    spec = importlib.util.spec_from_file_location("_value_types_invariants", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [value for value in module.ADVERSARIAL_VALUES if value is not None]


def generated_values() -> list:
    """The corpus, plus every Python whitespace character in every position, crossed with every currency
    symbol (and none), each sign and three number shapes. ASCII digits only: non-ASCII digits are a known
    gap with their own test."""
    values = set(_adversarial_values())
    # Every character str.strip() removes -- the whitespace coerce tolerates around a value.
    python_whitespace = [chr(c) for c in range(0x110000) if chr(c).isspace()]
    for w in python_whitespace:
        for c in ("", *CURRENCY_SYMBOLS):
            for s in _SIGNS:
                for d in _SHAPES:
                    for template in _POSITIONS:
                        values.add(template.format(w=w, s=s, c=c, d=d))
    return sorted(values)
