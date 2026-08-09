"""Guards against the demo dataset leaking into shipped code.

This work was diagnosed from one furniture supply-chain graph. Everything in
src/ must reason about graph *shapes* -- entity counts, distinct counts,
degree, completeness -- and never about suppliers. That includes prompt
strings, which are the largest and least reviewable overfitting surface.

This catches vocabulary overfitting only. It cannot see structural
overfitting (assuming one pattern per relationship type, assuming entities
below EXHAUSTIVE_SEARCH_LIMIT, assuming single-label nodes); that is what
tests/integration/test_graph_profile_shapes.py is for.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# Curated and deliberately narrow. Bare label names (Part, Product, Assembly)
# are EXCLUDED because `Part` collides with google.genai.types.Part, which this
# codebase uses legitimately -- a broad match would fail on correct code and be
# deleted by the next person. Do not "improve" this into a substring sweep.
FORBIDDEN_TOKENS = [
    "preferred_supplier",
    "supplier_id",
    "assembly_id",
    "part_id",
    "lead_time_days",
    "unit_cost",
    "minimum_order_quantity",
    "SUP-",
    "Screws",
]


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_dataset_vocabulary_absent_from_src(token):
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if token in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"Dataset token {token!r} found in src/: {offenders}. "
        "Shipped code must reason about graph shapes, not about one dataset. "
        "See 'Generality constraints' in the design spec."
    )
