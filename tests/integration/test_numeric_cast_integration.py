"""NUMERIC_CAST gives, in Neo4j, the numbers coerce gives in Python.

The contract (graph_profile.NUMERIC_CAST): coerce's number for every value coerce converts, null for a
blank, nothing asserted on a value coerce refuses. These tests pin the Cypher recipe to the converter;
tests/unit/test_numeric_cast.py pins only its text.
"""

import csv

import pytest
from numeric_cast_inputs import generated_values

from agentic_kg.common.config import reset_settings
from agentic_kg.common.graph_profile import NUMERIC_CAST, NUMERIC_CAST_GAP_CHECK
from agentic_kg.common.value_types import BLANK, CONVERTED, FLOAT, coerce

pytestmark = pytest.mark.integration

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


_TRIM_MISSES = "\x85\xa0\u2007\u202f"

# SC2's values: refunds in both sign positions and in brackets, a separator, a symbol other than $,
# explicit plus signs, a space after the sign, and a blank.
LEDGER_AMOUNTS = [
    "-$42.00",
    "$-1.50",
    "(42.00)",
    "1,234",
    "€5",
    "+5",
    "$+5",
    "- 42",
    "",
]
# Bare numbers padded with a no-break space, before or after: the profile calls them "yes".
LEDGER_COUNTS = [f"\xa0{n}" if n % 2 else f"{n}\xa0" for n in range(1, 10)]
# Twelve distinct amounts, then a non-number and a blank: more than the ten values the profile lists.
LARGE_AMOUNTS = [f"${n},{n:03d}.00" for n in range(1, 13)] + ["N/A", ""]

LEDGER_PLAN = {
    "Entry": {
        "construction_type": "node",
        "source_file": "ledger.csv",
        "label": "Entry",
        "unique_column_name": "entry_id",
        "properties": ["amount", "count"],
    }
}
LARGE_PLAN = {
    "Line": {
        "construction_type": "node",
        "source_file": "large.csv",
        "label": "Line",
        "unique_column_name": "line_id",
        "properties": ["amount"],
    }
}


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _records(result):
    assert "records" in result, result.get("error_message", result)
    return result["records"]


@pytest.fixture
def numbers_source(neo4j_graph_with_apoc, tmp_path, monkeypatch):
    """Point SOURCE_URI at two CSVs. Depends on neo4j_graph_with_apoc so it runs after that fixture sets
    SOURCE_URI to data/bom, and wins."""
    _write_csv(
        tmp_path / "ledger.csv",
        ["entry_id", "amount", "count"],
        [
            [f"E-{i}", a, c]
            for i, (a, c) in enumerate(zip(LEDGER_AMOUNTS, LEDGER_COUNTS))
        ],
    )
    _write_csv(
        tmp_path / "large.csv",
        ["line_id", "amount"],
        [[f"L-{i}", a] for i, a in enumerate(LARGE_AMOUNTS)],
    )
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()
    yield neo4j_graph_with_apoc
    reset_settings()


def _build_and_profile(graph, plan, monkeypatch):
    import agentic_kg.tools.cypher_tools as cypher_tools
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.common import graph_profile

    monkeypatch.setattr(kg, "graphdb", graph)
    monkeypatch.setattr(cypher_tools, "graphdb", graph)
    monkeypatch.setattr(graph_profile, "graphdb", graph)
    graph_profile.reset_cache()
    built = kg.construct_domain_graph(plan)
    assert built["status"] == "success", built.get("error_message")
    result = cypher_tools.get_graph_schema_with_profile()
    graph_profile.reset_cache()
    assert result["status"] == "success", result.get("error_message")
    return {
        (entity, prop["property"]): prop
        for entity, props in result["schema"]["profile"]["properties"].items()
        for prop in props
    }


def test_an_untyped_ledger_casts_to_the_loaders_numbers(numbers_source, monkeypatch):
    """SC2 end to end: built untyped, profiled, then cast with the exact expression rule 6 quotes."""
    graph = numbers_source
    profile = _build_and_profile(graph, LEDGER_PLAN, monkeypatch)
    assert profile[("Entry", "amount")]["numeric_like"] == "numeric_after_cleaning"
    assert profile[("Entry", "count")]["numeric_like"] == "yes"

    rows = _records(
        graph.send_query(
            "MATCH (n:Entry) "
            f"WITH n.amount AS v, n.count AS c RETURN v, {NUMERIC_CAST} AS cast_amount, "
            + NUMERIC_CAST.replace("(v,", "(c,")
            + " AS cast_count, c"
        )
    )
    assert sorted(row["v"] for row in rows) == sorted(LEDGER_AMOUNTS)
    # The padding reached the graph, so the count column really exercised it.
    assert all("\xa0" in row["c"] for row in rows)
    for row in rows:
        assert row["cast_amount"] == coerce(row["v"], FLOAT)[0], row["v"]
        assert row["cast_count"] == coerce(row["c"], FLOAT)[0], repr(row["c"])

    total = _records(
        graph.send_query(
            f"MATCH (n:Entry) WITH n.amount AS v RETURN sum({NUMERIC_CAST}) AS t"
        )
    )
    expected = sum(coerce(a, FLOAT)[0] or 0.0 for a in LEDGER_AMOUNTS)
    assert total[0]["t"] == pytest.approx(expected)


def test_the_gap_check_finds_a_non_number_the_profile_did_not_list(
    numbers_source, monkeypatch
):
    """Review Focus 5: on a "partial" column the check counts N/A and not the blank."""
    graph = numbers_source
    profile = _build_and_profile(graph, LARGE_PLAN, monkeypatch)
    assert profile[("Line", "amount")]["completeness"] == "partial"

    rows = _records(
        graph.send_query("MATCH (n:Line) WITH n.amount AS v " + NUMERIC_CAST_GAP_CHECK)
    )
    assert len(rows) == 1
    assert rows[0]["uncast"] == 1
    assert rows[0]["examples"] == ["N/A"]


def test_the_gap_check_counts_nan_and_infinity_and_lists_distinct_examples(neo4j_graph):
    """toFloat reads 'NaN' and 'Infinity', which coerce refuses, and either one spoils a sum."""
    values = ["42", "NaN", "Infinity", "N/A", "N/A", "N/A", "N/A", "N/A", "TBD", ""]
    rows = _records(
        neo4j_graph.send_query(
            "UNWIND $values AS v WITH v " + NUMERIC_CAST_GAP_CHECK, {"values": values}
        )
    )
    assert rows[0]["uncast"] == 8
    assert sorted(rows[0]["examples"]) == ["Infinity", "N/A", "NaN", "TBD"]


def test_the_cast_gives_coerces_number_for_every_value_coerce_converts(neo4j_graph):
    values = generated_values()
    rows = _records(
        neo4j_graph.send_query(
            f"UNWIND $values AS v RETURN v, {NUMERIC_CAST} AS cast", {"values": values}
        )
    )
    converted = 0
    wrong = []
    for row in rows:
        number, outcome = coerce(row["v"], FLOAT)
        if outcome == CONVERTED:
            converted += 1
            if row["cast"] != number:
                wrong.append((row["v"], number, row["cast"]))
        elif outcome == BLANK and row["cast"] is not None:
            wrong.append((row["v"], None, row["cast"]))
    assert converted > 3000, converted
    assert wrong == []


def test_the_cast_works_when_the_model_sends_the_escapes_decoded(neo4j_graph):
    """Review Focus 3: the agent's tool call may turn \\u00A0 into the raw character first."""
    decoded = NUMERIC_CAST
    for char in _TRIM_MISSES:
        decoded = decoded.replace(f"\\u{ord(char):04X}", char)
    assert decoded != NUMERIC_CAST
    values = [f"{char}42" for char in _TRIM_MISSES] + ["$\xa0-1.50", "(42.00)\u202f"]
    rows = _records(
        neo4j_graph.send_query(
            f"UNWIND $values AS v RETURN v, {decoded} AS cast", {"values": values}
        )
    )
    for row in rows:
        assert row["cast"] == coerce(row["v"], FLOAT)[0], repr(row["v"])


def test_the_cast_fails_loudly_on_a_value_that_is_not_text(neo4j_graph):
    """Review Focus 4: a property holding a real number beside text makes the cast raise, so the query
    fails visibly instead of returning a smaller sum."""
    neo4j_graph.send_query(
        "CREATE (:MixedAmount {amount: '$5'}), (:MixedAmount {amount: 42})"
    )
    result = neo4j_graph.send_query(
        f"MATCH (n:MixedAmount) WITH n.amount AS v RETURN sum({NUMERIC_CAST}) AS total"
    )
    assert result["status"] == "error", result


def test_the_cast_query_runs_on_non_ascii_digits(neo4j_graph):
    """Keeps the strict xfail below honest: a Cypher error there would also count as the expected
    failure. Here the query must run and return one row per value."""
    values = ["٤٢", "４２", "४२"]
    rows = _records(
        neo4j_graph.send_query(
            f"UNWIND $values AS v RETURN {NUMERIC_CAST} AS cast", {"values": values}
        )
    )
    assert len(rows) == len(values)


@pytest.mark.xfail(
    strict=True,
    reason="Known gap: non-ASCII decimal digits convert in Python and cast to null in Neo4j.",
)
def test_non_ascii_digits_cast_to_the_number_coerce_gives(neo4j_graph):
    values = ["٤٢", "４２", "४२"]
    rows = _records(
        neo4j_graph.send_query(
            f"UNWIND $values AS v RETURN {NUMERIC_CAST} AS cast", {"values": values}
        )
    )
    assert [row["cast"] for row in rows] == [coerce(v, FLOAT)[0] for v in values]
