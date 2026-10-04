"""Temporal property types against a real Neo4j.

Reads values back through the driver itself (not through send_query, which turns
every temporal value into text) so what is asserted is what the database stores.
Needs Docker; with colima see CLAUDE.md for the two environment variables.
"""

import csv
from datetime import date, timedelta

import pytest
from neo4j import time as neo4j_time

import agentic_kg.tools.cypher_tools as cypher_tools
import agentic_kg.tools.kg_construction_tools as kg
from agentic_kg.common.config import reset_settings
from agentic_kg.common.value_types import BLANK

pytestmark = pytest.mark.integration

# After the imports on purpose: Ruff's E402 refuses an import below this guard.
try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)

EVENTS = (
    "id,day,at_zoned,at_local\n"
    "e1,2025-03-04,2025-03-04T10:11:12.1Z,2025-03-04T10:11:12.1\n"
    "e2,2025-03-05,2025-03-05T10:11:12.123456789+01:00,2025-03-05 10:11:12.123456789\n"
    "e3,2024-12-31,2025-03-06T01:02:03-05:30,2025-03-06T01:02\n"
    "e4,,,\n"
    "e5,03/04/2025,2025-03-07,2025-03-07T10:11:12Z\n"
)
EVENT_TYPES = {"day": "date", "at_zoned": "datetime", "at_local": "localdatetime"}


@pytest.fixture
def temporal_sources(neo4j_graph, tmp_path, monkeypatch):
    """CSVs in a temp folder. Depends on neo4j_graph so it runs after that
    fixture sets SOURCE_URI to data/bom, and wins."""
    (tmp_path / "events.csv").write_text(EVENTS)
    (tmp_path / "wrong.csv").write_text(
        "id,day\nw1,03/04/2025\nw2,04/05/2025\nw3,13/05/2025\nw4,2025-03-04\n"
    )
    (tmp_path / "sparse.csv").write_text("id,day\ns1,\ns2,\ns3,\n")
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)
    yield
    reset_settings()


def _raw(neo4j_graph, query):
    """Rows as the driver returns them, temporal values as neo4j.time objects."""
    driver, config = neo4j_graph._connection()
    with driver.session(database=config.database) as session:
        return list(session.run(query))


def _event(neo4j_graph, event_id):
    return _raw(
        neo4j_graph,
        f"MATCH (n:Event {{id: '{event_id}'}}) RETURN n.day AS day, "
        "n.at_zoned AS zoned, n.at_local AS local, valueType(n.day) AS day_type, "
        "valueType(n.at_zoned) AS zoned_type, valueType(n.at_local) AS local_type",
    )[0]


def _load_events(label="Event"):
    return kg.load_nodes_from_csv(
        "events.csv", label, "id", list(EVENT_TYPES), EVENT_TYPES
    )


def test_each_type_is_stored_as_its_neo4j_type_and_keeps_every_digit(
    neo4j_graph, temporal_sources
):
    result = _load_events()
    assert result["status"] == "success", result.get("error_message")

    e1 = _event(neo4j_graph, "e1")
    assert e1["day"] == neo4j_time.Date(2025, 3, 4)
    assert e1["day_type"].startswith("DATE")
    assert e1["zoned_type"].startswith("ZONED DATETIME")
    assert e1["local_type"].startswith("LOCAL DATETIME")
    assert e1["zoned"].nanosecond == 100000000
    assert e1["zoned"].utcoffset() == timedelta(0)
    assert e1["local"].nanosecond == 100000000

    e2 = _event(neo4j_graph, "e2")
    assert e2["zoned"].nanosecond == 123456789
    assert e2["zoned"].utcoffset() == timedelta(hours=1)  # kept as written
    assert e2["local"].nanosecond == 123456789

    e3 = _event(neo4j_graph, "e3")
    assert e3["zoned"].utcoffset() == -timedelta(hours=5, minutes=30)


def test_blank_and_wrong_shaped_cells_are_cleared_and_the_load_succeeds(
    neo4j_graph, temporal_sources
):
    result = _load_events()
    assert result["status"] == "success"

    # e4 is blank everywhere: nothing stored, and blanks never count toward the gate.
    e4 = _event(neo4j_graph, "e4")
    assert (e4["day"], e4["zoned"], e4["local"]) == (None, None, None)
    # e5 holds a day/month date, a bare date in a datetime column and a zoned
    # value in a local column: each cleared, never trimmed or filled in.
    e5 = _event(neo4j_graph, "e5")
    assert (e5["day"], e5["zoned"], e5["local"]) == (None, None, None)

    counts = result["rows_loaded"]["type_conversion"]
    assert counts["day"]["unconvertible"] == 1
    assert counts["day"][BLANK] == 1
    assert "warning" in result["rows_loaded"]


def test_a_column_that_is_mostly_wrong_stops_its_rule(neo4j_graph, temporal_sources):
    result = kg.load_nodes_from_csv(
        "wrong.csv", "Wrong", "id", ["day"], {"day": "date"}
    )
    assert result["status"] == "error"
    assert "declared date" in result["error_message"]


def test_a_column_of_only_blanks_does_not_trip_the_gate(neo4j_graph, temporal_sources):
    result = kg.load_nodes_from_csv(
        "sparse.csv", "Sparse", "id", ["day"], {"day": "date"}
    )
    assert result["status"] == "success", result.get("error_message")


def test_a_typed_date_filters_and_orders_with_no_cast(neo4j_graph, temporal_sources):
    _load_events()
    rows = _raw(
        neo4j_graph,
        "MATCH (n:Event) WHERE n.day > date('2025-01-01') "
        "RETURN n.id AS id ORDER BY n.day",
    )
    assert [row["id"] for row in rows] == ["e1", "e2"]


def test_a_re_run_retypes_a_text_date_property(neo4j_graph, temporal_sources):
    """A graph built before the types existed holds the date as a STRING; a re-run
    with the type declared must leave a real DATE, not the old text."""
    kg.load_nodes_from_csv("events.csv", "Event", "id", ["day"])
    assert _event(neo4j_graph, "e1")["day_type"].startswith("STRING")

    result = kg.load_nodes_from_csv(
        "events.csv", "Event", "id", ["day"], {"day": "date"}
    )
    assert result["status"] == "success", result.get("error_message")
    e1 = _event(neo4j_graph, "e1")
    assert e1["day"] == neo4j_time.Date(2025, 3, 4)
    assert e1["day_type"].startswith("DATE")


ORDER_COUNT = 830
BLANK_SHIPPED = 21
NORTHWIND_DATES = ["orderDate", "requiredDate", "shippedDate"]


def _write_orders(path):
    """830 orders shaped like Northwind's orders.csv, 21 of them never shipped.

    Generated here, not read from data/traders: that folder is a local-only
    download (.git/info/exclude), so a test reading it fails in a fresh clone.
    One order every two days, shipped a week later, keeps the August 2013
    window used below at 15 rows -- under read_neo4j_cypher's 50-row cap, so a
    truncated result cannot hide a wrong answer.
    """
    start = date(2013, 1, 1)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "orderID",
                "customerID",
                "employeeID",
                "orderDate",
                "requiredDate",
                "shippedDate",
                "shipperID",
                "freight",
            ]
        )
        for i in range(ORDER_COUNT):
            ordered = start + timedelta(days=2 * i)
            shipped = "" if i % 40 == 5 else (ordered + timedelta(days=7)).isoformat()
            writer.writerow(
                [
                    10248 + i,
                    "VINET",
                    5,
                    ordered.isoformat(),
                    (ordered + timedelta(days=28)).isoformat(),
                    shipped,
                    3,
                    "32.38",
                ]
            )


@pytest.fixture
def orders_source(neo4j_graph, tmp_path, monkeypatch):
    """Yields the path of a generated orders.csv, with SOURCE_URI pointing at it."""
    orders = tmp_path / "orders.csv"
    _write_orders(orders)
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)
    yield orders
    reset_settings()


def test_the_generated_orders_have_the_blank_count_the_tests_assume(tmp_path):
    """Guards the generator, so the two tests below cannot pass on wrong data."""
    _write_orders(tmp_path / "orders.csv")
    with open(tmp_path / "orders.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == ORDER_COUNT
    assert sum(1 for row in rows if not row["shippedDate"]) == BLANK_SHIPPED


def test_order_dates_are_stored_as_real_dates(neo4j_graph, orders_source):
    result = kg.load_nodes_from_csv(
        "orders.csv",
        "NwOrder",
        "orderID",
        NORTHWIND_DATES,
        {name: "date" for name in NORTHWIND_DATES},
    )
    assert result["status"] == "success", result.get("error_message")
    loaded = result["rows_loaded"]
    assert loaded["rows"] == ORDER_COUNT
    assert "warning" not in loaded
    # The blank shippedDate cells are cleared without tripping the gate.
    assert loaded["type_conversion"]["shippedDate"][BLANK] == BLANK_SHIPPED

    row = _raw(
        neo4j_graph,
        "MATCH (n:NwOrder) RETURN count(n) AS orders, count(n.shippedDate) AS shipped, "
        "count(n.orderDate) AS ordered, "
        "sum(CASE WHEN valueType(n.orderDate) STARTS WITH 'DATE' THEN 1 ELSE 0 END) "
        "AS dated",
    )[0]
    assert (row["orders"], row["shipped"], row["ordered"], row["dated"]) == (
        ORDER_COUNT,
        ORDER_COUNT - BLANK_SHIPPED,
        ORDER_COUNT,
        ORDER_COUNT,
    )


def test_an_agent_style_date_query_returns_the_rows_the_file_says(
    neo4j_graph, orders_source
):
    """The path the GraphRAG agent uses (read_neo4j_cypher) over typed dates, checked
    against an independent computation from the CSV, not against another query."""
    kg.load_nodes_from_csv(
        "orders.csv",
        "NwOrder",
        "orderID",
        NORTHWIND_DATES,
        {name: "date" for name in NORTHWIND_DATES},
    )

    with open(orders_source, newline="") as handle:
        source = list(csv.DictReader(handle))
    low, high = date(2013, 8, 1), date(2013, 9, 1)
    expected = sorted(
        (row["shippedDate"], row["orderID"])
        for row in source
        if row["shippedDate"] and low <= date.fromisoformat(row["shippedDate"]) < high
    )
    # Fewer rows than the tool's cap, so the comparison below sees all of them.
    assert 0 < len(expected) < 50

    result = cypher_tools.read_neo4j_cypher(
        "MATCH (n:NwOrder) "
        "WHERE n.shippedDate >= date('2013-08-01') AND n.shippedDate < date('2013-09-01') "
        "RETURN n.shippedDate AS shipped, n.orderID AS id "
        "ORDER BY n.shippedDate, n.orderID"
    )
    assert result["status"] == "success", result.get("error_message")
    assert result["query_result"]["truncated"] is False
    records = result["query_result"]["records"]
    assert [(r["shipped"], r["id"]) for r in records] == expected
