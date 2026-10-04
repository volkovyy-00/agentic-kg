"""KG-48: proposing a node refuses a key whose rows would collapse.

These run the real propose tools against the bundled `data/bom` example and
against generated files; the module's own cases are in test_node_key_check.py.
"""

from pathlib import Path

import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.tool_result import tool_error
from agentic_kg.tools import construction_plan_tools as cpt
from agentic_kg.tools.construction_plan_tools import (
    PROPOSED_CONSTRUCTION_PLAN,
    propose_node_construction,
    propose_node_constructions,
)

BOM = Path(__file__).resolve().parents[2] / "data" / "bom"


class FakeToolContext:
    def __init__(self):
        self.state = {}


@pytest.fixture
def ctx():
    return FakeToolContext()


@pytest.fixture
def bom_source(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", str(BOM))
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def generated(monkeypatch, tmp_path):
    """Write a CSV under a SOURCE_URI of its own and return its name."""
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()

    def write(name, text):
        (tmp_path / name).write_text(text)
        return name

    yield write
    reset_settings()


def _plan(ctx):
    return ctx.state.get(PROPOSED_CONSTRUCTION_PLAN, {})


# --- the bundled example is not refused ----------------------------------------


@pytest.mark.parametrize(
    "source_file, label, key, properties",
    [
        ("products.csv", "Product", "product_id", ["product_name", "price"]),
        ("suppliers.csv", "Supplier", "supplier_id", ["name", "specialty", "city"]),
        ("assemblies.csv", "Assembly", "assembly_id", ["assembly_name", "product_id"]),
        # The one repeating key: 88 parts over 176 rows, one row per supplier.
        # Every row of a part agrees on its name, so nothing is lost.
        ("part_supplier_mapping.csv", "Part", "part_id", ["part_name"]),
    ],
)
def test_the_bundled_reference_nodes_are_proposed(
    bom_source, ctx, source_file, label, key, properties
):
    result = propose_node_construction(source_file, label, key, properties, ctx)

    assert result["status"] == "success", result.get("error_message")
    assert label in _plan(ctx)


def test_a_per_supplier_property_on_part_is_refused_and_not_stored(bom_source, ctx):
    """unit_cost differs for every supplier of a part, so a Part node would keep
    one supplier's cost. It belongs on the relationship."""
    result = propose_node_construction(
        "part_supplier_mapping.csv", "Part", "part_id", ["part_name", "unit_cost"], ctx
    )

    assert result["status"] == "error"
    message = result["error_message"]
    assert "176 rows, 88 distinct values, 0 blank" in message
    assert "'unit_cost'" in message
    assert "'part_name'" not in message
    assert "Part" not in _plan(ctx)


def test_a_node_keyed_on_a_per_row_property_holder_is_refused(bom_source, ctx):
    """KG-22's observed plan: SubAssembly keyed on a name that repeats over 88
    rows, holding the per-row part_id."""
    result = propose_node_construction(
        "components.csv", "SubAssembly", "sub_assembly_name", ["part_id"], ctx
    )

    assert result["status"] == "error"
    assert "88 rows, 27 distinct values, 0 blank" in result["error_message"]
    assert "'part_id'" in result["error_message"]


# --- generated files ----------------------------------------------------------


def test_an_order_line_node_keyed_on_the_order_is_refused(generated, ctx):
    """KG-33's observed plan: one row per product on an order, keyed by order."""
    name = generated(
        "order_details.csv",
        "orderID,productID,unitPrice,quantity\n"
        "10248,11,14.0,12\n10248,42,9.8,10\n10249,14,18.6,9\n10249,51,42.4,40\n",
    )

    result = propose_node_construction(
        name, "OrderDetail", "orderID", ["productID", "unitPrice", "quantity"], ctx
    )

    assert result["status"] == "error"
    message = result["error_message"]
    assert "4 rows, 2 distinct values, 0 blank" in message
    assert "'productID', 'unitPrice', 'quantity'" in message
    assert "'10248'" in message
    assert "OrderDetail" not in _plan(ctx)


def test_a_repeating_key_with_no_properties_is_accepted(generated, ctx):
    """Customer ids taken from an orders file with nothing else kept."""
    name = generated("orders.csv", "orderID,customerID\n1,C1\n2,C1\n3,C2\n")

    result = propose_node_construction(name, "Customer", "customerID", [], ctx)

    assert result["status"] == "success"


def test_a_blank_key_is_refused_even_when_every_property_agrees(generated, ctx):
    name = generated("plots.csv", "plot,soil\nA1,clay\n,clay\nA2,sand\n")

    result = propose_node_construction(name, "Plot", "plot", ["soil"], ctx)

    assert result["status"] == "error"
    assert "3 rows, 2 distinct values, 1 blank" in result["error_message"]
    assert "Plot" not in _plan(ctx)


def test_a_file_ending_in_a_blank_line_is_refused_in_words_that_name_the_cause(
    generated, ctx
):
    name = generated("plots.csv", "plot,soil\nA1,clay\nA2,sand\n\n")

    result = propose_node_construction(name, "Plot", "plot", ["soil"], ctx)

    assert result["status"] == "error"
    assert "a blank line at the end of the file counts" in result["error_message"]


def test_an_unreadable_property_list_skips_the_agreement_check(generated, ctx):
    """A bare string is not a property list; approval flags it. The check cannot
    judge it, so it must neither crash nor refuse on it."""
    name = generated("visits.csv", "plot,soil\nA1,clay\nA1,sand\n")

    result = propose_node_construction(name, "Plot", "plot", "soil", ctx)

    assert result["status"] == "success"


# --- the batch tool inherits the check ------------------------------------------


def test_a_batch_stops_at_the_collapsing_node_and_keeps_the_earlier_ones(
    bom_source, ctx
):
    result = propose_node_constructions(
        [
            {
                "approved_file": "products.csv",
                "proposed_label": "Product",
                "unique_column_name": "product_id",
                "proposed_properties": ["product_name"],
            },
            {
                "approved_file": "part_supplier_mapping.csv",
                "proposed_label": "Part",
                "unique_column_name": "part_id",
                "proposed_properties": ["part_name", "unit_cost"],
            },
        ],
        ctx,
    )

    assert result["status"] == "error"
    assert result["error_message"].startswith(
        "node construction 1 (Part) failed: Cannot key Part nodes"
    )
    assert set(_plan(ctx)) == {"Product"}


# --- order and failures -----------------------------------------------------------


def test_a_bad_name_or_header_is_refused_before_the_key_is_read(
    bom_source, ctx, monkeypatch
):
    def read(*args, **kwargs):
        raise AssertionError("the key column was read")

    monkeypatch.setattr(cpt, "summarize_node_key", read)

    bad_name = propose_node_construction(
        "products.csv", "bad label", "product_id", [], ctx
    )
    bad_header = propose_node_construction(
        "products.csv", "Product", "no_such_column", [], ctx
    )

    assert bad_name["status"] == "error"
    assert bad_header["status"] == "error"
    assert "no_such_column" in bad_header["error_message"]


def test_a_read_failure_comes_back_as_the_tools_own_error(bom_source, ctx, monkeypatch):
    monkeypatch.setattr(
        cpt, "summarize_node_key", lambda *args: (None, tool_error("disk fell over"))
    )

    result = propose_node_construction(
        "products.csv", "Product", "product_id", ["product_name"], ctx
    )

    assert result == tool_error("disk fell over")
    assert "Product" not in _plan(ctx)
