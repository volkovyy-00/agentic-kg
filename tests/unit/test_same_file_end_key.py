"""KG-60: an end on a node built from the relationship's own file matches its key.

The plans below are the shapes seen on 2026-10-10 (traders HANDLED_BY, bom
HAS_ASSEMBLY), cut down to the rules the check reads, plus the cases that must
still pass. The check is structural, so no file is read.
"""

from types import SimpleNamespace

import pytest

from agentic_kg.tools.construction_plan_tools import (
    APPROVED_CONSTRUCTION_PLAN,
    PROPOSED_CONSTRUCTION_PLAN,
    approve_proposed_construction_plan,
    check_construction_plan_consistency,
)


def _node(source_file, label, key, properties):
    return {
        "construction_type": "node",
        "source_file": source_file,
        "label": label,
        "unique_column_name": key,
        "properties": properties,
        "property_types": {},
    }


def _rel(source_file, rel_type, from_label, from_column, to_label, to_column, **extra):
    rule = {
        "construction_type": "relationship",
        "source_file": source_file,
        "relationship_type": rel_type,
        "from_node_label": from_label,
        "from_node_column": from_column,
        "to_node_label": to_label,
        "to_node_column": to_column,
        "properties": [],
        "property_types": {},
    }
    rule.update(extra)
    return rule


def _traders(order_file="orders.csv", **handled_by):
    """Order and Employee; HANDLED_BY reads orders.csv. As observed, it finds the
    Order by employeeID; `handled_by` patches the relationship."""
    rel = _rel(
        "orders.csv", "HANDLED_BY", "Order", "employeeID", "Employee", "employeeID"
    )
    rel.update(handled_by)
    return {
        "Order": _node(order_file, "Order", "orderID", ["employeeID", "orderDate"]),
        "Employee": _node(
            "employees.csv", "Employee", "employeeID", ["employeeName", "email"]
        ),
        "HANDLED_BY": rel,
    }


def _bom():
    return {
        "Product": _node("products.csv", "Product", "product_id", ["product_name"]),
        "Assembly": _node(
            "assemblies.csv", "Assembly", "assembly_id", ["assembly_name", "product_id"]
        ),
        "HAS_ASSEMBLY": _rel(
            "assemblies.csv",
            "HAS_ASSEMBLY",
            "Product",
            "product_id",
            "Assembly",
            "product_id",
        ),
    }


def _reports_to(**overrides):
    """Employee REPORTS_TO Employee, both ends on employees.csv: the from end is
    the row's own employee, the to end points at its manager."""
    rel = _rel(
        "employees.csv",
        "REPORTS_TO",
        "Employee",
        "employeeID",
        "Employee",
        "reportsTo",
        to_node_property="employeeID",
    )
    rel.update(overrides)
    return {
        "Employee": _node(
            "employees.csv",
            "Employee",
            "employeeID",
            ["employeeName", "email", "reportsTo", "reports_to_email"],
        ),
        "REPORTS_TO": rel,
    }


def _same_file_problems(plan):
    return [p for p in check_construction_plan_consistency(plan) if "built from" in p]


# --- refused ------------------------------------------------------------------


def test_the_traders_handled_by_shape_is_refused():
    problems = check_construction_plan_consistency(_traders())
    assert len(problems) == 1, problems
    problem = problems[0]
    for fragment in (
        "HANDLED_BY:",
        "'Order.employeeID'",
        "'orders.csv'",
        "'orderID'",
        "join the from end on 'orderID'",
        "key 'Order' by 'employeeID'",
    ):
        assert fragment in problem, fragment


def test_the_bom_has_assembly_shape_is_refused():
    problems = check_construction_plan_consistency(_bom())
    assert len(problems) == 1, problems
    problem = problems[0]
    for fragment in (
        "HAS_ASSEMBLY:",
        "'Assembly.product_id'",
        "'assemblies.csv'",
        "join the to end on 'assembly_id'",
    ):
        assert fragment in problem, fragment
    # The Product end reads a file Product is not built from: it is not named.
    assert "'Product.product_id'" not in problem


def test_a_property_unique_in_the_data_is_refused_too():
    """The check is structural: a second identifier is still not the key. Here
    the fix is to key Employee by email, which moves the row's own end too."""
    plan = _reports_to(to_node_column="reports_to_email", to_node_property="email")
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    problem = problems[0]
    for fragment in (
        "REPORTS_TO:",
        "'Employee.email'",
        "'employeeID'",
        "key 'Employee' by 'email' and join the from end on 'email' too",
    ):
        assert fragment in problem, fragment


def test_an_end_reading_a_differently_named_column_is_refused():
    """KG-45 lets an end read 'emp' and match Order.employeeID: the same bug
    under another column name."""
    plan = _traders(from_node_column="emp", from_node_property="employeeID")
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert "'Order.employeeID'" in problems[0]
    assert (
        "if 'emp' holds Order 'orderID' values, set 'from_node_property' to "
        "'orderID'" in problems[0]
    )


@pytest.mark.parametrize(
    "order_file, rel_file",
    [
        ("./orders.csv", "orders.csv"),
        ("orders.csv", "./orders.csv"),
        ("sales//orders.csv", "sales/orders.csv"),
        ("sales/./orders.csv", "sales/orders.csv"),
    ],
)
def test_two_spellings_of_one_file_are_the_same_file(order_file, rel_file):
    plan = _traders(order_file=order_file, source_file=rel_file)
    assert len(_same_file_problems(plan)) == 1


@pytest.mark.parametrize(
    "order_file, rel_file",
    [
        ("Orders.csv", "orders.csv"),
        ("sales\\orders.csv", "sales/orders.csv"),
        ("other/orders.csv", "orders.csv"),
    ],
)
def test_names_that_may_be_different_files_are_not_compared_as_one(
    order_file, rel_file
):
    """Case and backslashes are kept: on a POSIX source both are part of a name."""
    plan = _traders(order_file=order_file, source_file=rel_file)
    assert _same_file_problems(plan) == []


def test_a_column_that_may_hold_the_key_is_offered_the_key_as_its_property():
    """supersededBy may hold item IDs: keeping the column and matching the key is
    then the fix, and the other two would give the rule identical ends."""
    plan = _reports_to()
    plan["REPORTS_TO"].pop("to_node_property")
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert problems[0].endswith(
        "If 'reportsTo' holds Employee 'employeeID' values, set 'to_node_property' "
        "to 'employeeID'."
    )


def test_two_offending_ends_are_one_line_naming_both():
    plan = {
        "Order": _node("orders.csv", "Order", "orderID", ["employeeID"]),
        "Shipment": _node("orders.csv", "Shipment", "shipmentID", ["freight"]),
        "SHIPPED_AS": _rel(
            "orders.csv", "SHIPPED_AS", "Order", "employeeID", "Shipment", "freight"
        ),
    }
    problems = check_construction_plan_consistency(plan)
    assert len(problems) == 1, problems
    assert "'Order.employeeID'" in problems[0]
    assert "'Shipment.freight'" in problems[0]


def test_approval_refuses_the_traders_shape():
    ctx = SimpleNamespace(state={PROPOSED_CONSTRUCTION_PLAN: _traders()})
    result = approve_proposed_construction_plan(ctx)  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "'Order.employeeID'" in result["error_message"]
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state


# --- still approved -----------------------------------------------------------


def test_the_same_file_end_matched_on_its_key_is_consistent():
    plan = _traders(from_node_column="orderID")
    assert check_construction_plan_consistency(plan) == []


def test_a_pointer_end_matched_on_the_key_is_consistent():
    assert check_construction_plan_consistency(_reports_to()) == []


def test_a_shared_column_read_from_a_different_file_is_consistent():
    """One employee row linking all of that employee's orders stays allowed."""
    plan = _traders()
    plan["HANDLED_BY"] = _rel(
        "employees.csv", "HANDLED", "Employee", "employeeID", "Order", "employeeID"
    )
    assert check_construction_plan_consistency(plan) == []


# --- which fixes are offered, and when it stays silent ------------------------


def test_a_typed_property_gets_no_rekey_fix():
    """An identifier must stay text, so keying by a typed property would only be
    refused in turn. The typed-join refusal still fires, as its own line."""
    plan = _traders()
    plan["Order"]["property_types"] = {"employeeID": "integer"}
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert "key 'Order' by" not in problems[0]
    assert "join the from end on 'orderID'" in problems[0]


def test_no_rekey_fix_when_the_other_end_points_through_the_old_key():
    """The to end reads reportsTo, which holds employee IDs. Keyed by email,
    nothing could match it, so the rekey would trade one refusal for another."""
    plan = _reports_to(from_node_column="email", from_node_property="email")
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert "key 'Employee' by" not in problems[0]
    assert "join the from end on 'employeeID'" in problems[0]


def test_identical_ends_get_only_their_own_refusal():
    plan = _reports_to(to_node_column="email", from_node_column="email")
    plan["REPORTS_TO"].pop("to_node_property")
    problems = check_construction_plan_consistency(plan)
    assert len(problems) == 1, problems
    assert "itself included" in problems[0]


@pytest.mark.parametrize("no_key", [None, ""], ids=repr)
def test_a_node_without_a_key_is_told_to_key_it(no_key):
    plan = _traders()
    plan["Order"]["unique_column_name"] = no_key
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert problems[0].endswith("Key 'Order' by the column that identifies it first.")
    assert "None" not in problems[0]


def test_a_missing_node_rule_is_left_to_the_missing_node_problem():
    plan = _traders()
    del plan["Order"]
    problems = check_construction_plan_consistency(plan)
    assert problems
    assert all("has no node construction" in p for p in problems), problems


def test_an_unreadable_node_rule_is_left_to_its_own_problem():
    plan = _traders()
    plan["Order"]["properties"] = "employeeID"
    assert _same_file_problems(plan) == []


def test_a_property_the_node_does_not_carry_is_left_to_the_zero_match_problem():
    plan = _traders(from_node_column="salesRep")
    problems = check_construction_plan_consistency(plan)
    assert len(problems) == 1, problems
    assert "would match zero rows" in problems[0]


@pytest.mark.parametrize("rule", ["HANDLED_BY", "Order"])
@pytest.mark.parametrize("value", [None, ["orders.csv"]], ids=repr)
def test_a_source_file_that_is_not_text_is_not_compared(rule, value):
    plan = _traders()
    plan[rule]["source_file"] = value
    assert _same_file_problems(plan) == []


# --- final audit: following an offered fix is not refused in turn ----------


def _items(**same_title):
    """Item keyed by itemID, built from items.csv; SAME_TITLE reads items.csv
    with both ends on Item.title."""
    rel = _rel("items.csv", "SAME_TITLE", "Item", "title", "Item", "title")
    rel.update(same_title)
    return {
        "Item": _node("items.csv", "Item", "itemID", ["title", "supersededBy"]),
        "SAME_TITLE": rel,
    }


def test_identical_ends_off_the_key_in_their_own_file_are_told_to_key_the_other_end():
    """Moving one end to another column leaves the other on Item.title, which
    is refused in turn: the advice has to cover both ends."""
    problems = check_construction_plan_consistency(_items())
    assert len(problems) == 1, problems
    assert "itself included" in problems[0]
    assert "and join the other end on 'itemID'" in problems[0]


def test_following_the_identical_ends_advice_gives_a_consistent_plan():
    plan = _items(
        from_node_column="itemID",
        to_node_column="supersededBy",
        to_node_property="itemID",
    )
    assert check_construction_plan_consistency(plan) == []


def test_identical_ends_from_another_file_keep_their_advice_unchanged():
    plan = _items(source_file="links.csv")
    problems = check_construction_plan_consistency(plan)
    assert len(problems) == 1, problems
    assert "join the other end" not in problems[0]


def test_no_rekey_fix_when_another_relationship_joins_the_node_on_its_key():
    """Keyed by employeeID instead, PLACED_BY's Order end would match off the
    key and be refused in turn, offering the rekey back."""
    plan = _traders()
    plan["Customer"] = _node("customers.csv", "Customer", "customerID", [])
    plan["PLACED_BY"] = _rel(
        "orders.csv", "PLACED_BY", "Order", "orderID", "Customer", "customerID"
    )
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert "key 'Order' by" not in problems[0]
    assert "join the from end on 'orderID'" in problems[0]


def test_no_rekey_fix_when_a_sibling_relationship_joins_the_node_on_its_key():
    plan = _reports_to(to_node_column="reports_to_email", to_node_property="email")
    plan["Dept"] = _node("depts.csv", "Dept", "deptID", [])
    plan["WORKS_IN"] = _rel(
        "employees.csv", "WORKS_IN", "Employee", "employeeID", "Dept", "deptID"
    )
    problems = _same_file_problems(plan)
    assert len(problems) == 1, problems
    assert "key 'Employee' by" not in problems[0]
