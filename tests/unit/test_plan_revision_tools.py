"""The clear question after a way back: asked once, answered in a later turn."""

import pytest

from agentic_kg.tools import plan_revision_record as record
from agentic_kg.tools import plan_revision_tools as tools
from agentic_kg.tools.kg_construction_tools import (
    APPROVED_CONSTRUCTION_PLAN,
    NOT_APPROVED_MESSAGE,
)

_PLAN = {
    "Customer": {"construction_type": "node", "label": "Customer"},
    "BOUGHT": {"construction_type": "relationship", "relationship_type": "BOUGHT"},
}
_EMPTY = {
    "nodes": 0,
    "relationships": 0,
    "labels": {},
    "relationship_types": {},
    "constraints": [],
    "indexes": [],
}
_FULL = {
    "nodes": 5,
    "relationships": 2,
    "labels": {"Person": 3, "Customer": 2},
    "relationship_types": {"KNOWS": 1, "BOUGHT": 1},
    "constraints": ["c1"],
    "indexes": [],
}


class FakeToolContext:
    def __init__(self, state, invocation_id="inv-2"):
        self.state = state
        self.invocation_id = invocation_id


def _state(status=None, **fields):
    state = {APPROVED_CONSTRUCTION_PLAN: _PLAN}
    if status:
        state[record.PLAN_REVISION_KEY] = {"status": status, **fields}
    return state


@pytest.fixture
def contents(monkeypatch):
    def install(value):
        monkeypatch.setattr(
            tools, "database_contents", lambda: {"status": "success", "contents": value}
        )

    return install


@pytest.fixture
def erase(monkeypatch):
    calls = []

    def install(result=None):
        def fake():
            calls.append(True)
            return result or {"status": "success", "message": "reset"}

        monkeypatch.setattr(tools, "reset_neo4j_data", fake)
        return calls

    return install


# check_database_before_rebuild


def test_no_record_means_no_rebuild_and_no_query(monkeypatch):
    monkeypatch.setattr(tools, "database_contents", lambda: pytest.fail("queried"))
    result = tools.check_database_before_rebuild(FakeToolContext({}))
    assert result == {"status": "success", "rebuild": tools.NO_REBUILD_PENDING}


def test_no_record_and_no_approval_still_says_no_rebuild(monkeypatch):
    monkeypatch.setattr(tools, "database_contents", lambda: pytest.fail("queried"))
    result = tools.check_database_before_rebuild(
        FakeToolContext({APPROVED_CONSTRUCTION_PLAN: None})
    )
    assert result["rebuild"] == tools.NO_REBUILD_PENDING


@pytest.mark.parametrize("status", [record.PENDING, record.ASKED, record.ANSWERED])
def test_a_record_with_a_withdrawn_approval_is_refused(status, monkeypatch):
    monkeypatch.setattr(tools, "database_contents", lambda: pytest.fail("queried"))
    state = _state(status, asked_in="inv-1", contents=_FULL, cleared=None)
    state[APPROVED_CONSTRUCTION_PLAN] = None
    result = tools.check_database_before_rebuild(FakeToolContext(state))
    assert result == {"status": "error", "error_message": NOT_APPROVED_MESSAGE}


def test_answered_means_rebuild_owed_and_nothing_asked(monkeypatch):
    monkeypatch.setattr(tools, "database_contents", lambda: pytest.fail("queried"))
    result = tools.check_database_before_rebuild(
        FakeToolContext(_state(record.ANSWERED, cleared=False))
    )
    assert result == {
        "status": "success",
        "rebuild": tools.REBUILD_OWED + " " + tools.OTHER_REQUEST,
    }
    assert "steps 1 to 6" in tools.REBUILD_OWED


def test_pending_and_empty_settles_without_asking(contents):
    contents(_EMPTY)
    state = _state(record.PENDING)
    result = tools.check_database_before_rebuild(FakeToolContext(state))
    assert result["status"] == "success"
    assert "rebuild" in result
    assert record.revision(state) == {"status": record.ANSWERED, "cleared": None}


def test_pending_and_not_empty_asks_and_records_this_turn(contents):
    contents(_FULL)
    state = _state(record.PENDING)
    result = tools.check_database_before_rebuild(FakeToolContext(state, "inv-7"))
    assert "Person: 3" in result["question"]
    assert "whole database" in result["question"]
    assert record.revision(state) == {
        "status": record.ASKED,
        "asked_in": "inv-7",
        "contents": _FULL,
    }


def test_asked_in_this_turn_says_ask_now(contents):
    contents(_EMPTY)  # must not be read
    state = _state(record.ASKED, asked_in="inv-2", contents=_FULL)
    result = tools.check_database_before_rebuild(FakeToolContext(state, "inv-2"))
    assert "Ask the user now" in result["question"]


def test_asked_in_an_earlier_turn_says_answer_or_ask_again(contents):
    contents(_EMPTY)
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    result = tools.check_database_before_rebuild(FakeToolContext(state, "inv-2"))
    assert "earlier turn" in result["question"]
    assert "If it does neither, ask again" in result["question"]
    assert "whole database" in result["question"]
    assert "end your reply with the question" in result["question"]
    assert "including anything this program did not build" in result["question"]
    assert record.revision(state)["asked_in"] == "inv-1"


_OTHER_REQUEST = "If the user's latest message asks for something else"


def test_asked_in_an_earlier_turn_lets_a_different_request_win(contents):
    contents(_EMPTY)
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    result = tools.check_database_before_rebuild(FakeToolContext(state, "inv-2"))
    text = " ".join(result["question"].split())
    assert _OTHER_REQUEST in text
    assert "such as changing the plan again, do that instead" in text
    assert "If it does neither, ask again: list" in text


def test_answered_lets_a_different_request_win():
    result = tools.check_database_before_rebuild(
        FakeToolContext(_state(record.ANSWERED, cleared=False))
    )
    text = " ".join(result["rebuild"].split())
    assert _OTHER_REQUEST in text
    assert "such as changing the plan again, do that instead" in text


def test_a_failed_read_leaves_the_record_alone(monkeypatch):
    monkeypatch.setattr(
        tools,
        "database_contents",
        lambda: {"status": "error", "error_message": "NXDOMAIN"},
    )
    state = _state(record.PENDING)
    result = tools.check_database_before_rebuild(FakeToolContext(state))
    assert result == {"status": "error", "error_message": "NXDOMAIN"}
    assert record.revision(state) == {"status": record.PENDING}


def test_a_second_revision_after_a_rebuild_asks_again(contents):
    """Once per revision, not once per session."""
    contents(_FULL)
    state = _state(None)
    state[record.PLAN_REVISION_KEY] = None  # used up by the first rebuild
    record.mark_pending(state)  # the user went back again
    result = tools.check_database_before_rebuild(FakeToolContext(state, "inv-9"))
    assert "question" in result
    assert record.revision(state)["status"] == record.ASKED


# clear_database_for_rebuild and keep_database_for_rebuild


@pytest.mark.parametrize(
    "answer", ["clear_database_for_rebuild", "keep_database_for_rebuild"]
)
@pytest.mark.parametrize("status", [None, record.PENDING, record.ANSWERED])
def test_answers_refuse_unless_asked(answer, status, erase):
    calls = erase()
    state = _state(status, cleared=None)
    result = getattr(tools, answer)(FakeToolContext(state))
    assert result["status"] == "error"
    assert calls == []


@pytest.mark.parametrize(
    "answer", ["clear_database_for_rebuild", "keep_database_for_rebuild"]
)
def test_answers_refuse_in_the_turn_that_asked(answer, erase):
    calls = erase()
    state = _state(record.ASKED, asked_in="inv-2", contents=_FULL)
    result = getattr(tools, answer)(FakeToolContext(state, "inv-2"))
    assert result == {"status": "error", "error_message": tools.SAME_TURN_REFUSAL}
    assert record.revision(state)["status"] == record.ASKED
    assert calls == []


@pytest.mark.parametrize(
    "answer", ["clear_database_for_rebuild", "keep_database_for_rebuild"]
)
def test_answers_refuse_a_withdrawn_approval(answer, erase):
    calls = erase()
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    state[APPROVED_CONSTRUCTION_PLAN] = None
    result = getattr(tools, answer)(FakeToolContext(state, "inv-2"))
    assert result == {"status": "error", "error_message": NOT_APPROVED_MESSAGE}
    assert calls == []


def test_clear_erases_and_settles(erase):
    calls = erase()
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    result = tools.clear_database_for_rebuild(FakeToolContext(state, "inv-2"))
    assert result["status"] == "success"
    assert calls == [True]
    assert record.revision(state) == {"status": record.ANSWERED, "cleared": True}


def test_a_failed_erase_keeps_the_question_open(erase):
    erase({"status": "error", "error_message": "still holds c1"})
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    result = tools.clear_database_for_rebuild(FakeToolContext(state, "inv-2"))
    assert result == {"status": "error", "error_message": "still holds c1"}
    assert record.revision(state)["status"] == record.ASKED


def test_keep_lists_what_the_new_plan_does_not_build(contents, erase):
    calls = erase()
    contents(_FULL)
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    result = tools.keep_database_for_rebuild(FakeToolContext(state, "inv-2"))
    kept = result["kept"]
    assert kept["labels_not_built"] == {"Person": 3}
    assert kept["relationship_types_not_built"] == {"KNOWS": 1}
    assert "in the database but not built by the new plan" in kept["message"]
    assert "counts" in kept["message"]
    assert record.revision(state) == {"status": record.ANSWERED, "cleared": False}
    assert calls == []


def test_a_failed_read_in_keep_leaves_the_question_open(monkeypatch):
    monkeypatch.setattr(
        tools,
        "database_contents",
        lambda: {"status": "error", "error_message": "NXDOMAIN"},
    )
    state = _state(record.ASKED, asked_in="inv-1", contents=_FULL)
    result = tools.keep_database_for_rebuild(FakeToolContext(state, "inv-2"))
    assert result == {"status": "error", "error_message": "NXDOMAIN"}
    assert record.revision(state)["status"] == record.ASKED
