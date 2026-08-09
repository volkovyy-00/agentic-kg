# tests/unit/test_graph_profile_cache.py
"""Unit tests for profile cache invalidation.

Two layers, because one is not honest. The counter cannot see writes made
outside this process, and those happen in this workflow -- so a counter-only
cache would serve a schema for a database that no longer exists and state it
as fact, which is the exact failure class this work exists to fix.
"""

import pytest

from agentic_kg.common import graph_profile


class FakeDb:
    def __init__(self, nodes=10, rels=5):
        self.write_count = 0
        self.nodes = nodes
        self.rels = rels
        self.fingerprint_calls = 0

    def send_read_query(self, query, parameters=None, max_rows=None):
        if "AS nodes" in query:
            self.fingerprint_calls += 1
            records = [{"nodes": self.nodes, "rels": self.rels}]
        else:
            records = []
        return {
            "status": "success",
            "query_result": {
                "records": records,
                "row_count": len(records),
                "truncated": False,
            },
        }


@pytest.fixture
def db(monkeypatch):
    fake = FakeDb()
    monkeypatch.setattr(graph_profile, "graphdb", fake)
    monkeypatch.setattr(graph_profile, "build_profile", lambda schema: {"built": True})
    graph_profile.reset_cache()
    yield fake
    graph_profile.reset_cache()


def _loader(counter):
    def load():
        counter.append(1)
        return {"node_props": {}, "rel_props": {}, "relationships": []}

    return load


def test_second_call_is_served_from_cache(db):
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 1


def test_write_counter_change_invalidates(db):
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    db.write_count += 1
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2


def test_fingerprint_change_invalidates_without_a_counter_change(db):
    """The out-of-process write case -- the reason one layer is not enough."""
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    db.nodes = 0  # someone wiped the graph from a script
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2


def test_cache_hit_still_costs_exactly_one_fingerprint_query(db):
    graph_profile.get_cached_profile(_loader([]))
    graph_profile.get_cached_profile(_loader([]))
    assert db.fingerprint_calls == 2


def test_reset_cache_forces_recompute(db):
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    graph_profile.reset_cache()
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2


def test_unreadable_fingerprint_does_not_poison_the_cache(db, monkeypatch):
    """Would catch: caching an entry whose fingerprint is None.

    Storing None makes the NEXT call miss too, because a real fingerprint tuple
    never equals None -- so one transient failure on the fingerprint query costs
    two full cold rebuilds (hundreds of queries) for a graph that never changed.
    Recomputing once during the outage is correct; recomputing again afterwards,
    when the graph is provably unchanged, is the regression.
    """
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 1  # cold build, cache now warm

    # One transient failure: the fingerprint query returns no rows.
    real_send = db.send_read_query

    def blip(query, parameters=None, max_rows=None):
        if "AS nodes" in query:
            db.fingerprint_calls += 1
            return {"status": "error", "error_message": "transient"}
        return real_send(query, parameters, max_rows)

    monkeypatch.setattr(db, "send_read_query", blip)
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2, "an unreadable fingerprint must force a recompute"

    # Blip over, graph provably unchanged -- this must be served from cache.
    monkeypatch.setattr(db, "send_read_query", real_send)
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2, (
        "the failed call cached fingerprint=None, so the recovered call missed "
        "and rebuilt a second time"
    )
