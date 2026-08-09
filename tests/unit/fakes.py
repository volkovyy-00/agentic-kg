"""Shared graph-database fakes for the unit suite.

Before this module, four test files each grew their own way to fake
`graphdb`, three of them differing only in how canned rows are looked up. New
tests then copied whichever one happened to be nearest. The two shapes that
genuinely differ are kept as separate classes here rather than merged into one
configurable fake, because a fake with every dispatch strategy bolted together
is harder to read than the code it stands in for.

Deliberately NOT covered here: `test_kg_construction_tools.py`'s pop-a-queue
fake, whose responses are consumed in call order rather than matched, and the
several `FakeToolContext` classes, which fake ADK session state rather than a
database. Those are different contracts, not variants of this one.
"""


class RecordingGraphDb:
    """Records every query and answers with an empty success by default.

    Covers both entry points, since `graphdb` is one object to its callers:
    `send_query` returns the flat `records` shape, `send_read_query` the nested
    `query_result` payload shape. Getting those two mixed up is a real bug this
    fake should not paper over, so they stay distinct.
    """

    def __init__(self):
        self.queries = []
        self.read_queries = []

    def send_query(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        return {"status": "success", "records": []}

    def send_read_query(self, query, parameters=None, max_rows=None):
        self.read_queries.append((query, parameters, max_rows))
        return {
            "status": "success",
            "query_result": {"records": [], "row_count": 0, "truncated": False},
        }

    # _physical_schema reads both of these before its try block, so a fake
    # without them raises AttributeError instead of exercising the tool.
    def get_driver(self):
        return object()

    def get_config(self):
        return type("Cfg", (), {"database": "neo4j"})()


class ScriptedGraphDb(RecordingGraphDb):
    """Answers read queries from a table keyed by substring of the query text.

    `responses` maps a distinctive fragment of a query to the rows it should
    return; `fail_on` maps the same way to a structured error, which is how
    per-entity failure isolation gets exercised without a database.

    Pick needles that appear in exactly one query. A needle like "count(*)"
    also matches the entity-count queries, whose rows carry different columns,
    and the resulting KeyError looks nothing like the bug it came from.
    """

    def __init__(self, responses=None, fail_on=None):
        super().__init__()
        self.responses = responses or {}
        self.fail_on = fail_on or ()

    def send_read_query(self, query, parameters=None, max_rows=None):
        self.queries.append(query)
        for needle in self.fail_on:
            if needle in query:
                return {"status": "error", "error_message": "boom"}
        for needle, records in self.responses.items():
            if needle in query:
                return {
                    "status": "success",
                    "query_result": {
                        "records": records,
                        "row_count": len(records),
                        "truncated": False,
                    },
                }
        return {
            "status": "success",
            "query_result": {"records": [], "row_count": 0, "truncated": False},
        }
