import os
from typing import Any, Dict, Optional
import re
import atexit
import logging

from neo4j import (
    GraphDatabase,
    Query,
    READ_ACCESS,
    Result,
)

from .config import get_settings
from .pydantic_neo4j import Neo4jConfig
from .tool_result import tool_success, tool_error

logger = logging.getLogger(__name__)

# Bound on how long a query issued through the READ path may run. A hung tool
# call in `adk web` is indistinguishable from a routing bug, which this project
# has been burned by before.
#
# Applied by send_read_query only. send_query -- the write path -- is
# deliberately left unbounded for now: bulk loads and reset_neo4j_data's
# `DETACH DELETE ... IN TRANSACTIONS` legitimately run past 30s, so timing it
# needs its own limit rather than this one. That gap is real and open, not an
# oversight; a runaway write can still hang a turn.
QUERY_TIMEOUT_SECONDS = 30

# How many rows are retained and shown. Not a tuned constant -- a judgement
# about how many rows are worth reading individually before the honest answer
# is "aggregate this instead". No behaviour may depend on its exact value.
MAX_RETURNED_ROWS = 50

# How far we keep counting past the cap before reporting a floor instead of an
# exact total. Counting is cheap (an int); claiming an exact number we did not
# finish counting would not be.
ROW_COUNT_CEILING = 100_000

# Lists longer than this are replaced by a summary string. Embedding vectors
# would otherwise be pasted into the model's context verbatim.
MAX_INLINE_LIST_LENGTH = 32

_TRUNCATION_NOTE = (
    "Records are capped. Counts, rankings and superlatives must come from a "
    "Cypher aggregation, never from counting these rows."
)

# Every row was returned, but at least one oversized list inside them was
# replaced by a summary. Stated separately so a complete-in-rows result cannot
# silently imply complete-in-values.
_SUMMARY_NOTE = (
    "All rows were returned, but at least one oversized list value was "
    "replaced by a summary of its shape. Query the elements directly if you "
    "need them."
)

def load_neo4j_config_from_settings() -> Neo4jConfig:
    settings = get_settings()
    neo4j_config = Neo4jConfig(dsn=settings.neo4j_dsn)

    logger.info("Neo4j expected at: " + f"{neo4j_config.uri}")

    return neo4j_config

def make_driver(neo4j_config: Neo4jConfig) -> GraphDatabase | None:
    """
    Connects to a Neo4j Graph Database according to the provided configuration.
    """
    driver_params = neo4j_config.to_driver_params()

    # Initialize the driver
    driver_instance = GraphDatabase.driver(
        driver_params["uri"],
        auth=driver_params["auth"]
    )
    return driver_instance

# NOTE: a `sanitize()` helper used to live here -- a character-class strip for
# "when a query param is not possible". It had no callers, and stripping unsafe
# characters is the wrong shape for this codebase anyway: identifiers are now
# either validated and rejected (cypher_identifiers.checked(), for
# model-supplied names) or backtick-quoted and preserved
# (graph_profile.quote(), for names read out of the database). Silently
# rewriting a name is neither. Removed rather than kept as a template.


def is_symbol(symbol: str) -> bool:
    """Validate that a string is a valid Neo4j symbol (no spaces, not a Cypher keyword).

    Args:
        symbol: The string to validate

    Returns:
        True if the string is a valid symbol, False otherwise
    """
    # Check for spaces
    if ' ' in symbol:
        return False

    # Common Cypher keywords that should not be used as identifiers
    cypher_keywords = [
        'MATCH', 'RETURN', 'WHERE', 'CREATE', 'DELETE', 'REMOVE', 'SET',
        'ORDER', 'BY', 'SKIP', 'LIMIT', 'MERGE', 'ON', 'OPTIONAL', 'DETACH',
        'WITH', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS',
        'UNION', 'ALL', 'LOAD', 'CSV', 'FROM', 'START', 'YIELD', 'CALL',
        'CONSTRAINT', 'ASSERT', 'INDEX', 'UNIQUE', 'DROP', 'EXISTS', 'USING',
        'PERIODIC', 'COMMIT', 'FOREACH', 'TRUE', 'FALSE', 'NULL', 'NOT', 'AND', 'OR', 'XOR',
        'IS', 'IN', 'STARTS', 'ENDS', 'CONTAINS'
    ]

    # Check if the symbol is a Cypher keyword (case-insensitive)
    if symbol.upper() in cypher_keywords:
        return False

    return True


def is_write_query(query: str) -> bool:
    """Heuristic write detection, used ONLY as a cache-invalidation hint.

    This is deliberately not a security boundary and must never be used as
    one. It matches text, so it cannot tell a keyword from a string literal
    ("... CONTAINS 'set forth' ..." reads as a write) and it misses camelCase
    procedure calls (\\bMERGE\\b finds no boundary inside `mergeNodes`, which is
    what apoc.refactor.mergeNodes is). Read-only enforcement is the server's
    job via default_access_mode -- see send_read_query.

    As a cache hint both error directions are benign: a false positive costs
    one recomputation, and a false negative is caught by the fingerprint layer
    in graph_profile.
    """
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD|DROP)\b", query, re.IGNORECASE)
        is not None
    )

def result_to_adk(result: Result) -> Dict[str, Any]:
    eager_result = result.to_eager_result()
    records = [to_python(record.data()) for record in eager_result.records]
    return tool_success("records", records)

def to_python(value):
    from neo4j.graph import Node, Relationship, Path
    from neo4j import Record
    import neo4j.time
    if isinstance(value, Record):
        return {k: to_python(v) for k, v in value.items()}
    elif isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [to_python(v) for v in value]
    elif isinstance(value, Node):
        return {
            "id": value.id,
            "labels": list(value.labels),
            "properties": to_python(dict(value))
        }
    elif isinstance(value, Relationship):
        return {
            "id": value.id,
            "type": value.type,
            "start_node": value.start_node.id,
            "end_node": value.end_node.id,
            "properties": to_python(dict(value))
        }
    elif isinstance(value, Path):
        return {
            "nodes": [to_python(node) for node in value.nodes],
            "relationships": [to_python(rel) for rel in value.relationships]
        }
    elif isinstance(value, neo4j.time.DateTime):
        return value.iso_format()
    elif isinstance(value, (neo4j.time.Date, neo4j.time.Time, neo4j.time.Duration)):
        return str(value)
    else:
        return value


def _summarise(value, omitted: list):
    """Recursive worker for summarise_long_lists.

    Appends the length of each list it replaces to `omitted`, so the caller can
    tell the difference between "nothing was withheld" and "something was
    withheld silently". Without that signal the payload reports
    truncated: false while data has in fact been dropped -- the payload
    positively asserting completeness it cannot back up, which is the failure
    class this whole module exists to prevent.

    The caller uses both parts: truthiness decides the `values_summarised`
    flag, and the sum is logged, so how much was dropped is recoverable from
    the logs without re-running the query.
    """
    if isinstance(value, dict):
        return {k: _summarise(v, omitted) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_INLINE_LIST_LENGTH:
            # No empty-list guard needed: len > MAX_INLINE_LIST_LENGTH (>= 1)
            # already guarantees value[0] exists.
            omitted.append(len(value))
            return f"<list of {len(value)} {type(value[0]).__name__} values, omitted>"
        return [_summarise(v, omitted) for v in value]
    return value


def summarise_long_lists(value):
    """Replace oversized lists with a description of their shape.

    to_python recurses into lists, so `MATCH (c:Chunk) RETURN c` would return a
    full embedding vector. get_structured_schema's `sanitize` does not reach
    this path -- it only covers the library's own query family.
    """
    return _summarise(value, [])


class Neo4jForADK:
    """
    A wrapper for querying Neo4j which returns ADK-friendly responses.
    """
    _driver = None
    _neo4j_config: Neo4jConfig = None

    def __init__(self, neo4j_config: Neo4jConfig = None):
        if neo4j_config is None:
            self._neo4j_config = load_neo4j_config_from_settings()
        else:
            self._neo4j_config = neo4j_config
        self._driver = make_driver(self._neo4j_config)
        logger.debug(f"Neo4j driver initialized at {self._neo4j_config.uri}")

        # Bumped by send_query on every successful write. graph_profile's cache
        # reads this to invalidate without a round-trip. It counts in-process
        # writes only; writes from elsewhere are caught by the fingerprint.
        self.write_count = 0

    def get_driver(self):
        return self._driver

    def get_config(self):
        return self._neo4j_config

    def close(self):
        return self._driver.close()

    def send_query(self, cypher_query, parameters=None) -> Dict[str, Any]:
        # Session creation sits INSIDE the try deliberately. With it outside,
        # a driver that cannot open a session raises straight out of this
        # method instead of returning a structured error -- which in ADK
        # surfaces as an unhandled exception mid-turn rather than a message the
        # agent can react to. The counter is bumped only after result_to_adk
        # returns, so a write that fails never counts.
        session = None
        try:
            session = self._driver.session(database=self._neo4j_config.database)
            result = session.run(cypher_query, parameters or {})
            adk_result = result_to_adk(result)
            if is_write_query(cypher_query):
                self.write_count += 1
            return adk_result
        except Exception as e:
            return tool_error(str(e))
        finally:
            if session is not None:
                session.close()

    def send_read_query(
        self,
        cypher_query,
        parameters=None,
        max_rows: Optional[int] = MAX_RETURNED_ROWS,
    ) -> Dict[str, Any]:
        """Run a query read-only, timed, and with bounded row retention.

        Read-only is enforced by the *server* through default_access_mode, not
        by inspecting the query text -- text matching cannot distinguish a
        keyword from a string literal, and misses camelCase procedure calls
        like apoc.refactor.mergeNodes.

        Rows are streamed rather than materialised, so memory is bounded by
        max_rows instead of by the size of the result. Counting continues past
        max_rows up to ROW_COUNT_CEILING; beyond that the payload reports
        row_count_at_least rather than inventing an exact total.

        Pass max_rows=None to retain every row (used for internal aggregate
        queries whose results are already small).
        """
        session = None
        try:
            # Inside the try, for the same reason as send_query: a failure to
            # open the session must return a structured error, not raise.
            session = self._driver.session(
                database=self._neo4j_config.database,
                default_access_mode=READ_ACCESS,
            )
            query = Query(cypher_query, timeout=QUERY_TIMEOUT_SECONDS)
            result = session.run(query, parameters or {})

            records = []
            counted = 0
            hit_ceiling = False
            omitted: list = []
            for record in result:
                counted += 1
                if max_rows is None or len(records) < max_rows:
                    records.append(_summarise(to_python(record.data()), omitted))
                if counted >= ROW_COUNT_CEILING:
                    hit_ceiling = True
                    break

            # Rows were dropped either because the retention cap bit, or
            # because we stopped counting at the ceiling.
            truncated = (max_rows is not None and counted > len(records)) or hit_ceiling
            payload: Dict[str, Any] = {
                "records": records,
                "truncated": truncated,
                # Separate from `truncated`, which is strictly about ROWS. A
                # result can be complete in rows while an oversized list inside
                # one of them was replaced by a summary; reporting only
                # `truncated: false` there would assert a completeness the
                # payload does not have.
                "values_summarised": bool(omitted),
            }
            if hit_ceiling:
                payload["row_count_at_least"] = counted
            else:
                payload["row_count"] = counted
            if truncated:
                payload["note"] = _TRUNCATION_NOTE
            elif omitted:
                payload["note"] = _SUMMARY_NOTE
            if omitted:
                logger.debug(
                    "Summarised %d oversized list value(s) totalling %d elements",
                    len(omitted), sum(omitted))
            return tool_success("query_result", payload)
        except Exception as e:
            return tool_error(str(e))
        finally:
            if session is not None:
                session.close()

# Lazy singleton for the Neo4j client
_graphdb_singleton: Optional[Neo4jForADK] = None

def get_graphdb() -> Neo4jForADK:
    """Return a process-wide singleton instance of Neo4jForADK.

    Instantiates on first use and registers an atexit cleanup exactly once.
    """
    global _graphdb_singleton
    if _graphdb_singleton is None:
        _graphdb_singleton = Neo4jForADK()
        # Register cleanup only when the singleton is created
        atexit.register(_graphdb_singleton.close)
    return _graphdb_singleton

def close_graphdb():
    global _graphdb_singleton
    if _graphdb_singleton is not None:
        _graphdb_singleton.close()
        _graphdb_singleton = None
    