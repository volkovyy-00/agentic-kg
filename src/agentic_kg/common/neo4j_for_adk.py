import os
from typing import Any, Dict, Optional
import re
import atexit
import logging

from neo4j import (
    GraphDatabase,
    Result,
)

from .config import get_settings
from .pydantic_neo4j import Neo4jConfig
from .tool_result import tool_success, tool_error

logger = logging.getLogger(__name__)

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

def sanitize(cypher_name: str) -> str:
    """Very basic string sanitization when a query param is not possible."""
    return re.sub("[.,-:$()><{}[\]'\"`\s]", '', cypher_name)

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
        # Session creation must sit INSIDE the try. It is currently outside
        # (line 148), so a driver that cannot open a session raises straight
        # out of this method instead of returning a structured error -- which
        # in ADK surfaces as an unhandled exception mid-turn rather than a
        # message the agent can react to.
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
    