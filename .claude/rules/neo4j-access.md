---
paths:
  - "src/agentic_kg/common/neo4j_for_adk.py"
  - "src/agentic_kg/common/pydantic_neo4j.py"
  - "src/agentic_kg/common/graph_profile.py"
  - "src/agentic_kg/tools/cypher_tools.py"
  - "src/agentic_kg/tools/construction_plan_tools.py"
  - "src/agentic_kg/tools/kg_construction_tools.py"
  - "src/agentic_kg/tools/user_goal_tools.py"
  - "tests/integration/**"
---
# Neo4j access

- Every path that hands out or uses the driver gets it from `Neo4jForADK._connection()`; a new one
  adds its own close-then-call step to `tests/integration/test_connection_recovery.py`.
- Load rows with parameterised `UNWIND` batches read client-side; never `LOAD CSV`.
- Integration tests take the `neo4j_graph` fixture: one shared container (with APOC) per run, wiped and
  its caches reset before each test by `tests/integration/shared_database.py`; never start another container.
