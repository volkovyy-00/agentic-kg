---
paths:
  - "src/agentic_kg/tools/construction_plan_tools.py"
  - "src/agentic_kg/tools/kg_construction_tools.py"
  - "src/agentic_kg/tools/join_property_check.py"
  - "src/agentic_kg/tools/node_key_check.py"
  - "src/agentic_kg/tools/reference_reachability.py"
  - "src/agentic_kg/tools/relationship_endpoints.py"
  - "src/agentic_kg/common/cypher_identifiers.py"
  - "src/agentic_kg/common/value_types.py"
  - "src/agentic_kg/coordinators/multi_agent/sub_agents/schema_proposal_agent/**"
  - "src/agentic_kg/coordinators/multi_agent/sub_agents/graph_construction_agent/**"
---
# Construction plan: checks, names, relationship ends

- The plan checks are one rule set run through `find_plan_problems` at two points: approval and the
  refinement loop. Approval must never catch a check's exception. A new check joins
  `find_plan_problems`, never a critic-side tool.
- Read a relationship's ends only through `relationship_endpoints()`.
- Labels and types go through `checked()`, and key/join/matched names through `checked_field()`, where
  they enter the plan; every name goes through `quote()` in Cypher; never `$()` dynamic labels.
- A new tool that writes plan rules calls `node_rule_name_problem` / `relationship_rule_name_problem` (or
  `required_relationship_name_problem` then `matched_property_name_problem`) first and refuses with the
  build's text, adding the rename hint only when `renamable`, or moves them into
  `check_construction_plan_consistency`.
- One check is deliberately not a plan check: `node_key_check.py` reads the key's values when a node is
  proposed and is never re-run at approval, because a plan reaches the state only through the propose
  tools and the refusal has no build-time twin. Do not move it into `find_plan_problems` or give it a
  copy of the build's text.
