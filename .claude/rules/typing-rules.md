---
paths:
  - "src/agentic_kg/common/value_types.py"
  - "src/agentic_kg/tools/file_tools.py"
  - "src/agentic_kg/tools/kg_construction_tools.py"
---
# Typing rules

- Every typing rule (blank, a type's conversion, the majority share, the fraction rule, shape to suggested
  type) lives in `src/agentic_kg/common/value_types.py`. The hint tools and the loaders call it and never
  restate a rule, not even as an inline counter; a new entry point to the rules gets a test pinning it to
  `coerce`.
