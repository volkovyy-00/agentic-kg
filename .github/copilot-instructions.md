# Agentic Knowledge Graph Construction

Agentic Knowledge Graph is a multi-agent system for constructing knowledge graphs, built on Google ADK (Agent Development Kit) and designed to interact with Neo4j databases. It was originally forked from the companion project to the deeplearning.ai course on agentic knowledge graph construction, but is now developed as an independent project, not a teaching artifact.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup and Dependencies
- Install uv package manager: `pip3 install uv`
- Requires Python 3.12 (specified in .python-version and .tool-versions)
- Docker is required for integration tests
- Bootstrap the project:
  - `uv venv` -- creates virtual environment, takes ~0.05 seconds
  - `uv sync` -- installs all dependencies, takes ~6-7 seconds. NEVER CANCEL. Set timeout to 60+ minutes for safety.
  - `cp .env.example .env` -- set up environment configuration

### Environment Configuration
- Copy `.env.example` to `.env` and configure as needed
- Key variables:
  - `OPENROUTER_API_KEY=` -- required, one key covers every model call (routed via LiteLLM/OpenRouter)
  - `LLM_MODEL_CONVERSATIONAL=openai/gpt-4o-mini` -- conversational model configuration
  - `LLM_MODEL_REASONING=openai/gpt-4o` -- reasoning model configuration
  - `NEO4J_DSN=bolt://neo4j:secret@localhost:7687/neo4j` -- Neo4j connection (local or `neo4j+s://` for Aura)
  - `SOURCE_URI=./data/bom` -- where source files live (local path, `s3://`, or `https://`)
  - `LOGLEVEL=INFO` -- logging configuration

### Testing
- Unit tests (fast, no external dependencies): `uv run pytest -q` -- 253 tests, all passing. NEVER CANCEL. Set timeout to 30+ minutes.
  - Single file/test: `uv run pytest tests/unit/test_pydantic_neo4j.py -v`
- Integration tests (requires Docker): `uv run pytest -q -m integration` -- 20 tests, spins up Neo4j via Testcontainers. NEVER CANCEL. Set timeout to 120+ minutes.
  - If using colima instead of Docker Desktop: `export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` and `export TESTCONTAINERS_RYUK_DISABLED=true` first.

### Running the Agentic System
- Start the ADK web interface: `uv run adk web src/agentic_kg/coordinators/` -- starts on port 8000 by default
  - If port 8000 is in use: `uv run adk web src/agentic_kg/coordinators/ --port 8001`
  - Takes ~5 seconds to start. NEVER CANCEL. Set timeout to 60+ minutes.
  - Access at http://localhost:8000 (or configured port)
- Two available coordinators/agents:
  1. `multi_agent` - the product. Hierarchical system with five specialized sub-agents (user_intent,
     file_suggestion, schema_proposal, graph_construction, graphrag) collaborating through approval-gated phases.
  2. `single_agent` - a frozen legacy exercise carried over from the original course: a single agent that
     talks to Neo4j directly via Cypher. Not under active development.
- Web interface allows agent selection and interactive chat
- Full functionality requires LLM API keys configured in .env

## Validation
- Always manually test the ADK web interface after making changes to coordinators or agents
- Test both agent types (single_agent and multi_agent) when making significant changes
- Without LLM API keys, the system will show connection errors but the interface should still load
- ALWAYS run through at least one complete interaction scenario after making changes
- Take screenshots of the web interface when documenting UI changes

## Build and Test Validation
- The project builds successfully with `uv sync`
- Unit tests pass cleanly (253 tests); integration tests require Docker and a reachable Neo4j
- The ADK web interface loads and displays both agents correctly
- No linting or formatting tools, and no CI, are currently configured in the project

## Common Tasks

### Repository Structure
```
.
├── README.md                    # Project documentation
├── pyproject.toml              # Python project configuration
├── .env.example                # Environment variables template
├── .python-version             # Python version specification (3.12)
├── .tool-versions              # Tool versions specification
├── src/agentic_kg/             # Main source code
│   ├── coordinators/           # Agent coordinators (multi_agent, single_agent)
│   ├── agents/                 # Individual agent implementations
│   ├── common/                 # Shared utilities and configurations
│   ├── tools/                  # Tool implementations for agents
│   └── domain/                 # Domain-specific modules
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (fast, no external deps)
│   └── integration/            # Integration tests (requires Docker)
└── data/bom/                   # Sample data files for testing
```

### Key Files and Directories
- `src/agentic_kg/coordinators/multi_agent/` - Multi-agent coordinator with specialized sub-agents
- `src/agentic_kg/coordinators/single_agent/` - Simple single-agent coordinator
- `src/agentic_kg/common/config.py` - Configuration management
- `src/agentic_kg/common/neo4j_for_adk.py` - Neo4j integration for ADK
- `src/agentic_kg/tools/` - Various tools used by agents (Cypher, file operations, etc.)
- `data/bom/` - Contains sample CSV files for knowledge graph construction

### Dependencies
Key dependencies from pyproject.toml:
- `google-adk>=1.10.0` - Google Agent Development Kit (core framework)
- `neo4j>=5.28.2` - Neo4j database driver
- `neo4j-graphrag>=1.9.1` - Neo4j GraphRAG capabilities
- `litellm>=1.75.5.post1` - LLM integration layer
- `pydantic>=2.11.7` - Data validation and settings
- `testcontainers>=4.7.2` - Docker-based testing (dev dependency)

### Sample Data
The `data/bom/` directory contains bill of materials sample data:
- `assemblies.csv` - Assembly information
- `components.csv` - Component details
- `products.csv` - Product information
- `suppliers.csv` - Supplier data
- `part_supplier_mapping.csv` - Relationship mapping
- `product_reviews/` - Review data

### Expected Warnings
During startup, you may see these warnings which are normal:
- `UserWarning: Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"`
- `UserWarning: [EXPERIMENTAL] InMemoryCredentialService: This feature is experimental...`
- Font loading errors in browser console (due to network restrictions)

### Troubleshooting
- If `uv` command not found: `pip3 install uv`
- If port 8000 in use: use `--port 8001` flag with adk web command
- If integration tests fail to reach Docker: see the colima workaround in the Testing section above
- If LLM connection errors: configure API keys in .env file for full functionality
- If ADK web shows blank agents: ensure coordinators directory structure is correct