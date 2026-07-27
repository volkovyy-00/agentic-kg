# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing furniture/BOM workflow run end to end against Neo4j Aura by replacing database-host file access with a configurable source layer and driver-side CSV loading.

**Architecture:** One new module (`common/file_source.py`) owns all source-file access via `fsspec`, resolving a single `SOURCE_URI` setting into a filesystem plus root. Spreadsheet loading stops asking Neo4j to read files (`LOAD CSV FROM "file:///"`, which Aura forbids) and instead reads rows in Python and sends them as parameterised `UNWIND` batches. Model configuration moves to a single OpenRouter key with per-job model selection.

**Tech Stack:** Python 3.12, `uv`, Google ADK 1.10.0, LiteLLM, `neo4j` driver 5.28.2, `fsspec` 2024.12.0, `clevercsv` 0.8.3, pytest, testcontainers.

**Spec:** `docs/superpowers/specs/2026-07-27-foundation-design.md` — authoritative and fact-checked. Read it before starting.

## Global Constraints

- **Do not upgrade `google-adk`.** It stays at 1.10.0. Bounding the constraint to `>=1.10,<2` is in scope; upgrading is not.
- **Do not add embedding configuration.** Deferred to sub-project 2.
- **Do not change existing graph behaviour:** `MERGE` semantics (idempotent), uniqueness constraints created before loading, nodes loaded before relationships, construction-plan schema unchanged, CSV values remain strings.
- **Labels and relationship types are interpolated after `is_symbol()` validation** — never passed as Cypher `$()` dynamic labels. Dynamic labels plan as `Merge` instead of `MergeUniqueNode`, i.e. no index usage, until Neo4j 2025.11.
- **Baseline is 15 passing unit tests.** Keep them green at every commit.
- **No linter or formatter is configured.** Do not add one.
- Run tests with `uv run pytest -q` (unit) and `uv run pytest -q -m integration` (integration, needs Docker).
- Commit after every task. Work on branch `foundation-file-sources-and-models`.

---

## File Structure

**New files**

| Path | Responsibility |
|---|---|
| `src/agentic_kg/common/file_source.py` | Resolve `SOURCE_URI` → `(fs, root)`; list, open, exists, and native-path lookup by relative name. The only place that knows where files live. |
| `src/agentic_kg/common/csv_reader.py` | Turn a source-relative CSV into `(header, batch)` pairs. No database, no knowledge of graphs. |
| `src/agentic_kg/coordinators/multi_agent/names.py` | Holds the coordinator's agent name as a constant, breaking the import cycle that forced `finished()` to use private API. |
| `tests/unit/test_imports.py` | Imports every module in the package. |
| `tests/unit/test_file_source.py` | `file_source` against `memory://`. |
| `tests/unit/test_csv_reader.py` | Batching with no database. |
| `tests/integration/test_csv_loading_integration.py` | Loads `data/bom` into a container Neo4j and asserts counts. |

**Modified files**

| Path | Change |
|---|---|
| `src/agentic_kg/common/config.py` | New settings; `reset_settings()` for tests; `validate_env()` rewritten and actually invoked |
| `src/agentic_kg/common/llm_catalog.py` | Per-kind cache; honour settings; derive `openrouter/` prefix |
| `src/agentic_kg/common/neo4j_for_adk.py` | No change (reference only — `is_symbol` lives here) |
| `src/agentic_kg/tools/file_tools.py` | Thin callers; delete `import_markdown_file`; fix `approve_suggested_files` |
| `src/agentic_kg/tools/cypher_tools.py` | Delete `get_neo4j_import_dir()` |
| `src/agentic_kg/tools/kg_construction_tools.py` | Driver-side loading; result collection; delete `construct_node`/`construct_relationship` |
| `src/agentic_kg/tools/adk_tools.py` | `finished` → `make_finished(parent_agent_name)` |
| `src/agentic_kg/tools/toolset.py` | Delete the `tkinter` import |
| `src/agentic_kg/tools/user_intent_tools.py` | Fix the `tool_result` import path |
| `src/agentic_kg/coordinators/multi_agent/agent.py` | Source-location tool; model kinds; import `validate_env` |
| 5 sub-agent `variants.py`/`agent.py` files | `make_finished(...)`; model kinds |
| `pyproject.toml` | Bound `google-adk`; declare `fsspec`, `aiohttp` |
| `.env.example`, `README.md` | New settings; remove the import-directory setup step |

**Deleted**

- `src/agentic_kg/agents/file_suggestion_agent/` (whole package) — unreachable, and `variants.py` references eight undefined names while `agent.py` uses `LlmKind` without importing it. Superseded by `coordinators/multi_agent/sub_agents/file_suggestion_agent/`. See Task 1 note.
- `import_markdown_file`, `construct_node`, `construct_relationship`, `get_neo4j_import_dir`

---

## Task 1: Repair the module graph and lock it with a smoke test

**Why first:** every later task adds modules. Without a working import graph and a test that guards it, breakage stays invisible — exactly how the current three defects survived.

**Files:**
- Create: `tests/unit/test_imports.py`
- Modify: `src/agentic_kg/tools/toolset.py:1`
- Modify: `src/agentic_kg/tools/user_intent_tools.py:11`
- Delete: `src/agentic_kg/agents/file_suggestion_agent/` (entire directory)

**Interfaces:**
- Consumes: nothing
- Produces: a green `uv run pytest -q` with 16 tests, and an importable package

> **Note on the deletion.** The spec's components table says "resolve undefined `file_toolset`". Investigation showed that understates it: `agents/file_suggestion_agent/variants.py` references `file_toolset`, `finished`, `get_approved_user_goal`, `list_import_files`, `sample_file`, `search_file`, `set_suggested_files` and `approve_suggested_files` while importing only `get_approved_user_intent`; and `agents/file_suggestion_agent/agent.py:12` calls `get_llm(LlmKind.reasoning)` without importing `LlmKind`. Nothing in the codebase imports this package. Repairing it means inventing eight bindings for a module that duplicates a working one. Deleting is consistent with the spec's own stated reason for deleting `import_markdown_file`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_imports.py`:

```python
"""Every module in the package must import cleanly.

This guards against the class of defect where a module is broken for months
because nothing imports it. Both coordinators import fine today; the damage
was in modules nothing exercised.
"""
import importlib
import pkgutil

import agentic_kg


def test_every_module_imports():
    failures = []
    for module_info in pkgutil.walk_packages(agentic_kg.__path__, "agentic_kg."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001 - we want to report every failure
            failures.append(f"{module_info.name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/unit/test_imports.py -v`

Expected: FAIL, listing seven modules with `ModuleNotFoundError: No module named '_tkinter'` (on a Python built without Tk) plus `NameError: name 'file_toolset' is not defined`.

> If your Python *does* have Tk, the `tkinter` line imports fine and you will see only the `file_toolset` failures. Delete the import anyway — it is unused, and the project must not depend on a GUI toolkit being present.

- [ ] **Step 3: Delete the stray tkinter import**

In `src/agentic_kg/tools/toolset.py`, delete line 1 entirely:

```python
from tkinter import Label
```

The file must now begin:

```python
from typing import Callable, TypedDict, List

class ToolSet(TypedDict):
```

- [ ] **Step 4: Fix the wrong import path**

In `src/agentic_kg/tools/user_intent_tools.py:11`, change:

```python
from .tool_result import tool_success, tool_error
```

to:

```python
from agentic_kg.common.tool_result import tool_success, tool_error
```

(The relative import resolves to `agentic_kg/tools/tool_result.py`, which does not exist. The module is `agentic_kg/common/tool_result.py`.)

- [ ] **Step 5: Delete the abandoned agent package**

```bash
git rm -r src/agentic_kg/agents/file_suggestion_agent
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`

Expected: `16 passed` (15 baseline + the new smoke test).

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_imports.py src/agentic_kg/tools/toolset.py src/agentic_kg/tools/user_intent_tools.py
git commit -m "fix: repair module import graph and guard it with a smoke test

Three defects broke seven modules: an unused tkinter import, a relative
import resolving to a non-existent tools/tool_result.py, and an
unreachable agents/file_suggestion_agent package referencing eight
undefined names. The first two are fixed; the third is deleted, being
superseded by the multi_agent sub-agent of the same name."
```

---

## Task 2: Bound and declare dependencies

**Files:**
- Modify: `pyproject.toml:10-18`

**Interfaces:**
- Consumes: nothing
- Produces: `fsspec` and `aiohttp` as declared direct dependencies, safe for Task 3 to build on

- [ ] **Step 1: Verify the current state**

Run: `grep -n "google-adk\|fsspec\|aiohttp" pyproject.toml`

Expected: only `"google-adk>=1.10.0",` — an unbounded constraint. `fsspec` and `aiohttp` are absent (they arrive transitively).

- [ ] **Step 2: Edit the dependency list**

In `pyproject.toml`, replace the `dependencies` list with:

```toml
dependencies = [
    "aiohttp>=3.9",
    "clevercsv[full]>=0.8.3",
    "fsspec>=2024.9,<2025.0.0",
    "google-adk>=1.10,<2",
    "litellm>=1.75.5.post1,<1.82.7",
    "neo4j>=5.28.2",
    "neo4j-graphrag>=1.9.1",
    "pydantic>=2.11.7",
    "pydantic-settings>=2.10.1",
]
```

Rationale for each change:
- `google-adk>=1.10,<2` — the latest published version is 2.5.0. The old unbounded `>=1.10.0` would cross a major version on any fresh lock while the lockfile still claims 1.10.0.
- `fsspec` — Task 3 makes it a first-class dependency. It currently arrives via `neo4j-graphrag`, whose own ceiling is `<2025.0.0`; matching that ceiling avoids a resolver conflict.
- `aiohttp` — the only reason `https://` sources work. It currently arrives via the LLM stack.

- [ ] **Step 3: Re-lock and verify nothing moved**

Run:
```bash
uv lock
uv sync
uv run pytest -q
```

Expected: `16 passed`. If `uv lock` wants to change `google-adk` away from 1.10.0, stop — the constraint is wrong.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: bound google-adk below 2.0 and declare fsspec/aiohttp

google-adk>=1.10.0 was unbounded and would resolve to 2.5.0 on a fresh
lock. fsspec and aiohttp are used directly but were inherited as
transitive dependencies."
```

---

## Task 3: Add source settings and a test reset hook

**Files:**
- Modify: `src/agentic_kg/common/config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `agentic_kgSettings.source_uri: Optional[str]`
  - `agentic_kgSettings.openrouter_api_key: Optional[str]`
  - `agentic_kgSettings.llm_model_conversational: str`
  - `agentic_kgSettings.llm_model_reasoning: str`
  - `reset_settings() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_config.py`:

```python
from agentic_kg.common.config import get_settings, reset_settings


def test_source_uri_defaults_to_none(monkeypatch):
    monkeypatch.delenv("SOURCE_URI", raising=False)
    reset_settings()
    assert get_settings().source_uri is None


def test_source_uri_reads_from_environment(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", "memory://somewhere")
    reset_settings()
    assert get_settings().source_uri == "memory://somewhere"


def test_model_settings_have_defaults(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_REASONING", raising=False)
    monkeypatch.delenv("LLM_MODEL_CONVERSATIONAL", raising=False)
    reset_settings()
    settings = get_settings()
    assert settings.llm_model_reasoning
    assert settings.llm_model_conversational


def test_reset_settings_forces_reload(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", "memory://first")
    reset_settings()
    assert get_settings().source_uri == "memory://first"
    monkeypatch.setenv("SOURCE_URI", "memory://second")
    reset_settings()
    assert get_settings().source_uri == "memory://second"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/unit/test_config.py -v`

Expected: FAIL with `ImportError: cannot import name 'reset_settings'`.

- [ ] **Step 3: Add the settings and the reset hook**

In `src/agentic_kg/common/config.py`, inside `agentic_kgSettings`, add after the existing `llm_base_url` field:

```python
    # Source file location (local path, bucket URL, or http(s) URL).
    # Relative local paths are anchored to the repository root, not the CWD.
    source_uri: Optional[str] = Field(default=None)

    # OpenRouter is the single provider for chat, extraction and embeddings.
    openrouter_api_key: Optional[str] = Field(default=None)

    # Per-job models, stored in OpenRouter's spelling (e.g. "openai/gpt-4o").
    # The "openrouter/" prefix LiteLLM wants is derived, not configured.
    llm_model_conversational: str = Field(default="openai/gpt-4o-mini")
    llm_model_reasoning: str = Field(default="openai/gpt-4o")
```

Then add, immediately after `get_settings()`:

```python
def reset_settings() -> None:
    """Discard the cached settings so the next get_settings() re-reads the environment.

    Intended for tests. Production code should never need this.
    """
    global _settings
    _settings = None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_config.py -v`

Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: `20 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/config.py tests/unit/test_config.py
git commit -m "feat: add source and per-job model settings"
```

---

## Task 4: Build the file_source seam

**Files:**
- Create: `src/agentic_kg/common/file_source.py`
- Create: `tests/unit/test_file_source.py`

**Interfaces:**
- Consumes: `agentic_kg.common.config.get_settings`
- Produces:
  - `SourceError(Exception)`
  - `get_source_fs() -> tuple[AbstractFileSystem, str]`
  - `get_source_root() -> str`
  - `source_path(relative_path: str) -> str` — filesystem-native absolute path
  - `list_source_files() -> list[str]` — relative names, recursive, sorted
  - `open_source(relative_path: str, mode: str = "r", **kwargs)` — file handle
  - `source_exists(relative_path: str) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_file_source.py`:

```python
import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common import file_source


@pytest.fixture
def memory_source(monkeypatch):
    """A memory:// source populated with two files, one in a subdirectory.

    The fsspec memory filesystem is a process-global singleton, so the store
    must be cleared between tests or files leak across cases.
    """
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/top.csv", "w") as handle:
        handle.write("a,b\n1,2\n")
    with fs.open("/src/nested/deep.md", "w") as handle:
        handle.write("# hello\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_lists_relative_names_recursively(memory_source):
    assert file_source.list_source_files() == ["nested/deep.md", "top.csv"]


def test_opens_by_relative_name(memory_source):
    with file_source.open_source("top.csv") as handle:
        assert handle.read() == "a,b\n1,2\n"


def test_opens_file_in_subdirectory(memory_source):
    with file_source.open_source("nested/deep.md") as handle:
        assert handle.read() == "# hello\n"


def test_source_exists(memory_source):
    assert file_source.source_exists("top.csv") is True
    assert file_source.source_exists("absent.csv") is False


def test_opening_a_missing_file_raises(memory_source):
    with pytest.raises(FileNotFoundError):
        file_source.open_source("absent.csv")


def test_source_path_is_native_not_relative(memory_source):
    assert file_source.source_path("top.csv") == "/src/top.csv"


def test_unset_source_uri_raises_source_error(monkeypatch):
    monkeypatch.delenv("SOURCE_URI", raising=False)
    reset_settings()
    with pytest.raises(file_source.SourceError, match="SOURCE_URI"):
        file_source.get_source_fs()


def test_relative_path_anchors_to_repo_root_not_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCE_URI", "./data/bom")
    reset_settings()
    monkeypatch.chdir(tmp_path)
    _fs, root = file_source.get_source_fs()
    assert root.endswith("/data/bom")
    assert str(tmp_path) not in root


def test_uninstalled_scheme_raises_source_error(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", "s3://some-bucket/prefix")
    reset_settings()
    with pytest.raises(file_source.SourceError, match="s3"):
        file_source.get_source_fs()
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_file_source.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_kg.common.file_source'`.

- [ ] **Step 3: Write the implementation**

Create `src/agentic_kg/common/file_source.py`:

```python
"""Single owner of source-file access.

Every component that reads a source file goes through this module. It resolves
the configured SOURCE_URI into an fsspec filesystem plus a root path, and
exposes listing, opening and existence checks by *relative* name.

The relative-name convention matters: construction plans record files as
"assemblies.csv", not as absolute locations, so a plan built against a local
folder still works when the same files move elsewhere. This module is the one
place that knows the difference.
"""
import logging
from pathlib import Path
from typing import Any, Tuple

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from .config import get_settings

logger = logging.getLogger(__name__)

# .../<repo>/src/agentic_kg/common/file_source.py -> .../<repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]


class SourceError(Exception):
    """The configured source location is unusable."""


def _anchor(uri: str) -> str:
    """Absolutise a relative local path against the repository root.

    fsspec resolves relative paths against the process working directory, and
    `adk web` makes no promise about what that is. Anything with a scheme, and
    anything already absolute, passes through untouched.
    """
    if "://" in uri:
        return uri
    path = Path(uri)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def get_source_fs() -> Tuple[AbstractFileSystem, str]:
    """Return the configured filesystem and its root path.

    Raises:
        SourceError: if SOURCE_URI is unset, or names a scheme whose backing
            package is not installed (e.g. s3:// without s3fs).
    """
    uri = get_settings().source_uri
    if not uri:
        raise SourceError(
            "SOURCE_URI is not set. Point it at a folder of source files, "
            "for example SOURCE_URI=./data/bom"
        )
    try:
        fs, root = url_to_fs(_anchor(uri))
    except ImportError as exc:
        raise SourceError(
            f"SOURCE_URI '{uri}' needs a package that is not installed: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - fsspec raises a variety of types
        raise SourceError(f"SOURCE_URI '{uri}' could not be resolved: {exc}") from exc
    return fs, root.rstrip("/")


def get_source_root() -> str:
    """Return the resolved root path, for display to a user."""
    _fs, root = get_source_fs()
    return root


def source_path(relative_path: str) -> str:
    """Return the filesystem-native absolute path for a relative name."""
    _fs, root = get_source_fs()
    return f"{root}/{relative_path}"


def list_source_files() -> list[str]:
    """List every file under the source root, as sorted relative names.

    Raises:
        SourceError: if the root does not exist.
    """
    fs, root = get_source_fs()
    if not fs.exists(root):
        raise SourceError(f"Source location does not exist: {root}")
    prefix_length = len(root) + 1
    return sorted(found[prefix_length:] for found in fs.find(root))


def source_exists(relative_path: str) -> bool:
    """Whether a file exists at the given relative name."""
    fs, _root = get_source_fs()
    return bool(fs.exists(source_path(relative_path)))


def open_source(relative_path: str, mode: str = "r", **kwargs: Any):
    """Open a source file by relative name.

    Text mode is the default because clevercsv requires an iterable of str.

    Raises:
        FileNotFoundError: if the file does not exist.
        SourceError: if the source location is misconfigured.
    """
    fs, _root = get_source_fs()
    full_path = source_path(relative_path)
    if not fs.exists(full_path):
        raise FileNotFoundError(f"No such source file: {relative_path}")
    if "b" not in mode:
        kwargs.setdefault("newline", "")
    return fs.open(full_path, mode, **kwargs)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_file_source.py -v`

Expected: PASS (9 tests).

If `test_uninstalled_scheme_raises_source_error` fails because `s3fs` happens to be installed in your environment, that is a genuine environment difference — confirm with `uv run python -c "import s3fs"`. If it is installed, change the test's URI to a scheme with no backing package, such as `"abfs://container/path"`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: `29 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/file_source.py tests/unit/test_file_source.py
git commit -m "feat: add fsspec-backed file_source seam

One owner for source-file access, resolving SOURCE_URI to an (fs, root)
pair. Relative URIs anchor to the repo root rather than the process CWD,
because adk web makes no promise about the working directory."
```

---

## Task 5: Build the CSV reader

**Files:**
- Create: `src/agentic_kg/common/csv_reader.py`
- Create: `tests/unit/test_csv_reader.py`

**Interfaces:**
- Consumes: `file_source.open_source`
- Produces: `read_csv_batches(relative_path: str, batch_size: int = 1000) -> Iterator[tuple[list[str], list[dict[str, str]]]]`

Batching is deliberately separate from sending so it can be tested without a database.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_csv_reader.py`:

```python
import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.csv_reader import read_csv_batches


@pytest.fixture
def csv_source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/csv/people.csv", "w") as handle:
        handle.write("id,name\n1,Ada\n2,Grace\n3,Alan\n")
    with fs.open("/csv/semicolons.csv", "w") as handle:
        handle.write("id;name\n1;Ada\n")
    with fs.open("/csv/ragged.csv", "w") as handle:
        handle.write("id,name,note\n1,Ada\n")
    with fs.open("/csv/headeronly.csv", "w") as handle:
        handle.write("id,name\n")
    monkeypatch.setenv("SOURCE_URI", "memory://csv")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_yields_header_and_rows_as_dicts(csv_source):
    batches = list(read_csv_batches("people.csv"))
    assert len(batches) == 1
    header, rows = batches[0]
    assert header == ["id", "name"]
    assert rows == [
        {"id": "1", "name": "Ada"},
        {"id": "2", "name": "Grace"},
        {"id": "3", "name": "Alan"},
    ]


def test_splits_into_batches(csv_source):
    batches = list(read_csv_batches("people.csv", batch_size=2))
    assert [len(rows) for _header, rows in batches] == [2, 1]


def test_detects_non_comma_separator(csv_source):
    _header, rows = next(iter(read_csv_batches("semicolons.csv")))
    assert rows == [{"id": "1", "name": "Ada"}]


def test_short_row_omits_missing_column(csv_source):
    _header, rows = next(iter(read_csv_batches("ragged.csv")))
    assert rows == [{"id": "1", "name": "Ada"}]
    assert "note" not in rows[0]


def test_header_only_file_yields_nothing(csv_source):
    assert list(read_csv_batches("headeronly.csv")) == []


def test_values_stay_strings(csv_source):
    _header, rows = next(iter(read_csv_batches("people.csv")))
    assert all(isinstance(value, str) for value in rows[0].values())
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_csv_reader.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_kg.common.csv_reader'`.

- [ ] **Step 3: Write the implementation**

Create `src/agentic_kg/common/csv_reader.py`:

```python
"""Read source CSVs into batches of row dictionaries.

Kept separate from anything that talks to Neo4j so that batching can be tested
without a database. Values stay strings, exactly as LOAD CSV produced them —
typed fields are deliberately out of scope.
"""
import logging
from typing import Iterator, List, Tuple

import clevercsv

from .file_source import open_source

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000
_SNIFF_BYTES = 2048


def _make_reader(handle, relative_path: str):
    """Build a clevercsv reader, sniffing the dialect where possible."""
    sample = handle.read(_SNIFF_BYTES)
    handle.seek(0)
    dialect = None
    try:
        dialect = clevercsv.Sniffer().sniff(sample)
    except clevercsv.Error:
        logger.warning("Could not sniff CSV dialect for %s; using default", relative_path)
    # sniff() returns a degenerate SimpleDialect('', '', '') for empty or
    # trivial samples rather than raising, so the except clause above does not
    # fire for those. Check the delimiter explicitly.
    if dialect is None or not getattr(dialect, "delimiter", ""):
        return clevercsv.reader(handle)
    return clevercsv.reader(handle, dialect)


def read_csv_batches(
    relative_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Tuple[List[str], List[dict]]]:
    """Yield (header, batch) pairs for a source-relative CSV file.

    Rows shorter than the header omit the missing keys rather than padding with
    empty strings, which keeps the resulting graph free of meaningless blanks.

    Args:
        relative_path: file name relative to the source root
        batch_size: rows per batch

    Yields:
        (header, rows) where rows is a list of dicts of column name to string
    """
    with open_source(relative_path, "r") as handle:
        reader = _make_reader(handle, relative_path)
        header = next(reader, [])
        if not header:
            return
        batch: List[dict] = []
        for row in reader:
            batch.append({key: value for key, value in zip(header, row)})
            if len(batch) >= batch_size:
                yield header, batch
                batch = []
        if batch:
            yield header, batch
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_csv_reader.py -v`

Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: `35 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/csv_reader.py tests/unit/test_csv_reader.py
git commit -m "feat: add CSV batch reader independent of the database"
```

---

## Task 6: Convert the file tools to file_source

**Files:**
- Modify: `src/agentic_kg/tools/file_tools.py`
- Modify: `src/agentic_kg/tools/cypher_tools.py:205-216` (delete `get_neo4j_import_dir`)

**Interfaces:**
- Consumes: `file_source.list_source_files`, `open_source`, `get_source_root`, `SourceError`
- Produces: `get_source_location(tool_context) -> dict` (replaces the agent-facing import-directory tool); `list_import_files`, `sample_file`, `search_file`, `search_csv_file`, `approve_suggested_files` all unchanged in signature

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_file_tools.py`:

```python
import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import file_tools


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext — the file tools only use .state."""

    def __init__(self):
        self.state = {}


@pytest.fixture
def memory_source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/people.csv", "w") as handle:
        handle.write("id,name\n1,Ada\n2,Grace\n")
    with fs.open("/src/notes/readme.md", "w") as handle:
        handle.write("# Title\nAda appears here.\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_list_import_files_returns_relative_names(memory_source):
    context = FakeToolContext()
    result = file_tools.list_import_files(context)
    assert result["status"] == "success"
    assert result["all_available_files"] == ["notes/readme.md", "people.csv"]


def test_sample_file_reads_content(memory_source):
    context = FakeToolContext()
    result = file_tools.sample_file("people.csv", context)
    assert result["status"] == "success"
    assert "Ada" in result["sample"]["content"]


def test_sample_file_missing_returns_error(memory_source):
    context = FakeToolContext()
    result = file_tools.sample_file("nope.csv", context)
    assert result["status"] == "error"


def test_search_file_finds_matching_line(memory_source):
    result = file_tools.search_file("notes/readme.md", "ada")
    assert result["status"] == "success"
    assert result["search_results"]["metadata"]["lines_found"] == 1


def test_approve_suggested_files_returns_a_result(memory_source):
    context = FakeToolContext()
    file_tools.set_suggested_files(["people.csv"], context)
    result = file_tools.approve_suggested_files(context)
    assert result is not None
    assert result["status"] == "success"
    assert context.state["approved_file_list"] == ["people.csv"]


def test_approve_without_suggestions_returns_error(memory_source):
    context = FakeToolContext()
    result = file_tools.approve_suggested_files(context)
    assert result["status"] == "error"


def test_unset_source_uri_surfaces_as_tool_error(monkeypatch):
    monkeypatch.delenv("SOURCE_URI", raising=False)
    reset_settings()
    result = file_tools.list_import_files(FakeToolContext())
    assert result["status"] == "error"
    assert "SOURCE_URI" in result["error_message"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_file_tools.py -v`

Expected: FAIL — the tools still call `get_neo4j_import_dir()`, which hits the database. Failures will be connection or permission errors, and `test_approve_suggested_files_returns_a_result` will fail on `result is not None`.

- [ ] **Step 3: Rewrite the imports and helpers in file_tools.py**

Replace the import block at the top of `src/agentic_kg/tools/file_tools.py` (lines 1-18) with:

```python
import logging

import clevercsv
from itertools import islice

from google.adk.tools import ToolContext
from typing import Dict, Any, List

from agentic_kg.common.tool_result import tool_success, tool_error
from agentic_kg.common.file_source import (
    SourceError,
    get_source_root,
    list_source_files,
    open_source,
    source_exists,
)

logger = logging.getLogger(__name__)

ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_file_list"
APPROVED_FILES = "approved_file_list"
```

- [ ] **Step 4: Replace `list_import_files`**

```python
def list_import_files(tool_context: ToolContext) -> dict:
    """Lists files available for knowledge graph construction.

    All names are relative to the configured source location.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, an
              'all_available_files' key with a list of relative file names.
    """
    try:
        file_names = list_source_files()
    except SourceError as exc:
        return tool_error(str(exc))

    tool_context.state[ALL_AVAILABLE_FILES] = file_names
    return tool_success(ALL_AVAILABLE_FILES, file_names)
```

- [ ] **Step 5: Add the source-location tool and fix `approve_suggested_files`**

Add a new tool (this replaces `get_neo4j_import_dir` in the coordinator's tool list):

```python
def get_source_location(tool_context: ToolContext) -> Dict[str, Any]:
    """Reports where the system is reading source files from."""
    try:
        return tool_success("source_location", get_source_root())
    except SourceError as exc:
        return tool_error(str(exc))
```

Replace `approve_suggested_files` entirely — it currently falls off the end and returns `None`:

```python
def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Approves the suggested files for further processing."""
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("Current files have not been set. Take no action other than to inform user.")

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])
```

- [ ] **Step 6: Replace the three readers**

Replace `sample_file` with:

```python
def sample_file(file_path: str, tool_context: ToolContext) -> dict:
    """Samples a file by reading up to 100 lines as text.

    Args:
      file_path: file to sample, relative to the source location
      tool_context: ToolContext object

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'sample' key with
              metadata and content.
    """
    suffix = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    mimetype = {"csv": "text/csv", "md": "text/markdown"}.get(suffix, "text/plain")

    result = {
        "metadata": {"path": file_path, "mimetype": mimetype},
        "annotations": [],
    }

    try:
        with open_source(file_path, "r") as handle:
            result["content"] = "".join(islice(handle, 100))
    except SourceError as exc:
        return tool_error(str(exc))
    except FileNotFoundError:
        return tool_error(f"Path does not exist: {file_path}")
    except Exception as exc:  # noqa: BLE001 - report decoding failures to the agent
        return tool_error(f"Error reading or processing file {file_path}: {exc}")

    return tool_success("sample", result)
```

Replace the body of `search_csv_file` so that it opens through `open_source`. Keep its signature and return shape. The two `open(p, ...)` calls become:

```python
        with open_source(file_path, "r") as csvfile:
```

and delete the `get_neo4j_import_dir` / `Path` resolution block at the top of the function, replacing it with:

```python
    try:
        if not source_exists(file_path):
            return tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return tool_error(str(exc))
```

Replace `search_file` with:

```python
def search_file(file_path: str, query: str) -> dict:
    """Searches any text file for lines containing the query string, case-insensitively.

    Args:
      file_path: path relative to the source location
      query: the string to search for

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'search_results'
              key with 'matching_lines' and metadata.
    """
    try:
        if not source_exists(file_path):
            return tool_error(f"File does not exist: {file_path}")
    except SourceError as exc:
        return tool_error(str(exc))

    if not query:
        return tool_success(SEARCH_RESULTS, {
            "metadata": {"path": file_path, "query": query, "lines_found": 0},
            "matching_lines": [],
        })

    matching_lines = []
    search_query = query.lower()
    try:
        with open_source(file_path, "r") as handle:
            for line_number, line in enumerate(handle, 1):
                if search_query in line.lower():
                    matching_lines.append({
                        "line_number": line_number,
                        "content": line.strip(),
                    })
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Error reading or searching file {file_path}: {exc}")

    return tool_success(SEARCH_RESULTS, {
        "metadata": {
            "path": file_path,
            "query": query,
            "lines_found": len(matching_lines),
        },
        "matching_lines": matching_lines,
    })
```

- [ ] **Step 7: Delete `import_markdown_file`**

Delete the entire `import_markdown_file` function (the last function in the file). It imports `agentic_kg.sub_agents.cypher_agent.tools`, a module path that does not exist.

- [ ] **Step 8: Delete `get_neo4j_import_dir`**

In `src/agentic_kg/tools/cypher_tools.py`, delete the whole `get_neo4j_import_dir()` function (lines 205-216).

Run `grep -rn "get_neo4j_import_dir" src/ tests/` and fix every remaining reference. Expected remaining reference: `coordinators/multi_agent/agent.py` — handled in Task 8.

- [ ] **Step 9: Run the tests**

Run: `uv run pytest tests/unit/test_file_tools.py -v`

Expected: PASS (7 tests).

- [ ] **Step 10: Run the full suite**

Run: `uv run pytest -q`

Expected: `42 passed`.

- [ ] **Step 11: Commit**

```bash
git add src/agentic_kg/tools/file_tools.py src/agentic_kg/tools/cypher_tools.py tests/unit/test_file_tools.py
git commit -m "refactor: read source files through file_source

Deletes get_neo4j_import_dir (dbms.listConfig is forbidden on Aura) and
the dead import_markdown_file. Fixes approve_suggested_files, which
returned None where every other tool returns a ToolResult."
```

---

## Task 7: Driver-side CSV loading

**Files:**
- Modify: `src/agentic_kg/tools/kg_construction_tools.py` (whole file)
- Create: `tests/unit/test_kg_construction_tools.py`

**Interfaces:**
- Consumes: `csv_reader.read_csv_batches`, `neo4j_for_adk.is_symbol`, `cypher_tools.create_uniqueness_constraint`
- Produces: `load_nodes_from_csv(...)`, `import_nodes(rule)`, `import_relationships(rule)`, `construct_domain_graph(plan)`, `build_graph_from_construction_rules(tool_context)` — all returning `ToolResult` shapes

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_kg_construction_tools.py`:

```python
"""Unit tests for query construction and result collection.

The database is faked: these tests assert what Cypher gets built and how
failures propagate, without a Neo4j instance.
"""
import pytest

from agentic_kg.tools import kg_construction_tools as kg


class FakeGraphDb:
    def __init__(self, responses=None):
        self.queries = []
        self.responses = responses or []

    def send_query(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        if self.responses:
            return self.responses.pop(0)
        return {"status": "success", "records": []}


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeGraphDb()
    monkeypatch.setattr(kg, "graphdb", db)
    return db


@pytest.fixture
def one_batch(monkeypatch):
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "name"], [{"id": "1", "name": "Ada"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)


def test_node_query_interpolates_label_not_dynamic(fake_db, one_batch):
    kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    query, params = fake_db.queries[0]
    assert "MERGE (n:Person" in query
    assert "$($label)" not in query
    assert params["rows"] == [{"id": "1", "name": "Ada"}]


def test_node_query_uses_unwind_not_load_csv(fake_db, one_batch):
    kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    query, _params = fake_db.queries[0]
    assert query.strip().startswith("UNWIND $rows AS row")
    assert "LOAD CSV" not in query
    assert "file:///" not in query


def test_invalid_label_is_rejected_before_any_query(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Not A Label", "id", ["name"])
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_invalid_column_is_rejected_before_any_query(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Person", "not a column", ["name"])
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_relationship_columns_are_validated(fake_db, one_batch):
    rule = {
        "source_file": "knows.csv",
        "relationship_type": "KNOWS",
        "from_node_label": "Person",
        "from_node_column": "bad column",
        "to_node_label": "Person",
        "to_node_column": "to_id",
        "properties": [],
    }
    result = kg.import_relationships(rule)
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_batch_failure_reports_rows_committed(monkeypatch, one_batch):
    db = FakeGraphDb(responses=[{"status": "error", "error_message": "boom"}])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "error"
    assert "people.csv" in result["error_message"]
    assert "boom" in result["error_message"]


def test_construct_domain_graph_reports_failure(monkeypatch):
    monkeypatch.setattr(kg, "import_nodes", lambda rule: {"status": "error", "error_message": "nope"})
    monkeypatch.setattr(kg, "import_relationships", lambda rule: {"status": "success"})
    plan = {"Person": {"construction_type": "node", "label": "Person"}}
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "error", "a failed import must not be reported as success"


def test_construct_domain_graph_loads_nodes_before_relationships(monkeypatch):
    order = []
    monkeypatch.setattr(kg, "import_nodes", lambda rule: order.append("node") or {"status": "success"})
    monkeypatch.setattr(kg, "import_relationships", lambda rule: order.append("rel") or {"status": "success"})
    plan = {
        "KNOWS": {"construction_type": "relationship", "relationship_type": "KNOWS"},
        "Person": {"construction_type": "node", "label": "Person"},
    }
    kg.construct_domain_graph(plan)
    assert order == ["node", "rel"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_kg_construction_tools.py -v`

Expected: FAIL — `read_csv_batches` is not yet imported into the module, and the queries still use `LOAD CSV`.

- [ ] **Step 3: Rewrite kg_construction_tools.py**

Replace the entire contents of `src/agentic_kg/tools/kg_construction_tools.py`:

```python
"""Build the domain graph from approved construction rules.

Rows are read in Python and sent as parameterised UNWIND batches rather than
asking Neo4j to read files itself. Aura forbids LOAD CSV FROM "file:///", and
client-side reading works identically against a local instance.

Labels and relationship types are interpolated into the query text after
is_symbol() validation rather than passed as Cypher dynamic labels: dynamic
labels plan as Merge instead of MergeUniqueNode, so they cannot use the
uniqueness index and every row triggers an all-nodes scan.
"""
import logging

from google.adk.tools import ToolContext
from typing import Any, Dict, List

from agentic_kg.common.csv_reader import read_csv_batches
from agentic_kg.common.file_source import SourceError
from agentic_kg.common.neo4j_for_adk import get_graphdb, is_symbol
from agentic_kg.common.tool_result import tool_error, tool_success
from agentic_kg.tools.cypher_tools import create_uniqueness_constraint

logger = logging.getLogger(__name__)

graphdb = get_graphdb()

APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"


class InvalidIdentifier(ValueError):
    """A label, relationship type or column name failed validation."""


def _checked(kind: str, value: str) -> str:
    """Validate an identifier destined for interpolation into Cypher."""
    if not value or not is_symbol(value):
        raise InvalidIdentifier(
            f"Invalid {kind}: '{value}'. It cannot contain spaces or be a Cypher keyword."
        )
    return value


def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: List[str],
) -> Dict[str, Any]:
    """Load nodes from a source CSV in batches."""
    try:
        label = _checked("label", label)
        unique_column_name = _checked("column name", unique_column_name)
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    query = f"""UNWIND $rows AS row
    MERGE (n:{label} {{ {unique_column_name} : row[$unique_column_name] }})
    FOREACH (k IN $properties | SET n[k] = row[k])
    """

    rows_committed = 0
    try:
        for _header, batch in read_csv_batches(source_file):
            result = graphdb.send_query(query, {
                "rows": batch,
                "unique_column_name": unique_column_name,
                "properties": properties,
            })
            if result["status"] == "error":
                return tool_error(
                    f"{source_file}: load failed after {rows_committed} rows committed "
                    f"(the failing batch was rolled back): {result['error_message']}"
                )
            rows_committed += len(batch)
    except (SourceError, FileNotFoundError) as exc:
        return tool_error(f"{source_file}: {exc}")

    return tool_success("rows_loaded", {"source_file": source_file, "rows": rows_committed})


def import_nodes(node_construction: dict) -> Dict[str, Any]:
    """Import nodes as defined by a node construction rule."""
    uniqueness_result = create_uniqueness_constraint(
        node_construction["label"],
        node_construction["unique_column_name"],
    )
    if uniqueness_result["status"] == "error":
        return uniqueness_result

    return load_nodes_from_csv(
        node_construction["source_file"],
        node_construction["label"],
        node_construction["unique_column_name"],
        node_construction["properties"],
    )


def import_relationships(relationship_construction: dict) -> Dict[str, Any]:
    """Import relationships as defined by a relationship construction rule."""
    try:
        relationship_type = _checked(
            "relationship type", relationship_construction["relationship_type"])
        from_label = _checked("label", relationship_construction["from_node_label"])
        to_label = _checked("label", relationship_construction["to_node_label"])
        from_column = _checked("column name", relationship_construction["from_node_column"])
        to_column = _checked("column name", relationship_construction["to_node_column"])
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    source_file = relationship_construction["source_file"]
    properties = relationship_construction["properties"]

    query = f"""UNWIND $rows AS row
    MATCH (from_node:{from_label} {{ {from_column} : row[$from_node_column] }}),
          (to_node:{to_label} {{ {to_column} : row[$to_node_column] }})
    MERGE (from_node)-[r:{relationship_type}]->(to_node)
    FOREACH (k IN $properties | SET r[k] = row[k])
    """

    rows_committed = 0
    try:
        for _header, batch in read_csv_batches(source_file):
            result = graphdb.send_query(query, {
                "rows": batch,
                "from_node_column": from_column,
                "to_node_column": to_column,
                "properties": properties,
            })
            if result["status"] == "error":
                return tool_error(
                    f"{source_file}: load failed after {rows_committed} rows committed "
                    f"(the failing batch was rolled back): {result['error_message']}"
                )
            rows_committed += len(batch)
    except (SourceError, FileNotFoundError) as exc:
        return tool_error(f"{source_file}: {exc}")

    return tool_success("rows_loaded", {"source_file": source_file, "rows": rows_committed})


def construct_domain_graph(construction_plan: dict) -> Dict[str, Any]:
    """Construct a domain graph according to a construction plan.

    Nodes are loaded before relationships, because the relationship query
    matches nodes that must already exist.
    """
    logger.debug("Building domain graph from plan: %s", construction_plan)

    outcomes = {}
    failures = []

    node_rules = [rule for rule in construction_plan.values()
                  if rule["construction_type"] == "node"]
    for rule in node_rules:
        result = import_nodes(rule)
        key = rule.get("label", rule.get("source_file", "?"))
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    relationship_rules = [rule for rule in construction_plan.values()
                          if rule["construction_type"] == "relationship"]
    for rule in relationship_rules:
        result = import_relationships(rule)
        key = rule.get("relationship_type", rule.get("source_file", "?"))
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    if failures:
        return tool_error("Graph construction had failures:\n" + "\n".join(failures))

    return tool_success("domain_graph_constructed", outcomes)


def build_graph_from_construction_rules(tool_context: ToolContext) -> Dict[str, Any]:
    """Build a graph from the approved construction rules."""
    if APPROVED_CONSTRUCTION_PLAN not in tool_context.state:
        return tool_error(f"{APPROVED_CONSTRUCTION_PLAN} not set.")

    return construct_domain_graph(tool_context.state[APPROVED_CONSTRUCTION_PLAN])
```

Note this deletes `construct_node` and `construct_relationship`, which had no callers.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_kg_construction_tools.py -v`

Expected: PASS (8 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: `50 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/tools/kg_construction_tools.py tests/unit/test_kg_construction_tools.py
git commit -m "feat: load CSVs driver-side with index-backed MERGE

Replaces LOAD CSV FROM file:/// with parameterised UNWIND batches.
Labels are interpolated after is_symbol validation rather than passed as
Cypher dynamic labels, which plan as Merge (no index) instead of
MergeUniqueNode. construct_domain_graph now reports failures instead of
discarding every result and always claiming success."
```

---

## Task 8: Per-job model selection

**Files:**
- Modify: `src/agentic_kg/common/llm_catalog.py`
- Modify: 8 agent construction sites (listed in Step 4)
- Create: `tests/unit/test_llm_catalog.py`

**Interfaces:**
- Consumes: `config.get_settings`
- Produces: `get_llm(kind: LlmKind = LlmKind.reasoning) -> LiteLlm` honouring settings and caching per kind

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_llm_catalog.py`:

```python
from agentic_kg.common.config import reset_settings
from agentic_kg.common import llm_catalog
from agentic_kg.common.llm_catalog import LlmKind, get_llm


def _clear_cache():
    llm_catalog._llm_instances.clear()


def test_reasoning_and_conversational_get_different_models(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_REASONING", "openai/gpt-4o")
    monkeypatch.setenv("LLM_MODEL_CONVERSATIONAL", "openai/gpt-4o-mini")
    reset_settings()
    _clear_cache()
    assert get_llm(LlmKind.reasoning).model == "openrouter/openai/gpt-4o"
    assert get_llm(LlmKind.conversational).model == "openrouter/openai/gpt-4o-mini"


def test_openrouter_prefix_is_derived_not_configured(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_REASONING", "anthropic/claude-3.5-sonnet")
    reset_settings()
    _clear_cache()
    assert get_llm(LlmKind.reasoning).model == "openrouter/anthropic/claude-3.5-sonnet"


def test_existing_prefix_is_not_doubled(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_REASONING", "openrouter/openai/gpt-4o")
    reset_settings()
    _clear_cache()
    assert get_llm(LlmKind.reasoning).model == "openrouter/openai/gpt-4o"


def test_instances_are_cached_per_kind(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_REASONING", "openai/gpt-4o")
    reset_settings()
    _clear_cache()
    assert get_llm(LlmKind.reasoning) is get_llm(LlmKind.reasoning)
    assert get_llm(LlmKind.reasoning) is not get_llm(LlmKind.conversational)


def test_returns_a_litellm_instance_not_a_string(monkeypatch):
    """ADK never registers LiteLlm in its LLMRegistry, so agents must be handed
    an instance. A bare model string raises at agent construction."""
    from google.adk.models.lite_llm import LiteLlm
    reset_settings()
    _clear_cache()
    assert isinstance(get_llm(LlmKind.reasoning), LiteLlm)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_llm_catalog.py -v`

Expected: FAIL — `AttributeError: module 'agentic_kg.common.llm_catalog' has no attribute '_llm_instances'`.

- [ ] **Step 3: Rewrite the catalog**

In `src/agentic_kg/common/llm_catalog.py`, replace everything from `_llm_instance: LiteLlm | None = None` to the end of the file with:

```python
_OPENROUTER_PREFIX = "openrouter/"

# Cached per kind. The previous single slot meant whichever caller ran first
# silently chose the model for every other caller.
_llm_instances: dict["LlmKind", LiteLlm] = {}


def _model_name(kind: LlmKind) -> str:
    """Resolve a kind to a LiteLLM model string.

    Settings hold OpenRouter's spelling ("openai/gpt-4o"); the "openrouter/"
    prefix LiteLLM routes on is derived here so the two cannot drift.
    """
    settings = get_settings()
    configured = (
        settings.llm_model_reasoning
        if kind is LlmKind.reasoning
        else settings.llm_model_conversational
    )
    if configured.startswith(_OPENROUTER_PREFIX):
        return configured
    return f"{_OPENROUTER_PREFIX}{configured}"


def get_llm(kind: LlmKind = LlmKind.reasoning) -> LiteLlm:
    """Return the LiteLlm instance for a kind of work.

    Returns an instance, never a bare model string: ADK registers only Gemini
    in its LLMRegistry, so `Agent(model="openrouter/...")` fails to resolve.
    """
    if kind not in _llm_instances:
        model = _model_name(kind)
        logger.info("Creating LLM for %s: %s", kind.value, model)
        _llm_instances[kind] = LiteLlm(model=model)
    return _llm_instances[kind]
```

Delete the now-unused `MODEL_*`, `LM_STUDIO_*` and `OLLAMA_*` constants only if nothing else references them — check with `grep -rn "MODEL_GPT_4O\|LM_STUDIO\|OLLAMA" src/`. If they are unreferenced, delete them; otherwise leave them.

- [ ] **Step 4: Assign kinds at every call site**

There are eight call sites. For each, ensure `LlmKind` is imported alongside `get_llm` and pass the kind explicitly:

| File | Line | Kind |
|---|---|---|
| `coordinators/multi_agent/agent.py` | 12 | `LlmKind.conversational` |
| `coordinators/multi_agent/sub_agents/user_intent_agent/agent.py` | 11 | `LlmKind.conversational` |
| `coordinators/multi_agent/sub_agents/file_suggestion_agent/agent.py` | 12 | `LlmKind.conversational` |
| `coordinators/multi_agent/sub_agents/graphrag_agent/agent.py` | 11 | `LlmKind.conversational` |
| `coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py` | 31 | `LlmKind.reasoning` |
| `coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py` | 41 | `LlmKind.reasoning` |
| `coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py` | 63 | `LlmKind.reasoning` |
| `coordinators/multi_agent/sub_agents/graph_construction_agent/agent.py` | 11 | `LlmKind.reasoning` |

Each import line changes from:

```python
from agentic_kg.common.llm_catalog import get_llm
```

to:

```python
from agentic_kg.common.llm_catalog import get_llm, LlmKind
```

and each call from `model=get_llm(),` to `model=get_llm(LlmKind.conversational),` or `model=get_llm(LlmKind.reasoning),` per the table.

Three further sites sit outside the `multi_agent` tree and also need kinds:

| File | Line | Kind |
|---|---|---|
| `coordinators/single_agent/agent.py` | 12 | `LlmKind.conversational` |
| `agents/cypher_agent/agent.py` | 10 | `LlmKind.conversational` |
| `agents/user_intent_agent/agent.py` | 13 | already `LlmKind.reasoning` — leave as is |

(The twelfth site was in `agents/file_suggestion_agent/agent.py`, deleted in Task 1. That call was unreachable anyway: it referenced `LlmKind` without importing it.)

- [ ] **Step 5: Verify no site was missed**

Run: `grep -rn "get_llm()" src/`

Expected: no output. Every call now passes an explicit kind.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -q`

Expected: `55 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_kg/common/llm_catalog.py src/agentic_kg/coordinators src/agentic_kg/agents tests/unit/test_llm_catalog.py
git commit -m "feat: per-job model selection via OpenRouter

The catalog hardcoded gpt-4o-mini, ignored the configured model, and
cached a single instance regardless of the requested kind. All eleven
remaining call sites now declare conversational or reasoning."
```

---

## Task 9: Remove the private-API dependency from finished()

**Files:**
- Create: `src/agentic_kg/coordinators/multi_agent/names.py`
- Modify: `src/agentic_kg/tools/adk_tools.py`
- Modify: 5 sites that put `finished` in a tools list
- Create: `tests/unit/test_adk_tools.py`

**Interfaces:**
- Consumes: nothing
- Produces: `make_finished(parent_agent_name: str) -> Callable`; `COORDINATOR_AGENT_NAME: str`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_adk_tools.py`:

```python
from agentic_kg.tools.adk_tools import make_finished


class FakeActions:
    def __init__(self):
        self.escalate = False
        self.transfer_to_agent = None


class FakeToolContext:
    def __init__(self):
        self.actions = FakeActions()


def test_finished_transfers_to_the_bound_parent():
    finished = make_finished("kg_construction_agent_v1")
    context = FakeToolContext()
    finished(context)
    assert context.actions.transfer_to_agent == "kg_construction_agent_v1"


def test_finished_sets_escalate():
    finished = make_finished("anything")
    context = FakeToolContext()
    finished(context)
    assert context.actions.escalate is True


def test_finished_takes_no_arguments_beyond_context():
    """A zero-argument tool is more reliable than one requiring the model to
    reproduce an agent name, which is why this is not ADK's transfer_to_agent."""
    import inspect
    finished = make_finished("x")
    parameters = list(inspect.signature(finished).parameters)
    assert parameters == ["tool_context"]


def test_tool_is_still_named_finished():
    assert make_finished("x").__name__ == "finished"


def test_no_private_attribute_access():
    """The old implementation reached into tool_context._invocation_context."""
    import inspect
    source = inspect.getsource(make_finished)
    assert "_invocation_context" not in source
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_adk_tools.py -v`

Expected: FAIL — `ImportError: cannot import name 'make_finished'`.

- [ ] **Step 3: Create the constants module**

Create `src/agentic_kg/coordinators/multi_agent/names.py`:

```python
"""Agent names shared between the coordinator and its sub-agents.

Sub-agents are constructed at import time, before the coordinator exists, so
they cannot discover their parent's name at runtime without reaching into ADK
private attributes. Holding the name here breaks that cycle. This module must
import nothing from the package, or the cycle returns.
"""

COORDINATOR_AGENT_NAME = "kg_construction_agent_v1"
```

- [ ] **Step 4: Replace `finished` with a factory**

Replace the entire contents of `src/agentic_kg/tools/adk_tools.py`:

```python
from typing import Any, Callable, Dict

from google.adk.tools import ToolContext


def make_finished(parent_agent_name: str) -> Callable[[ToolContext], Dict[str, Any]]:
    """Build a zero-argument 'finished' tool bound to a parent agent's name.

    ADK offers a public transfer_to_agent(agent_name, tool_context), but it
    requires the model to reproduce the target name as an argument. A
    zero-argument tool is categorically more reliable, especially on smaller
    models. Binding the name at construction avoids both the argument and the
    private-attribute lookup the previous implementation used.
    """

    def finished(tool_context: ToolContext) -> Dict[str, Any]:
        """Finish the current phase and hand control back to the coordinator."""
        tool_context.actions.escalate = True
        tool_context.actions.transfer_to_agent = parent_agent_name
        return {}

    return finished
```

`escalate` is currently inert — ADK reads it for control flow only in `LoopAgent`, and no caller runs inside one — but it is retained so a future phase inside a loop behaves correctly.

- [ ] **Step 5: Update the five call sites**

In each file below, replace the `finished` import with the factory and build the tool once at module level.

Files:
- `coordinators/multi_agent/sub_agents/user_intent_agent/variants.py`
- `coordinators/multi_agent/sub_agents/file_suggestion_agent/variants.py`
- `coordinators/multi_agent/sub_agents/graph_construction_agent/variants.py`
- `coordinators/multi_agent/sub_agents/graphrag_agent/variants.py`
- `coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py`

In each, change:

```python
from agentic_kg.tools.adk_tools import finished
```

to:

```python
from agentic_kg.tools.adk_tools import make_finished
from agentic_kg.coordinators.multi_agent.names import COORDINATOR_AGENT_NAME

finished = make_finished(COORDINATOR_AGENT_NAME)
```

The tools lists themselves are unchanged — they still reference the name `finished`.

- [ ] **Step 6: Point the coordinator at the shared constant**

In `coordinators/multi_agent/agent.py`, import the constant and use it as the agent name so the two cannot drift:

```python
from .names import COORDINATOR_AGENT_NAME
```

and change `name="kg_construction_agent_v1",` to `name=COORDINATOR_AGENT_NAME,`.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q`

Expected: `60 passed`.

- [ ] **Step 8: Commit**

```bash
git add src/agentic_kg/tools/adk_tools.py src/agentic_kg/coordinators tests/unit/test_adk_tools.py
git commit -m "refactor: bind finished() to its parent at construction

Removes the tool_context._invocation_context.agent.parent_agent.name
private lookup. The tool stays zero-argument, which is why it exists
rather than ADK's transfer_to_agent."
```

---

## Task 10: Wire up the coordinator and environment validation

**Files:**
- Modify: `src/agentic_kg/common/config.py` (`validate_env`)
- Modify: `src/agentic_kg/coordinators/multi_agent/agent.py`

**Interfaces:**
- Consumes: `file_tools.get_source_location`, `config.validate_env`
- Produces: a coordinator that reports the source location and validates configuration on import

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config.py`:

```python
import pytest

from agentic_kg.common.config import validate_env


def test_validate_env_requires_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_settings()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        validate_env()


def test_validate_env_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
    reset_settings()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        validate_env()


def test_validate_env_passes_with_a_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-realish")
    reset_settings()
    validate_env()
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/unit/test_config.py -v`

Expected: FAIL — the current `validate_env` checks `openai_api_key`, not `openrouter_api_key`.

- [ ] **Step 3: Rewrite validate_env**

Replace `validate_env()` in `src/agentic_kg/common/config.py`:

```python
def validate_env() -> None:
    """Validate configuration required for the system to function.

    Raises:
        ValueError: if the OpenRouter key is missing or still a placeholder.
    """
    settings = get_settings()

    key = settings.openrouter_api_key
    if not key or key.startswith("YOUR_"):
        raise ValueError(
            "OPENROUTER_API_KEY is not set (or is still the placeholder). "
            "One OpenRouter key covers chat, extraction and embeddings."
        )

    if not settings.source_uri:
        logger.warning(
            "SOURCE_URI is not set. File tools will report an error until it is."
        )
```

Note `source_uri` warns rather than raises: the coordinator can still answer questions about the database when no source is configured.

- [ ] **Step 4: Wire validation into the live path**

In `src/agentic_kg/coordinators/multi_agent/agent.py`, replace the import of `get_neo4j_import_dir` and add validation. The import block becomes:

```python
from google.adk.agents import LlmAgent

from agentic_kg.common.config import validate_env
from agentic_kg.common.llm_catalog import get_llm, LlmKind
from agentic_kg.tools.cypher_tools import get_physical_schema, neo4j_is_ready
from agentic_kg.tools.file_tools import get_source_location

from .names import COORDINATOR_AGENT_NAME
from .sub_agents import (
    user_intent_agent, file_suggestion_agent, schema_proposal_agent,
    graph_construction_agent, graphrag_agent,
)

validate_env()
```

Update the instruction text, replacing the import-directory line:

```
        - finding where source files are read from with the 'get_source_location' tool
```

And the tools list:

```python
    tools=[
        get_physical_schema,
        get_source_location,
        neo4j_is_ready
    ]
```

- [ ] **Step 5: Verify the coordinator still loads**

Run:
```bash
OPENROUTER_API_KEY=sk-or-test SOURCE_URI=./data/bom uv run python -c "from agentic_kg.coordinators.multi_agent.agent import root_agent; print(root_agent.name, len(root_agent.tools), len(root_agent.sub_agents))"
```

Expected: `kg_construction_agent_v1 3 5`

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`

Expected: `63 passed`.

> If `test_imports.py` now fails because importing the coordinator calls `validate_env()` without a key set, add `OPENROUTER_API_KEY` to the test environment via a `tests/conftest.py`:
> ```python
> import os
> os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test-key")
> os.environ.setdefault("SOURCE_URI", "./data/bom")
> ```
> This is expected, not a failure of the design — the smoke test importing every module is precisely what surfaces it.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_kg/common/config.py src/agentic_kg/coordinators/multi_agent/agent.py tests/
git commit -m "feat: validate OpenRouter config on the live coordinator path

validate_env previously checked the OpenAI key and was called only from
agentic_kg/agent.py, which adk web never loads, so it had never run."
```

---

## Task 11: Integration test against a real Neo4j

**Files:**
- Create: `tests/integration/test_csv_loading_integration.py`

**Interfaces:**
- Consumes: everything built so far
- Produces: proof that the loading path works against a real database

- [ ] **Step 1: Write the test**

Create `tests/integration/test_csv_loading_integration.py`:

```python
"""Load the bundled BOM CSVs into a real Neo4j and assert the result.

This exercises the same code that runs against Aura. It is representative
precisely because the file:/// path was removed rather than kept alongside —
there is only one loading implementation to test.
"""
import pytest

pytestmark = pytest.mark.integration

try:
    import docker
    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


PLAN = {
    "Product": {
        "construction_type": "node",
        "source_file": "products.csv",
        "label": "Product",
        "unique_column_name": "product_id",
        "properties": ["product_name", "price", "description"],
    },
    "Supplier": {
        "construction_type": "node",
        "source_file": "suppliers.csv",
        "label": "Supplier",
        "unique_column_name": "supplier_id",
        # suppliers.csv columns are: supplier_id,name,specialty,city,country,
        # website,contact_email — note "name", not "supplier_name"
        "properties": ["name", "specialty", "city", "country"],
    },
    "SUPPLIED_BY": {
        "construction_type": "relationship",
        "source_file": "part_supplier_mapping.csv",
        "relationship_type": "SUPPLIED_BY",
        "from_node_label": "Part",
        "from_node_column": "part_id",
        "to_node_label": "Supplier",
        "to_node_column": "supplier_id",
        "properties": ["lead_time_days", "unit_cost"],
    },
    "Part": {
        "construction_type": "node",
        "source_file": "part_supplier_mapping.csv",
        "label": "Part",
        "unique_column_name": "part_id",
        "properties": ["part_name"],
    },
}


@pytest.fixture
def neo4j_graph(monkeypatch):
    from testcontainers.neo4j import Neo4jContainer

    with Neo4jContainer(image="neo4j:5") as container:
        url = container.get_connection_url()
        host_port = url.split("//")[1]
        monkeypatch.setenv("NEO4J_DSN", f"bolt://neo4j:password@{host_port}/neo4j")
        monkeypatch.setenv("SOURCE_URI", "./data/bom")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        from agentic_kg.common.config import reset_settings
        import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
        reset_settings()
        neo4j_for_adk.close_graphdb()

        yield neo4j_for_adk.get_graphdb()

        neo4j_for_adk.close_graphdb()


def test_loads_bom_csvs_into_the_graph(neo4j_graph, monkeypatch):
    import agentic_kg.tools.kg_construction_tools as kg
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    import agentic_kg.tools.cypher_tools as cypher_tools
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    result = kg.construct_domain_graph(PLAN)
    assert result["status"] == "success", result.get("error_message")

    # Counts come from the bundled data: products.csv has 10 data rows,
    # suppliers.csv has 20, and part_supplier_mapping.csv has 176 rows over
    # 88 distinct part_id values.
    products = neo4j_graph.send_query("MATCH (p:Product) RETURN count(p) AS c")
    assert products["records"][0]["c"] == 10

    suppliers = neo4j_graph.send_query("MATCH (s:Supplier) RETURN count(s) AS c")
    assert suppliers["records"][0]["c"] == 20

    parts = neo4j_graph.send_query("MATCH (p:Part) RETURN count(p) AS c")
    assert parts["records"][0]["c"] == 88

    rels = neo4j_graph.send_query(
        "MATCH (:Part)-[r:SUPPLIED_BY]->(:Supplier) RETURN count(r) AS c")
    assert rels["records"][0]["c"] > 0

    # A property from suppliers.csv must actually have landed
    named = neo4j_graph.send_query(
        "MATCH (s:Supplier) WHERE s.name IS NOT NULL RETURN count(s) AS c")
    assert named["records"][0]["c"] == 20


def test_loading_twice_is_idempotent(neo4j_graph, monkeypatch):
    import agentic_kg.tools.kg_construction_tools as kg
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    import agentic_kg.tools.cypher_tools as cypher_tools
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    kg.construct_domain_graph(PLAN)
    first = neo4j_graph.send_query("MATCH (n) RETURN count(n) AS c")["records"][0]["c"]
    kg.construct_domain_graph(PLAN)
    second = neo4j_graph.send_query("MATCH (n) RETURN count(n) AS c")["records"][0]["c"]

    assert first == second, "MERGE should update rather than duplicate"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest -q -m integration tests/integration/test_csv_loading_integration.py`

Expected: PASS (2 tests). If using colima rather than Docker Desktop, export first:
```bash
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
export TESTCONTAINERS_RYUK_DISABLED=true
```

The expected counts were taken from the bundled data on 2026-07-27: `products.csv` 10 data rows, `suppliers.csv` 20, `part_supplier_mapping.csv` 176 rows over 88 distinct `part_id` values. If your checkout differs, correct the assertions to the real numbers — do not weaken them to `> 0`, which would pass even if loading were broken.

- [ ] **Step 3: Run the whole suite both ways**

Run:
```bash
uv run pytest -q
uv run pytest -q -m integration
```

Expected: unit `63 passed`; integration passes with Docker available.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_csv_loading_integration.py
git commit -m "test: load BOM CSVs into a container Neo4j end to end"
```

---

## Task 12: Documentation and manual acceptance

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything
- Produces: a working setup path for a fresh clone

- [ ] **Step 1: Update .env.example**

Replace the LLM and add the source sections:

```bash
# --- Logging ---
LOGLEVEL=INFO

# --- LLM provider (single key covers chat, extraction and embeddings) ---
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY

# --- Per-job models, in OpenRouter's spelling ---
# The "openrouter/" prefix LiteLLM needs is added automatically.
LLM_MODEL_CONVERSATIONAL=openai/gpt-4o-mini
LLM_MODEL_REASONING=openai/gpt-4o

# --- Where source files live ---
# A local folder, an s3:// bucket (needs s3fs installed), or an https:// URL.
# Relative paths are resolved against the repository root, not your shell's
# working directory.
SOURCE_URI=./data/bom

# --- Neo4j connection ---
# DSN format: scheme://[username[:password]@]host[:port][/database]
NEO4J_DSN=bolt://neo4j:secret@localhost:7687/neo4j

# --- Tests ---
# RUN_NEO4J_IT=1
```

- [ ] **Step 2: Update the README setup section**

Replace section "### 3) Make CSV files available for import" entirely with:

```markdown
### 3) Point the system at your source files

Set `SOURCE_URI` in `.env` to wherever your files live. The bundled example
data works out of the box:

```
SOURCE_URI=./data/bom
```

Files are read by the application, not by the database, so nothing needs to be
copied into Neo4j's import directory — and this works against Neo4j Aura,
which has no such directory.

Relative paths resolve against the repository root. Absolute paths, `s3://`
URLs (requires `s3fs`) and `https://` URLs also work.
```

Also update the environment variable list in "### 2) Set up environment variables":

```markdown
- `OPENROUTER_API_KEY=sk-or-...` (required — one key covers every model)
- `NEO4J_DSN=bolt://neo4j:secret@localhost:7687/neo4j`
- `SOURCE_URI=./data/bom`
```

- [ ] **Step 3: Commit the docs**

```bash
git add .env.example README.md
git commit -m "docs: replace the import-directory setup step with SOURCE_URI"
```

- [ ] **Step 4: Manual acceptance — the actual done-condition**

This cannot be automated; there is no Aura in CI. Do not skip it and do not fake it.

1. Set `.env` to your Aura DSN, a real `OPENROUTER_API_KEY`, and `SOURCE_URI=./data/bom`
2. Confirm the database is empty: `MATCH (n) RETURN count(n)` should be 0, or accept that existing data will be merged into
3. Run `uv run adk web src/agentic_kg/coordinators/`
4. Choose `multi_agent`
5. Ask "Is Neo4j ready?" and "Where are my files?" — expect a ready message and the resolved source path
6. Drive the full workflow: state a goal about supply chain analysis, approve the suggested files, approve the proposed construction plan, let construction run
7. Confirm the graph exists: ask the agent to count nodes by label
8. Confirm each phase handed back correctly — you should not see the coordinator get stuck after any sub-agent finishes

- [ ] **Step 5: Record the outcome**

Append a short "Acceptance" note to the spec recording the date, what was run, and any deviation observed. If anything failed, do not mark this task complete — open the specific problem instead.

```bash
git add docs/superpowers/specs/2026-07-27-foundation-design.md
git commit -m "docs: record Foundation acceptance run"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `file_source.py` seam, repo-root anchoring | 4 |
| Driver-side CSV loading, no `file:///` | 7 |
| Interpolated validated labels, not dynamic `$()` | 7 |
| Symmetric identifier validation | 7 |
| Per-file result collection | 7 |
| Delete `get_neo4j_import_dir` | 6 |
| Thin file tools | 6 |
| Delete `import_markdown_file` | 6 |
| Delete `construct_node`/`construct_relationship` | 7 |
| Fix `approve_suggested_files` | 6 |
| Three import defects | 1 |
| `SOURCE_URI` + OpenRouter + model settings | 3 |
| `llm_catalog` per-kind cache | 8 |
| Kinds at every call site | 8 |
| `validate_env` wired in | 10 |
| `make_finished` + `names.py` | 9 |
| Dependency bounds, `fsspec`, `aiohttp` | 2 |
| `.env.example`, README | 12 |
| Unit tests: file_source with `memory://` | 4 |
| Unit tests: CSV batching without a database | 5 |
| Integration test with testcontainers | 11 |
| Import smoke test | 1 |
| Manual Aura acceptance | 12 |

No gaps.

**Placeholder scan:** No "TBD", "TODO", "add error handling", or "similar to Task N". Every code step contains runnable code.

**Type consistency:** `read_csv_batches` yields `(header, rows)` in Task 5 and is consumed as `for _header, batch in ...` in Task 7. `get_source_fs()` returns `(fs, root)` in Task 4 and is destructured the same way throughout. `SourceError` is raised in Task 4 and caught in Tasks 6 and 7. `make_finished` is defined in Task 9 and called as `make_finished(COORDINATOR_AGENT_NAME)` in the same task. `reset_settings` is defined in Task 3 and used in Tasks 4, 5, 8, 10, 11.

**Known deviation from the spec:** the spec's components table says "resolve undefined `file_toolset`" in `agents/file_suggestion_agent/variants.py`; this plan deletes the package instead, for the reasons documented in Task 1. This reduces the `get_llm()` call sites from twelve to eleven.
