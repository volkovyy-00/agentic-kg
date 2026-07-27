# Fix: unit tests silently reading real `.env` instead of `os.environ`

## Root cause

`agentic_kgSettings.model_config` (`src/agentic_kg/common/config.py`) sets
`env_file=".env"`. `pydantic-settings` reads that file unconditionally when
constructing `agentic_kgSettings()`, on top of `os.environ`. `monkeypatch.delenv()`
only removes a variable from `os.environ` — it does nothing to the `.env` file, so
any test that does `monkeypatch.delenv("SOURCE_URI")` and then asserts the setting
is `None` would instead see the value pydantic-settings read from `.env`, on any
machine that has a populated `.env` (e.g. after following the README setup step).
CI never caught this because CI has no `.env` file. Same bug class as the earlier
`NEO4J_PASSWORD` container-credential leak fixed on this branch
(`tests/integration/test_csv_loading_integration.py`).

## Mechanism chosen

Added an **autouse, session-scoped** fixture in `tests/conftest.py`
(`_unit_tests_ignore_dotenv`) that sets
`agentic_kgSettings.model_config["env_file"] = None` for the duration of the
pytest session, and restores the original value (`".env"`) on teardown.

Why this approach over alternatives:
- **Session-scoped**, not function-scoped: `model_config` is a class-level dict
  shared by every `agentic_kgSettings()` instance; there is no per-test state to
  reset, and no test in the suite relies on real `.env` loading behaviour (verified
  by grep — the only `.env` references outside this fixture are in the integration
  suite, which is a Testcontainers/Docker test, out of scope here). A single
  session-wide flip avoids repeating the mutation on every one of the ~80 tests.
- **Mutating `model_config["env_file"]`** rather than patching `_env_file` on each
  `agentic_kgSettings()` call site, or setting `ENV_FILE=/dev/null`-style tricks:
  it's the one property that actually controls `.env` reading in pydantic-settings,
  it's reverted cleanly, and it doesn't touch `os.environ` (which the tests still
  need to control via `monkeypatch`) or the real `.env` file (never opened,
  moved, or renamed).
- Restoring the original value on teardown means if anything ever runs
  `agentic_kgSettings()` after the test session tears down (unlikely, but cheap
  to guarantee), it goes back to reading `.env` as normal.

Also added a small regression-guard test,
`test_unit_suite_ignores_dotenv_file` in `tests/unit/test_config.py`, asserting
`agentic_kgSettings.model_config.get("env_file") is None` during a test run. If
the conftest fixture is ever deleted or stops applying, this fails loudly and
immediately, rather than the five original tests failing mysteriously and only
on machines with a populated `.env`.

## Files changed

- `tests/conftest.py` — new `_unit_tests_ignore_dotenv` autouse session fixture,
  with a comment explaining why it exists (referencing the earlier
  `NEO4J_PASSWORD` leak as the same bug class).
- `tests/unit/test_config.py` — new `test_unit_suite_ignores_dotenv_file`
  regression guard; import of `agentic_kgSettings` added.

No production code changed. No changes to `pyproject.toml` or `uv.lock`.

## Verification

### Before the fix (reproduced by temporarily stashing the fix, `.env` present)

```
FAILED tests/unit/test_config.py::test_source_uri_defaults_to_none - Assertio...
FAILED tests/unit/test_config.py::test_validate_env_requires_openrouter_key
FAILED tests/unit/test_config.py::test_validate_env_warns_but_does_not_raise_without_source_uri
FAILED tests/unit/test_file_source.py::test_unset_source_uri_raises_source_error
FAILED tests/unit/test_file_tools.py::test_unset_source_uri_surfaces_as_tool_error
5 failed, 71 passed, 3 skipped, 2 warnings in 8.76s
```

This matches exactly the five tests reported in the defect.

### After the fix, with `.env` present (the real file at repo root, untouched)

Ran from the repo root, `.env` present with real credentials:

```
$ uv run pytest
........................................................................ [ 93%]
.....                                                                    [100%]
77 passed, 3 skipped, 2 warnings in 11.36s
```

(77 = the original 76 plus the new `test_unit_suite_ignores_dotenv_file`
regression guard.) Also ran the five originally-failing tests individually and
confirmed all five pass:

```
$ uv run pytest tests/unit/test_config.py::test_source_uri_defaults_to_none \
    tests/unit/test_config.py::test_validate_env_requires_openrouter_key \
    tests/unit/test_config.py::test_validate_env_warns_but_does_not_raise_without_source_uri \
    tests/unit/test_file_source.py::test_unset_source_uri_raises_source_error \
    tests/unit/test_file_tools.py::test_unset_source_uri_surfaces_as_tool_error -v
5 passed, 2 warnings in 6.49s
```

### After the fix, simulating the no-`.env` (CI) case, without touching the real file

`.env` was never moved, renamed, or deleted. Instead, since pydantic-settings
resolves a relative `env_file` path against the process's current working
directory (confirmed directly: `agentic_kgSettings()` constructed after
`os.chdir("/tmp")` reads `source_uri=None` even though the repo's `.env` sets
it), the suite was run with the process's cwd pointed at an empty scratch
directory that has no `.env`, using `uv run --project <repo-abs-path> pytest
<repo-abs-path>/tests`:

```
$ cd /private/tmp/claude-503/.../scratchpad/no-dotenv-cwd   # empty dir, no .env
$ uv run --project /Users/volkovyy/Projects/workshops/kg_construction/agentic-kg \
    pytest /Users/volkovyy/Projects/workshops/kg_construction/agentic-kg/tests
........................................................................ [ 93%]
.....                                                                    [100%]
77 passed, 3 skipped, 2 warnings in 9.98s
```

Same result as the CI (no-`.env`) scenario, confirming the suite is no longer
sensitive either way to whether a `.env` file is present. No `uv sync`/`uv lock`/
Docker was run; disk usage was checked before and after (`df -h`) and did not
change (~333Mi free throughout).

## Confirmation: production `.env` loading is unaffected

`model_config["env_file"]` is only mutated inside the pytest fixture, and
restored on teardown. Outside of pytest (i.e. in a real run of the application),
`agentic_kgSettings.model_config["env_file"]` is untouched. Verified directly:

```
$ uv run python -c "
from agentic_kg.common.config import agentic_kgSettings, get_settings
print('model_config env_file (outside pytest):', agentic_kgSettings.model_config.get('env_file'))
s = get_settings()
print('source_uri loaded from real .env:', bool(s.source_uri))
print('openrouter_api_key loaded (non-empty):', bool(s.openrouter_api_key))
"
model_config env_file (outside pytest): .env
source_uri loaded from real .env: True
openrouter_api_key loaded (non-empty): True
```

(Only booleans/the literal config value are printed — no secret values from
`.env` were displayed or logged at any point during this fix.)

Confirms: outside of the test session, `agentic_kgSettings` still declares
`env_file=".env"` and a real `get_settings()` call still loads real values from
the developer's `.env` file exactly as before this change.
