import os

import pytest

from agentic_kg.common.config import agentic_kgSettings, reset_settings

os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test-key")
os.environ.setdefault("SOURCE_URI", "./data/bom")


@pytest.fixture(autouse=True, scope="session")
def _unit_tests_ignore_dotenv():
    """Stop a developer's real `.env` from leaking into unit test assertions.

    agentic_kgSettings.model_config sets env_file=".env", and pydantic-settings
    reads that file unconditionally -- regardless of what monkeypatch.delenv()
    does to os.environ. So on a machine with a populated .env (as the README's
    setup step produces), tests like test_source_uri_defaults_to_none that
    delenv() a setting and assert it is absent would instead see the value
    from .env and fail, while the same tests pass "by accident" on a clean
    machine or in CI (which has no .env). That made the unit suite silently
    depend on ambient developer state -- the same bug class as the earlier
    NEO4J_PASSWORD container leak fixed elsewhere on this branch (see
    tests/integration/test_csv_loading_integration.py).

    Disable .env loading for the whole test session so settings construction
    # depends only on os.environ, which monkeypatch controls. This does not
    # change production behaviour: application code never touches
    # model_config, so a real run of the app still reads .env normally.
    """
    original_env_file = agentic_kgSettings.model_config.get("env_file")
    agentic_kgSettings.model_config["env_file"] = None
    yield
    agentic_kgSettings.model_config["env_file"] = original_env_file


@pytest.fixture(autouse=True)
def _reset_settings_after_test():
    """Discard the cached settings singleton after every test.

    reset_settings() was previously only ever called *before* assertions
    within a test, never after, so a config test could leave a stale
    (env-mismatched) settings singleton cached for whichever test runs next.
    """
    yield
    reset_settings()
