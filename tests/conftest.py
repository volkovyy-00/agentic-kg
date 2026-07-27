import os

import pytest

from agentic_kg.common.config import reset_settings

os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test-key")
os.environ.setdefault("SOURCE_URI", "./data/bom")


@pytest.fixture(autouse=True)
def _reset_settings_after_test():
    """Discard the cached settings singleton after every test.

    reset_settings() was previously only ever called *before* assertions
    within a test, never after, so a config test could leave a stale
    (env-mismatched) settings singleton cached for whichever test runs next.
    """
    yield
    reset_settings()
