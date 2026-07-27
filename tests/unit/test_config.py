import pytest

from agentic_kg.common.config import get_settings, reset_settings, validate_env


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


def test_validate_env_warns_but_does_not_raise_without_source_uri(monkeypatch, caplog):
    """A missing SOURCE_URI is a warning, not a ValueError: the OpenRouter key
    is the one setting validate_env() must enforce hard, since file tools can
    still report their own tool_error later and the coordinator may only want
    to check readiness, not import files, in a given turn."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-realish")
    monkeypatch.delenv("SOURCE_URI", raising=False)
    reset_settings()

    with caplog.at_level("WARNING"):
        validate_env()

    assert any("SOURCE_URI" in record.message for record in caplog.records)
