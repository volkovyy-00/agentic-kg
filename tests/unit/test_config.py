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
