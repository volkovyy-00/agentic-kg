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
