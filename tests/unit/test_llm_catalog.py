from agentic_kg.common import llm_catalog
from agentic_kg.common.config import reset_settings
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


def test_llm_calls_have_an_explicit_timeout_and_retries(monkeypatch):
    """Without an explicit timeout, a stalled OpenRouter backend leaves the
    whole turn hanging on LiteLLM's default (hundreds of seconds) with no
    error surfaced to the user."""
    reset_settings()
    _clear_cache()
    args = get_llm(LlmKind.reasoning)._additional_args
    assert args.get("timeout") == llm_catalog._LLM_TIMEOUT_SECONDS
    assert args.get("num_retries") == llm_catalog._LLM_NUM_RETRIES
    assert args["timeout"] > 0


def test_reasoning_kind_asks_for_a_low_reasoning_effort(monkeypatch):
    """A dedicated reasoning model defaults to its highest internal-reasoning
    tier on every call, which made schema-proposal turns take tens of minutes.
    The lower effort has to actually reach LiteLLM to have that effect."""
    reset_settings()
    _clear_cache()
    args = get_llm(LlmKind.reasoning)._additional_args
    assert args.get("reasoning_effort") == llm_catalog._REASONING_EFFORT
    assert args["reasoning_effort"] == "low"


def test_conversational_kind_does_not_send_a_reasoning_effort(monkeypatch):
    """Conversational models generally reject the parameter outright, so it
    must be absent rather than merely unset."""
    reset_settings()
    _clear_cache()
    assert "reasoning_effort" not in get_llm(LlmKind.conversational)._additional_args


def test_returns_a_litellm_instance_not_a_string(monkeypatch):
    """ADK never registers LiteLlm in its LLMRegistry, so agents must be handed
    an instance. A bare model string raises at agent construction."""
    from google.adk.models.lite_llm import LiteLlm

    reset_settings()
    _clear_cache()
    assert isinstance(get_llm(LlmKind.reasoning), LiteLlm)
