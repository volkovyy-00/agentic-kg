from enum import Enum

import logging

from google.adk.models.lite_llm import LiteLlm
import litellm

from .config import get_settings

logger = logging.getLogger(__name__)

litellm.log_raw_request_response = False
litellm.suppress_debug_info = True
litellm.turn_off_message_logging=True
litellm.logging = False
litellm._logging._disable_debugging()


class LlmKind(str, Enum):
    reasoning = 'reasoning'
    conversational = 'conversational'


_OPENROUTER_PREFIX = "openrouter/"

# Without an explicit timeout, LiteLLM falls back to its own default (hundreds
# of seconds), so a stalled OpenRouter backend leaves a turn hanging with no
# error for the user to react to. Every observed healthy call in this system
# completes well under a minute; 300s is generous headroom for slow reasoning
# models while still guaranteeing the turn fails fast enough to retry.
_LLM_TIMEOUT_SECONDS = 300
_LLM_NUM_RETRIES = 2

# OpenRouter pre-authorizes a call against the account balance using the
# model's max output tokens when none is given (65536 for gpt-5), then
# refunds the unused portion after generation — so a call that will only
# ever produce a few hundred tokens can still get rejected with a 402 well
# before the balance is actually exhausted. Capping this bounds that
# pre-authorization to something every observed turn in this system fits
# comfortably under.
_LLM_MAX_TOKENS = 8192

# The reasoning kind's model may be a dedicated "reasoning" model (e.g. gpt-5)
# that defaults to its highest internal-reasoning tier on every call. This
# agent's reasoning workloads (schema proposal/critique) are many small,
# structured tool-orchestration steps, not deep multi-step proofs, so a lower
# effort cuts latency per call substantially without a quality regression
# we've observed. Only passed for LlmKind.reasoning below.
_REASONING_EFFORT = "low"

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
        kwargs = {
            "timeout": _LLM_TIMEOUT_SECONDS,
            "num_retries": _LLM_NUM_RETRIES,
            "max_tokens": _LLM_MAX_TOKENS,
        }
        if kind is LlmKind.reasoning:
            kwargs["reasoning_effort"] = _REASONING_EFFORT
        _llm_instances[kind] = LiteLlm(model=model, **kwargs)
    return _llm_instances[kind]
