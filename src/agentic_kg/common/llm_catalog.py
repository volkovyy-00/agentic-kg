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
