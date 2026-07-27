import logging
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from agentic_kg.common.pydantic_neo4j import Neo4jDsn

class agentic_kgSettings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    # Logging configuration
    loglevel: str = Field(default="INFO")

    # LLM Provider configuration
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    gemini_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    # LLM Model configuration
    llm_model: Optional[str] = Field(default="openai/gpt-4o")
    llm_base_url: Optional[str] = Field(default=None)

    # Source file location (local path, bucket URL, or http(s) URL).
    # Relative local paths are anchored to the repository root, not the CWD.
    source_uri: Optional[str] = Field(default=None)

    # OpenRouter is the single provider for chat, extraction and embeddings.
    openrouter_api_key: Optional[str] = Field(default=None)

    # Per-job models, stored in OpenRouter's spelling (e.g. "openai/gpt-4o").
    # The "openrouter/" prefix LiteLLM wants is derived, not configured.
    llm_model_conversational: str = Field(default="openai/gpt-4o-mini")
    llm_model_reasoning: str = Field(default="openai/gpt-4o")

    # Neo4j configuration
    neo4j_dsn: Optional[Neo4jDsn] = Field(default="bolt://localhost:7687")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

# Global settings instance
_settings: Optional[agentic_kgSettings] = None
logger = logging.getLogger(__name__)

def get_settings() -> agentic_kgSettings:
    """Get the application settings singleton, loading and initializing if necessary."""
    global _settings
    if _settings is None:
        _settings = agentic_kgSettings()

    # Configure logging only once to avoid double handlers
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=_settings.loglevel.upper())

    return _settings


def reset_settings() -> None:
    """Discard the cached settings so the next get_settings() re-reads the environment.

    Intended for tests. Production code should never need this.
    """
    global _settings
    _settings = None


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
