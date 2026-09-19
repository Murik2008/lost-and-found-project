"""
Application configuration via pydantic-settings.

"""

from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (VLM)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.6-flash"

    # Embedding
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"

    # API keys (never hard-code values, use .env)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # Database (empty string = in-memory fallback for local runs)
    database_url: str = ""
    db_pool_min_size: int = 1
    db_pool_max_size: int = 5
    db_command_timeout: float = 10.0

    # Filesystem
    data_dir: Path = Path("data")

    @property
    def lost_dir(self) -> Path:
        return self.data_dir / "lost"

    @property
    def found_dir(self) -> Path:
        return self.data_dir / "found"

    # Validation
    max_file_size_bytes: int = 5 * 1024 * 1024

    @property
    def max_file_size(self) -> int:
        return self.max_file_size_bytes

    # Logging
    log_level: str = "INFO"

    # Retry policy
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 30.0
    ai_timeout_seconds: float = 30.0

    # Concurrency (max parallel AI calls)
    concurrency_limit: int = 8

    # Offline demo mode (no API keys needed, deterministic fake providers)
    offline_mode: bool = False

    @property
    def has_api_keys(self) -> bool:
        return bool(
            self.anthropic_api_key or self.openai_api_key or self.google_api_key
        )

    @property
    def use_offline(self) -> bool:
        return self.offline_mode


settings = Settings()