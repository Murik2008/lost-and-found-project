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

    # ── LLM (VLM)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"

    # ── Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # ── API keys
    openai_api_key: str = ""
    google_api_key: str = ""

    # ── Database
    database_url: str = ""

    # ── Filesystem
    data_dir: Path = Path("data")

    @property
    def lost_dir(self) -> Path:
        return self.data_dir / "lost"

    @property
    def found_dir(self) -> Path:
        return self.data_dir / "found"

    # ── Validation
    max_file_size_bytes: int = 5 * 1024 * 1024   # 5 MB

    # ── Logging
    log_level: str = "INFO"

    # ── Retry policy ()
    max_retries: int = 3
    retry_min_wait: float = 1.0   # seconds
    retry_max_wait: float = 30.0  # seconds


settings = Settings()