"""Application settings, read from environment variables or a local ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BYTES_PER_MB = 1024 * 1024


class Settings(BaseSettings):
    """Runtime configuration. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"
    storage_dir: Path = Path("./data/uploads")
    max_upload_mb: int = Field(default=50, gt=0)
    max_unzipped_mb: int = Field(default=500, gt=0)
    log_level: str = "INFO"
    densify_max_segment_deg: float = Field(default=0.01, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        """Upload size cap in bytes."""
        return self.max_upload_mb * _BYTES_PER_MB

    @property
    def max_unzipped_bytes(self) -> int:
        """Cap on the total uncompressed size of a zip archive, in bytes."""
        return self.max_unzipped_mb * _BYTES_PER_MB


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built once from the environment."""
    return Settings()
