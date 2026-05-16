from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PipelineModeling"
    app_env: str = "local"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    model_dir: Path = Path("/app/models")
    active_model_file: str = "model_7_final.keras"
    metadata_file: str = "metadata_7.json"
    observations_path: Path = Path("/app/data/observations.jsonl")
    max_upload_bytes: int = 5 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="")

    @property
    def active_model_path(self) -> Path:
        return self.model_dir / self.active_model_file

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
