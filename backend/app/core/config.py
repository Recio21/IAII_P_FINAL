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

    # CSV de orígenes permitidos para CORS. pydantic_settings lee esto como str puro
    # y se parsea en el punto de uso para evitar el JSON-decode automático de List[str].
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

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
