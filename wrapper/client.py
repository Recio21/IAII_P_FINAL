from __future__ import annotations
from pathlib import Path
from typing import Any
import httpx


class PipelineClientError(RuntimeError):
    pass


class PipelineClient:
    """Cliente reutilizable para FastAPI. Sin lógica de negocio.
    Reutilizable desde frontend (uso server-side), seeder, scripts y tests.
    """

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    def __enter__(self) -> "PipelineClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _json(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            raise PipelineClientError(f"{r.request.method} {r.request.url} -> {r.status_code}: {r.text}")
        return r.json()

    def health(self) -> dict:
        return self._json(self._client.get("/health"))

    def model_status(self) -> dict:
        return self._json(self._client.get("/model/status"))

    def predict_image(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        with path.open("rb") as f:
            files = {"file": (path.name, f, "image/jpeg")}
            return self._json(self._client.post("/predict", files=files))

    def predict_bytes(self, raw: bytes, filename: str = "image.jpg", content_type: str = "image/jpeg") -> dict:
        files = {"file": (filename, raw, content_type)}
        return self._json(self._client.post("/predict", files=files))

    def retrain(self) -> dict:
        return self._json(self._client.post("/retrain"))

    def switch_model_version(self, model_file: str) -> dict:
        return self._json(self._client.post("/model/switch", json={"model_file": model_file}))

    def send_observation(self, payload: dict) -> dict:
        return self._json(self._client.post("/observations", json=payload))
