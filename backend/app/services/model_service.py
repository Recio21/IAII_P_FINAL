import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    class_names: List[str]
    image_size: Tuple[int, int]
    channels: int = 1
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "ModelMetadata":
        raw = json.loads(path.read_text(encoding="utf-8"))
        size = raw.get("IMG_SIZE", [128, 128])
        return cls(
            class_names=raw["class_names"],
            image_size=(int(size[0]), int(size[1])),
            channels=int(raw.get("channels", 1)),
            extra=raw,
        )


class ModelService:
    """Singleton de modelo Keras. Carga lazy, cambia de versión bajo lock.
    No recarga pesos por request; lectura concurrente con RLock para predict.
    """

    def __init__(self, model_dir: Path, active_file: str, metadata_file: str, metrics) -> None:
        self._model_dir = model_dir
        self._active_file = active_file
        self._metadata_file = metadata_file
        self._metrics = metrics
        self._model = None
        self._metadata: ModelMetadata | None = None
        self._lock = threading.RLock()
        self._load_seconds: float | None = None

    @property
    def version(self) -> str:
        return Path(self._active_file).stem

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("Model metadata not loaded")
        return self._metadata

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from tensorflow.keras.models import load_model  # local import: faster cold start
        with self._lock:
            start = time.perf_counter()
            meta_path = self._model_dir / self._metadata_file
            model_path = self._model_dir / self._active_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            self._metadata = ModelMetadata.from_file(meta_path)
            self._model = load_model(model_path, compile=False)
            elapsed = time.perf_counter() - start
            self._load_seconds = elapsed
            self._metrics.model_load_seconds.observe(elapsed)
            self._metrics.active_model_info.labels(model_version=self.version).set(1)
            logger.info("Loaded model %s in %.2fs", self._active_file, elapsed)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            return self._model.predict(batch, verbose=0)

    def switch_version(self, new_file: str) -> tuple[str, str]:
        from tensorflow.keras.models import load_model
        new_path = self._model_dir / new_file
        if not new_path.exists():
            raise FileNotFoundError(f"Candidate model not found: {new_path}")
        previous = self._active_file
        start = time.perf_counter()
        candidate = load_model(new_path, compile=False)  # load outside lock to avoid blocking inference
        with self._lock:
            self._metrics.active_model_info.labels(model_version=self.version).set(0)
            self._model = candidate
            self._active_file = new_file
            elapsed = time.perf_counter() - start
            self._metrics.model_switch_seconds.observe(elapsed)
            self._metrics.active_model_info.labels(model_version=self.version).set(1)
            logger.info("Switched model %s -> %s in %.2fs", previous, new_file, elapsed)
        return Path(previous).stem, self.version

    def status(self) -> dict:
        meta = self._metadata
        return {
            "model_version": self.version,
            "loaded": self.loaded,
            "classes": meta.class_names if meta else [],
            "image_size": list(meta.image_size) if meta else [],
            "channels": meta.channels if meta else 0,
            "load_seconds": self._load_seconds,
        }
