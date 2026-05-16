import json
import threading
import time
from pathlib import Path


class ObservationService:
    """Persistencia simple en JSONL append-only. Sin DB.
    Pensado para alimentar drift y futuras métricas de calidad.
    """

    def __init__(self, path: Path, metrics) -> None:
        self._path = path
        self._metrics = metrics
        self._lock = threading.Lock()
        self._count = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                self._count = sum(1 for _ in f)

    def append(self, payload: dict) -> int:
        record = {"ts": time.time(), **payload}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._count += 1
            self._metrics.observations_total.inc()
        return self._count

    @property
    def total(self) -> int:
        return self._count
