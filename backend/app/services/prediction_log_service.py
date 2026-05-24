import json
import threading
import time
from pathlib import Path
from typing import List, Dict


class PredictionLogService:
    """Persistencia append-only de cada predicción en JSONL.
    Permite reconstruir el historial completo independientemente de Prometheus.
    Thread-safe mediante lock; lecturas con tail eficiente para grandes ficheros.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict) -> None:
        entry = {"ts": time.time(), **record}
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self, limit: int = 500, offset: int = 0) -> List[Dict]:
        if not self._path.exists():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        # Orden cronológico inverso (más reciente primero)
        lines = list(reversed(lines))
        page = lines[offset: offset + limit]
        records = []
        for line in page:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def count(self) -> int:
        if not self._path.exists():
            return 0
        with self._lock:
            with self._path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
