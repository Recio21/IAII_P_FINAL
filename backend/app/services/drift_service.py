from collections import deque, Counter
from typing import Iterable
import math
import threading


class DriftService:
    """Drift básico simulado: KL-like divergence entre distribución de clases
    de la ventana reciente vs distribución de referencia uniforme.
    Sustituible por estadísticas reales (PSI, KS) cuando haya labels reales.
    """

    def __init__(self, classes: list[str], window: int = 200, metrics=None) -> None:
        self._classes = classes
        self._window: deque[str] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._metrics = metrics

    def update(self, emotion: str) -> float:
        with self._lock:
            self._window.append(emotion)
            score = self._compute_score()
            if self._metrics is not None:
                self._metrics.drift_score.set(score)
            return score

    def _compute_score(self) -> float:
        if not self._window or not self._classes:
            return 0.0
        n = len(self._window)
        counts = Counter(self._window)
        eps = 1e-9
        uniform = 1.0 / len(self._classes)
        kl = 0.0
        for c in self._classes:
            p = counts.get(c, 0) / n
            kl += (p + eps) * math.log((p + eps) / (uniform + eps))
        return max(0.0, min(1.0, kl))
