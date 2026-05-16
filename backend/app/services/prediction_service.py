import logging
import time
from typing import Dict
import numpy as np
from app.inference.preprocessing import preprocess_image_for_emotion_model
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_service: ModelService, metrics) -> None:
        self._model = model_service
        self._metrics = metrics

    def predict_from_bytes(self, raw: bytes) -> Dict:
        meta = self._model.metadata
        start = time.perf_counter()
        try:
            batch = preprocess_image_for_emotion_model(
                raw, image_size=meta.image_size, channels=meta.channels,
            )
            preds = self._model.predict(batch)[0]
        finally:
            self._metrics.inference_latency.observe(time.perf_counter() - start)

        probs = np.asarray(preds, dtype=np.float32)
        if probs.ndim != 1 or probs.shape[0] != len(meta.class_names):
            raise RuntimeError(
                f"Model output shape {probs.shape} incompatible with classes {len(meta.class_names)}"
            )
        idx = int(np.argmax(probs))
        emotion = meta.class_names[idx]
        confidence = float(probs[idx])
        probabilities = {c: float(p) for c, p in zip(meta.class_names, probs)}
        self._metrics.predictions_total.labels(
            model_version=self._model.version, emotion=emotion,
        ).inc()
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": self._model.version,
        }
