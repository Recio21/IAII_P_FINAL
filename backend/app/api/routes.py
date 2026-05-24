from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Response, Query
from pydantic import BaseModel
from app.schemas.prediction import (
    PredictionResponse, HealthResponse, ModelStatusResponse,
    SwitchModelRequest, SwitchModelResponse, RetrainResponse,
)
from app.schemas.observation import ObservationRequest, ObservationResponse
from app.inference.preprocessing import ImageValidationError
from app.core.config import get_settings

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


@router.get("/health", response_model=HealthResponse)
def health():
    s = get_settings()
    return HealthResponse(status="ok", app=s.app_name, env=s.app_env)


@router.get("/model/status", response_model=ModelStatusResponse)
def model_status(svc=Depends(_services)):
    return ModelStatusResponse(**svc.model.status())


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), svc=Depends(_services)):
    settings = get_settings()
    if not (file.content_type or "").startswith("image/"):
        svc.metrics.errors_total.labels(endpoint="/predict", kind="content_type").inc()
        raise HTTPException(status_code=415, detail="Expected image/* content-type")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > settings.max_upload_bytes:
        svc.metrics.errors_total.labels(endpoint="/predict", kind="too_large").inc()
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        result = svc.prediction.predict_from_bytes(raw)
    except ImageValidationError as exc:
        svc.metrics.errors_total.labels(endpoint="/predict", kind="invalid_image").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        svc.metrics.errors_total.labels(endpoint="/predict", kind="inference").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    svc.drift.update(result["emotion"])
    svc.prediction_log.append({
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "model_version": result["model_version"],
        "filename": file.filename or "unknown",
    })
    return PredictionResponse(**result)


@router.post("/model/switch", response_model=SwitchModelResponse)
def switch_model(req: SwitchModelRequest, svc=Depends(_services)):
    try:
        previous, current = svc.model.switch_version(req.model_file)
    except FileNotFoundError as exc:
        svc.metrics.errors_total.labels(endpoint="/model/switch", kind="not_found").inc()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        svc.metrics.errors_total.labels(endpoint="/model/switch", kind="switch").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SwitchModelResponse(previous_version=previous, current_version=current, switched=True)


class RetrainRequest(BaseModel):
    epochs: int = 5

@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest = RetrainRequest(), svc=Depends(_services)):
    result = svc.github_sync.trigger_training(epochs=req.epochs)
    return RetrainResponse(triggered=result["ok"], detail=result["message"])


@router.post("/observations", response_model=ObservationResponse)
def observations(obs: ObservationRequest, svc=Depends(_services)):
    total = svc.observation.append(obs.model_dump())
    return ObservationResponse(stored=True, total=total)


@router.get("/predictions/history")
def predictions_history(
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    svc=Depends(_services),
):
    records = svc.prediction_log.read_all(limit=limit, offset=offset)
    total = svc.prediction_log.count()
    return {"total": total, "limit": limit, "offset": offset, "records": records}


@router.post("/models/sync")
def sync_models(svc=Depends(_services)):
    """Descarga desde GitHub Releases los modelos que aún no están en /app/models."""
    result = svc.github_sync.sync()
    return result


@router.get("/models/list")
def list_models(svc=Depends(_services)):
    import json as _json
    settings = get_settings()
    active_status = svc.model.status()
    active_file = active_status.get("model_version", "")
    models = []
    for f in sorted(settings.model_dir.glob("*.keras")):
        is_active = f.name == active_file or f.stem == active_file
        # Extract version number to find matching metadata file
        parts = f.stem.split("_")  # model_7_final → ['model', '7', 'final']
        version = parts[1] if len(parts) >= 2 else None
        meta = {}
        if version:
            meta_path = settings.model_dir / f"metadata_{version}.json"
            if meta_path.exists():
                try:
                    meta = _json.loads(meta_path.read_text())
                except Exception:
                    pass
        models.append({
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "active": is_active,
            "val_accuracy": meta.get("val_accuracy"),
            "val_loss": meta.get("val_loss"),
            "epochs_trained": meta.get("epochs_trained"),
            "train_seconds": meta.get("train_seconds"),
        })
    return {"models": models, "active": active_file}


@router.get("/metrics")
def metrics(svc=Depends(_services)):
    data, ctype = svc.metrics.render()
    return Response(content=data, media_type=ctype)
