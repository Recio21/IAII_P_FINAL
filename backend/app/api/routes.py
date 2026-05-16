from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Response
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


@router.post("/retrain", response_model=RetrainResponse)
def retrain(svc=Depends(_services)):
    # Hook placeholder: el reentrenamiento real es offline (notebook/pipeline DVC).
    # Aquí sólo se registra la intención y se prepara para pull del nuevo artefacto.
    ok, detail = svc.dvc.pull()
    return RetrainResponse(
        triggered=True,
        detail=f"Retrain trigger received. dvc_pull_ok={ok}. {detail or ''}".strip(),
    )


@router.post("/observations", response_model=ObservationResponse)
def observations(obs: ObservationRequest, svc=Depends(_services)):
    total = svc.observation.append(obs.model_dump())
    return ObservationResponse(stored=True, total=total)


@router.get("/metrics")
def metrics(svc=Depends(_services)):
    data, ctype = svc.metrics.render()
    return Response(content=data, media_type=ctype)
