from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.metrics_service import MetricsService
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.services.observation_service import ObservationService
from app.services.drift_service import DriftService
from app.services.dvc_service import DVCService


@dataclass
class Services:
    metrics: MetricsService
    model: ModelService
    prediction: PredictionService
    observation: ObservationService
    drift: DriftService
    dvc: DVCService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    metrics = MetricsService()
    model = ModelService(
        model_dir=settings.model_dir,
        active_file=settings.active_model_file,
        metadata_file=settings.metadata_file,
        metrics=metrics,
    )
    model.load()
    prediction = PredictionService(model, metrics)
    observation = ObservationService(settings.observations_path, metrics)
    drift = DriftService(model.metadata.class_names, metrics=metrics)
    dvc = DVCService(repo_root=settings.model_dir.parent, model_dir=settings.model_dir)
    app.state.services = Services(metrics, model, prediction, observation, drift, dvc)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
