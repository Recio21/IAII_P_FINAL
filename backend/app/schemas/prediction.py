from pydantic import BaseModel, Field
from typing import Dict


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    app: str
    env: str


class ModelStatusResponse(BaseModel):
    model_version: str
    loaded: bool
    classes: list[str]
    image_size: list[int]
    channels: int
    load_seconds: float | None = None


class SwitchModelRequest(BaseModel):
    model_file: str


class SwitchModelResponse(BaseModel):
    previous_version: str
    current_version: str
    switched: bool


class RetrainResponse(BaseModel):
    triggered: bool
    detail: str
