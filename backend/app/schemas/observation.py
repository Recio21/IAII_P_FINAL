from pydantic import BaseModel, Field
from typing import Dict, Optional


class ObservationRequest(BaseModel):
    predicted_emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_version: str
    true_label: Optional[str] = None
    source: Optional[str] = "seeder"


class ObservationResponse(BaseModel):
    stored: bool
    total: int
