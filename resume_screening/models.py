from pydantic import BaseModel
from typing import Dict

class ScoreResponse(BaseModel):
    score: float
    details: Dict[str, float]
