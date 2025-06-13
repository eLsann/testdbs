from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    success: bool
    predicted_class: int
    class_name: str
    confidence: float
    all_probabilities: List[float]
    message: str
    
class ModelInfo(BaseModel):
    loaded: bool
    path: Optional[str] = None
    model_type: str
    class_names: List[str]
    input_shape: Optional[str] = None
    output_shape: Optional[str] = None
