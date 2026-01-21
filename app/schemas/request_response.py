from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date

class PredictRequest(BaseModel):
    # Core project details (UI)
    project_name: str = Field(..., example="Four Seasons Private Residences")
    district: str = Field(..., example="Mumbai Suburban")
    taluka: Optional[str] = Field(None, example="Borivali")
    village: Optional[str] = Field(None, example="Versova")
    
    # Location
    pin_code: str = Field(..., example="400014")
    lat: Optional[float] = Field(None, example=19.0147)
    lng: Optional[float] = Field(None, example=72.8777)
    
    # Project stats
    total_units: int = Field(..., gt=0, example=30)
    booked_units: int = Field(..., ge=0, example=20)
    project_area: float = Field(..., gt=0, example=15000)
    
    # Structural (UI usually calls floors/FSI, we map them)
    fsi: float = Field(2.5, gt=0, example=2.5) # Default if not in UI explicitly, though UI has stats
    floors: int = Field(15, gt=0, example=15) # Default/Optional in UI sometimes
    
    # Dates
    proposed_completion_date: Optional[str] = Field(None, example="2026-12-31")
    revised_completion_date: Optional[str] = Field(None, example="2027-06-30")
    extended_completion_date: Optional[str] = Field(None, example="2028-01-01")
    
    # Legal
    cases: int = Field(0, ge=0, example=0)
    
    # Manual overrides (optional)
    has_delay: Optional[bool] = None
    has_extension: Optional[bool] = None

class PredictResponse(BaseModel):
    risk_level: str
    completion_probability: float
    delay_risk: str
    confidence_score: float
    key_factors: List[str]
    model_version: str

class MapPredictRequest(BaseModel):
    lat: float
    lng: float

class MapPredictResponse(BaseModel):
    location: Dict[str, float]
    nearby_risk_index: float
    district_context: str
    predicted_rera_safety: str
    message: str

class ExplanationResponse(BaseModel):
    base_value: float
    feature_importance: List[Dict]
