from fastapi import APIRouter
from app.schemas.request_response import MapPredictRequest, MapPredictResponse
from app.services.geo_utils import reverse_geocode, calculate_spatial_risk
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/map", response_model=MapPredictResponse)
async def map_predict(req: MapPredictRequest):
    # 1. Reverse geocode finding context
    geo_context = await reverse_geocode(req.lat, req.lng)
    
    # 2. Query spatial risk index
    spatial_risk = calculate_spatial_risk(req.lat, req.lng)
    
    # 3. Formulate safety label
    safety = "HIGHLY_SAFE"
    if spatial_risk > 0.3: safety = "MODERATE_RISK"
    elif spatial_risk > 0.6: safety = "CRITICAL_CAUTION"
    
    return MapPredictResponse(
        location={"lat": req.lat, "lng": req.lng},
        nearby_risk_index=spatial_risk,
        district_context=geo_context.get('district', 'Mumbai Suburban'),
        predicted_rera_safety=safety,
        message=f"Cluster analysis shows localized {safety.replace('_', ' ').lower()} profile."
    )
