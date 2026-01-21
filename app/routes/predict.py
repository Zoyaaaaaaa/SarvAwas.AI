from fastapi import APIRouter, HTTPException
from app.schemas.request_response import PredictRequest, PredictResponse, ExplanationResponse
from app.models.loader import store
from app.services.feature_engineering import preprocess_input, generate_verdict_reasons
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    start_time = time.time()
    try:
        # 1. Feature Engineering
        input_data = req.dict()
        input_df = preprocess_input(input_data, store.metadata)
        
        # 2. Inference
        comp_prob = float(store.models['completion_probability'].predict(input_df)[0])
        delay_class = int(store.models['delay_risk_class'].predict(input_df)[0])
        
        # 3. Post-processing
        risk_level = "LOW"
        if comp_prob < 0.4: risk_level = "HIGH"
        elif comp_prob < 0.7: risk_level = "MEDIUM"
        
        delay_labels = ["MINIMAL", "SHORT", "MODERATE", "SEVERE"]
        
        reasons = generate_verdict_reasons(comp_prob, input_data)
        
        # 4. Observability
        latency = (time.time() - start_time) * 1000
        logger.info(f"Prediction for district {req.district} completed in {latency:.2f}ms")
        
        return PredictResponse(
            risk_level=risk_level,
            completion_probability=round(comp_prob, 2),
            delay_risk=delay_labels[delay_class],
            confidence_score=round(comp_prob * 100, 1),
            key_factors=reasons,
            model_version="1.0.0-prod"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

@router.post("/explain", response_model=ExplanationResponse)
async def explain(req: PredictRequest):
    if not store.is_loaded or not store.shap_explainer:
        raise HTTPException(status_code=503, detail="Explanation engine not available")
    
    try:
        input_df = preprocess_input(req.dict(), store.metadata)
        shap_values = store.shap_explainer(input_df)
        
        importance = []
        for i, col in enumerate(store.metadata['feature_cols']):
            importance.append({
                "feature": col,
                "effect": float(shap_values.values[0][i]),
                "description": f"{'Increases' if shap_values.values[0][i] > 0 else 'Decreases'} completion risk"
            })
            
        return ExplanationResponse(
            base_value=float(shap_values.base_values[0]),
            feature_importance=sorted(importance, key=lambda x: abs(x['effect']), reverse=True)
        )
    except Exception as e:
        logger.error(f"SHAP generation failed: {e}")
        raise HTTPException(status_code=500, detail="Explanation error")
