from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, map
from app.models.loader import store
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# Setup Logging for Observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rera_production")

app = FastAPI(
    title="RERA Risk Prediction API",
    description="Production-grade explainable AI for real estate project safety.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten this in real production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument for Prometheus metrics (Observability)
Instrumentator().instrument(app).expose(app)

# Lifecycle events
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing RERA Risk Engine...")
    store.load()
    if store.is_loaded:
        logger.info("RERA Risk Engine initialized successfully.")
    else:
        logger.error("RERA Risk Engine failed to initialize correctly.")

# Routes
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(map.router, prefix="/api/v1/predict", tags=["Map"])

@app.get("/")
async def root():
    return {
        "name": "RERA Risk Prediction API",
        "status": "online",
        "engine_ready": store.is_loaded
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if store.is_loaded else "degraded",
        "model_loaded": store.is_loaded
    }

# reload trigger