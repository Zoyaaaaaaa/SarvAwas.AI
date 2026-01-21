import pickle
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path(__file__).parent
MODELS_PKL = MODELS_DIR / "models.pkl"
METADATA_PKL = MODELS_DIR / "metadata.pkl"
SHAP_PKL = MODELS_DIR / "shap_explainer.pkl"

class ModelStore:
    def __init__(self):
        self.models = None
        self.metadata = None
        self.shap_explainer = None
        self.is_loaded = False

    def load(self):
        try:
            if MODELS_PKL.exists():
                with open(MODELS_PKL, "rb") as f:
                    self.models = pickle.load(f)
                with open(METADATA_PKL, "rb") as f:
                    self.metadata = pickle.load(f)
                
                # SHAP is large, load on demand or at startup?
                # For free tier, maybe on demand if memory is tight, 
                # but for speed let's try reading it.
                if SHAP_PKL.exists():
                    with open(SHAP_PKL, "rb") as f:
                        self.shap_explainer = pickle.load(f)
                
                self.is_loaded = True
                logger.info("All model artifacts loaded into memory.")
            else:
                logger.error(f"Model file missing at {MODELS_PKL}")
        except Exception as e:
            logger.error(f"Critical error loading models: {e}")

# Global instance
store = ModelStore()
