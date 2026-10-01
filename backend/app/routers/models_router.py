import json
from pathlib import Path

from fastapi import APIRouter, Depends

from .. import models
from ..auth import get_current_user

router = APIRouter(prefix="/models", tags=["models"])

# app/routers/models_router.py -> routers/ -> app/ -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_MODELS_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_METRIC_PATHS = {
    "baseline": ML_MODELS_DIR / "baseline" / "metrics.json",
    "augmented": ML_MODELS_DIR / "augmented" / "metrics.json",
    "ssl_domain_gen": ML_MODELS_DIR / "ssl_domain_gen" / "metrics.json",
}


@router.get("/compare")
def compare_models(current_user: models.User = Depends(get_current_user)):
    """Powers the Model Performance Dashboard — serves the metrics.json
    saved by each ml/scripts training script (accuracy, precision, recall,
    F1, inference time, parameter count, model size, and the
    standard-vs-unseen-domain generalization numbers)."""
    results = {}
    for name, path in MODEL_METRIC_PATHS.items():
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
        else:
            results[name] = None
    return results
