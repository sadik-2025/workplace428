from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..auth import get_current_user
from ..database import get_db

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("")
def get_stats(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Powers the interactive Dashboard. Covers 6 of the suggested
    components: total predictions, most frequent symbol, prediction
    distribution by class, average confidence, predictions over time, and
    confidence distribution."""
    predictions = db.query(models.Prediction).all()
    total = len(predictions)

    if total == 0:
        return {
            "total_predictions": 0,
            "most_frequent_symbol": None,
            "prediction_distribution": {},
            "average_confidence": None,
            "predictions_over_time": {},
            "confidence_distribution": {},
        }

    class_counts = Counter(p.predicted_class for p in predictions)
    most_frequent = class_counts.most_common(1)[0][0]
    avg_confidence = sum(p.confidence for p in predictions) / total

    by_day = Counter(p.created_at.strftime("%Y-%m-%d") for p in predictions)

    buckets = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for p in predictions:
        c = p.confidence
        if c < 0.5:
            buckets["0.0-0.5"] += 1
        elif c < 0.7:
            buckets["0.5-0.7"] += 1
        elif c < 0.9:
            buckets["0.7-0.9"] += 1
        else:
            buckets["0.9-1.0"] += 1

    return {
        "total_predictions": total,
        "most_frequent_symbol": most_frequent,
        "prediction_distribution": dict(class_counts),
        "average_confidence": avg_confidence,
        "predictions_over_time": dict(sorted(by_day.items())),
        "confidence_distribution": buckets,
    }
