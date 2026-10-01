import io
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, ImageFilter, ImageOps
from sqlalchemy.orm import Session

from .. import models, schemas
from ..auth import get_current_user
from ..database import get_db
from ..ml_inference import predict_image

router = APIRouter(prefix="/predict", tags=["prediction"])

# Transformations applied for the "Model Robustness Analysis" feature — the
# same uploaded image is tested under each of these and the results
# compared, so you can see how confidence/correctness holds up.
ROBUSTNESS_TRANSFORMS = {
    "original": lambda img: img,
    "rotated_+15": lambda img: img.rotate(15),
    "rotated_-15": lambda img: img.rotate(-15),
    "inverted": lambda img: ImageOps.invert(img.convert("L")),
    "blurred": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
    "high_contrast": lambda img: ImageOps.autocontrast(img.convert("L")),
}


def _read_image(contents: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


@router.post("", response_model=schemas.PredictionOut)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("ssl_domain_gen"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    image = _read_image(await file.read())

    try:
        result = predict_image(image, model_name=model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prediction = models.Prediction(
        user_id=current_user.id,
        model_used=result["model_used"],
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        class_probabilities=json.dumps(result["class_probabilities"]),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return schemas.PredictionOut(
        id=prediction.id,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        top_k=result["top_k"],
        class_probabilities=result["class_probabilities"],
        model_used=result["model_used"],
        created_at=prediction.created_at,
    )


@router.post("/robustness")
async def predict_robustness(
    file: UploadFile = File(...),
    model_name: str = Form("ssl_domain_gen"),
    current_user: models.User = Depends(get_current_user),
):
    image = _read_image(await file.read())

    try:
        results = {
            name: predict_image(transform_fn(image), model_name=model_name, top_k=1)
            for name, transform_fn in ROBUSTNESS_TRANSFORMS.items()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"model_used": model_name, "transformations": results}
