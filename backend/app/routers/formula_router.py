import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from .. import models
from ..auth import get_current_user
from ..formula_utils import segment_symbols, to_display_symbol
from ..ml_inference import predict_image

router = APIRouter(prefix="/predict", tags=["formula"])


@router.post("/formula")
async def predict_formula(
    file: UploadFile = File(...),
    model_name: str = Form("ssl_domain_gen"),
    current_user: models.User = Depends(get_current_user),
):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    crops = segment_symbols(image)
    if not crops:
        raise HTTPException(
            status_code=400,
            detail="No symbols detected. Try a clearer photo with good contrast between ink and background.",
        )

    segments = []
    expression_parts = []
    for c in crops:
        try:
            result = predict_image(c["image"], model_name=model_name, top_k=1)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        segments.append(
            {
                "bbox": c["bbox"],
                "predicted_class": result["predicted_class"],
                "display_symbol": to_display_symbol(result["predicted_class"]),
                "confidence": result["confidence"],
            }
        )
        expression_parts.append(to_display_symbol(result["predicted_class"]))

    return {
        "model_used": model_name,
        "segments": segments,
        "reconstructed_expression": "".join(expression_parts),
    }
