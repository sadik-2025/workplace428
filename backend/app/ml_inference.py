"""
Loads the three models trained in Phases 2-4 (baseline, augmented,
ssl_domain_gen) and exposes a single predict_image() function used by the
prediction endpoints.

Reuses the exact model architectures from ml/scripts/ (model_baseline.py,
model_ssl.py) rather than duplicating them, so the backend always matches
whatever was actually trained — add ml/scripts to sys.path and import
directly from there.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# app/ml_inference.py -> app/ -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SCRIPTS_DIR = PROJECT_ROOT / "ml" / "scripts"
ML_MODELS_DIR = PROJECT_ROOT / "ml" / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
if not(SPLITS_DIR / "class_to_idx.json").exists(): SPLITS_DIR = PROJECT_ROOT / "ml" / "models"
sys.path.insert(0, str(ML_SCRIPTS_DIR))

from model_baseline import BaselineCNN  # noqa: E402
from model_ssl import ClassifierHead, Encoder  # noqa: E402

IMG_SIZE = 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

_class_to_idx_path = SPLITS_DIR / "class_to_idx.json"
if not _class_to_idx_path.exists():
    raise FileNotFoundError(
        f"{_class_to_idx_path} not found. Run any of the ml/scripts training "
        "scripts at least once (they create the split + class list) before "
        "starting the backend."
    )
with open(_class_to_idx_path) as f:
    CLASS_TO_IDX = json.load(f)
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

_models_cache = {}


def _load_baseline():
    model = BaselineCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(DEVICE)
    model.load_state_dict(
        torch.load(ML_MODELS_DIR / "baseline" / "best_model.pt", map_location=DEVICE)
    )
    model.eval()
    return model


def _load_augmented():
    model = BaselineCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(DEVICE)
    model.load_state_dict(
        torch.load(ML_MODELS_DIR / "augmented" / "best_model.pt", map_location=DEVICE)
    )
    model.eval()
    return model


def _load_ssl_domain_gen():
    encoder = Encoder(img_size=IMG_SIZE).to(DEVICE)
    encoder.load_state_dict(
        torch.load(ML_MODELS_DIR / "ssl_domain_gen" / "best_encoder.pt", map_location=DEVICE)
    )
    classifier = ClassifierHead(encoder.out_dim, NUM_CLASSES).to(DEVICE)
    classifier.load_state_dict(
        torch.load(ML_MODELS_DIR / "ssl_domain_gen" / "best_classifier.pt", map_location=DEVICE)
    )
    encoder.eval()
    classifier.eval()
    return {"encoder": encoder, "classifier": classifier}


MODEL_LOADERS = {
    "baseline": _load_baseline,
    "augmented": _load_augmented,
    "ssl_domain_gen": _load_ssl_domain_gen,
}


def get_model(model_name: str):
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_LOADERS.keys())}")
    if model_name not in _models_cache:
        # Lazy-loaded on first request, then cached in memory for the life
        # of the server process.
        _models_cache[model_name] = MODEL_LOADERS[model_name]()
    return _models_cache[model_name]


@torch.no_grad()
def predict_image(image: Image.Image, model_name: str = "ssl_domain_gen", top_k: int = 5):
    image = image.convert("L")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    model = get_model(model_name)
    if model_name == "ssl_domain_gen":
        feats = model["encoder"](tensor)
        logits = model["classifier"](feats)
    else:
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    k = min(top_k, NUM_CLASSES)
    top_probs, top_idxs = torch.topk(probs, k=k)

    predicted_idx = int(top_idxs[0])
    predicted_class = IDX_TO_CLASS[predicted_idx]
    confidence = float(top_probs[0])

    top_k_list = [
        {"class": IDX_TO_CLASS[int(idx)], "probability": float(p)}
        for idx, p in zip(top_idxs, top_probs)
    ]
    class_probabilities = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_k_list,
        "class_probabilities": class_probabilities,
        "model_used": model_name,
    }
