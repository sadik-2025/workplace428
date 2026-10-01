"""
Phase 7 — Handwritten Mathematical Formula Recognition.

Segments individual symbols out of a photo/scan of a handwritten formula
using classical computer vision (OpenCV contour detection), rather than a
learned detector — appropriate scope for this project given the existing
classifier only recognizes single symbols.

Pipeline:
  1. Grayscale + Otsu threshold to separate ink from background.
  2. Find external contours -> bounding boxes.
  3. Merge boxes that likely belong to the SAME symbol but got split into
     multiple contours (e.g. the two bars of "=", or the dot-line-dot of
     "div") — detected by significant horizontal (x-axis) overlap, which
     means the pieces are stacked vertically rather than side by side.
  4. Sort remaining boxes left-to-right (reading order).
  5. Crop each with padding, pad to a square canvas (so thin symbols like
     "1" or "-" aren't squished when later resized to the model's input
     size), ready to hand to predict_image() one at a time.
"""

from PIL import Image

try:
    import cv2
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless and numpy are required for formula recognition. "
        "Install with: pip install opencv-python-headless numpy"
    ) from e

# Maps classifier class names to the symbol shown in the reconstructed expression.
CLASS_TO_SYMBOL = {
    "add": "+",
    "sub": "-",
    "mul": "\u00d7",  # ×
    "div": "\u00f7",  # ÷
    "eq": "=",
    "dec": ".",
}


def to_display_symbol(class_name: str) -> str:
    return CLASS_TO_SYMBOL.get(class_name, class_name)


def _x_overlap_ratio(b1, b2) -> float:
    x1a, x1b = b1[0], b1[0] + b1[2]
    x2a, x2b = b2[0], b2[0] + b2[2]
    inter = max(0, min(x1b, x2b) - max(x1a, x2a))
    union_width = max(x1b, x2b) - min(x1a, x2a)
    return inter / union_width if union_width > 0 else 0


def _merge_close_boxes(boxes, x_overlap_thresh: float = 0.3):
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _x_overlap_ratio(boxes[i], boxes[j]) > x_overlap_thresh:
                    x = min(boxes[i][0], boxes[j][0])
                    y = min(boxes[i][1], boxes[j][1])
                    x2 = max(boxes[i][0] + boxes[i][2], boxes[j][0] + boxes[j][2])
                    y2 = max(boxes[i][1] + boxes[i][3], boxes[j][1] + boxes[j][3])
                    boxes[i] = (x, y, x2 - x, y2 - y)
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _pad_to_square(img: Image.Image, background: int = 255) -> Image.Image:
    w, h = img.size
    size = max(w, h)
    square = Image.new("L", (size, size), color=background)
    square.paste(img, ((size - w) // 2, (size - h) // 2))
    return square


def segment_symbols(pil_image: Image.Image, min_area: int = 30, padding: int = 10):
    """Returns a list of {"bbox": [x, y, w, h], "image": PIL.Image} in
    left-to-right reading order, ready for classification."""
    gray = pil_image.convert("L")
    arr = np.array(gray)

    # Otsu threshold, inverted so ink (assumed darker than background) -> white
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    boxes = _merge_close_boxes(boxes)
    boxes.sort(key=lambda b: b[0])  # left to right

    h_img, w_img = arr.shape
    results = []
    for (x, y, w, h) in boxes:
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)
        crop = gray.crop((x0, y0, x1, y1))
        crop = _pad_to_square(crop)
        results.append({"bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], "image": crop})

    return results
