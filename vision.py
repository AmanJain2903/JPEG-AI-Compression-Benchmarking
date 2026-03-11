"""
vision.py — Machine Vision Test Engine
=======================================
Wraps Ultralytics YOLOv8 to run object detection on original and
compressed images, enabling comparison of detection confidence degradation
introduced by different codecs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """Single bounding-box detection."""

    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class DetectionResult:
    """All detections for one image variant."""

    label: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def num_detections(self) -> int:
        return len(self.detections)


# ---------------------------------------------------------------------------
# YOLOv8 wrapper
# ---------------------------------------------------------------------------

_yolo_model: YOLO | None = None


def _get_yolo() -> YOLO:
    """Lazy-load the YOLOv8-nano model (smallest, fast for benchmarking)."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def run_detection(img: Image.Image, label: str = "image", conf_threshold: float = 0.25) -> DetectionResult:
    """
    Run YOLOv8 inference on a PIL image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (RGB).
    label : str
        Human-readable label for this variant (e.g. "Original", "JPEG q=50").
    conf_threshold : float
        Minimum confidence for a detection to be included.

    Returns
    -------
    DetectionResult
        Structured detections with class names, confidences, and boxes.
    """
    model = _get_yolo()
    results = model(np.asarray(img.convert("RGB")), verbose=False, conf=conf_threshold)

    detections: list[Detection] = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].tolist()
            detections.append(
                Detection(
                    class_name=r.names[cls_id],
                    confidence=conf,
                    box=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                )
            )

    return DetectionResult(label=label, detections=detections)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_detections(
    original: Image.Image,
    variants: dict[str, Image.Image],
    conf_threshold: float = 0.25,
) -> list[DetectionResult]:
    """
    Run YOLO on the original image and every variant, returning a list
    of ``DetectionResult`` objects suitable for tabular display.

    Parameters
    ----------
    original : PIL.Image.Image
        The uncompressed source image.
    variants : dict[str, PIL.Image.Image]
        Mapping of codec label -> reconstructed image.
    conf_threshold : float
        Detection confidence threshold.
    """
    results = [run_detection(original, label="Original", conf_threshold=conf_threshold)]
    for label, img in variants.items():
        results.append(run_detection(img, label=label, conf_threshold=conf_threshold))
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_PALETTE = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D",
    "#CFD231", "#48F90A", "#92CC17", "#3DDB86",
    "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF",
    "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def draw_detections(img: Image.Image, det_result: DetectionResult) -> Image.Image:
    """
    Draw bounding boxes and labels onto a copy of *img*.

    Returns a new PIL image with annotated boxes so the original is
    never mutated.
    """
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, img.height // 40))
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, det in enumerate(det_result.detections):
        color = _PALETTE[idx % len(_PALETTE)]
        x1, y1, x2, y2 = det.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 300))

        label_text = f"{det.class_name} {det.confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), label_text, fill="white", font=font)

    return annotated
