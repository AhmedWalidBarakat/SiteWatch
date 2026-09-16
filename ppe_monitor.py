"""
Core PPE compliance detection pipeline.

Approach: use a pretrained YOLOv8 model (trained on COCO) purely for person
detection - no custom training or labeled PPE dataset required - then apply
HSV color-based heuristics within each person's head and torso regions to
flag whether a hard hat and high-visibility vest are present.

This is intentionally a lightweight, dependency-free-of-training approach:
a fast first pass suitable for a proof of concept, not a production-grade
learned PPE classifier (see README for that tradeoff).
"""

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class id for "person"

# HSV ranges (OpenCV: H 0-179, S/V 0-255)
HARDHAT_HSV_RANGES = [
    ((15, 80, 120), (35, 255, 255)),   # yellow/orange hard hats
    ((0, 80, 120), (10, 255, 255)),    # red hard hats
    ((100, 80, 80), (130, 255, 255)),  # blue hard hats
    ((0, 0, 180), (179, 40, 255)),     # white hard hats
]
VEST_HSV_RANGES = [
    ((20, 100, 120), (35, 255, 255)),  # hi-vis yellow/green
    ((5, 120, 120), (18, 255, 255)),   # hi-vis orange
]

HEAD_FRACTION = (0.0, 0.22)
TORSO_FRACTION = (0.25, 0.65)

HARDHAT_THRESHOLD = 0.12
VEST_THRESHOLD = 0.15

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def _color_fraction(region: np.ndarray, ranges) -> float:
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    return float(np.count_nonzero(mask)) / mask.size


def analyze_person(image: np.ndarray, box) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    height = y2 - y1
    head = image[y1 + int(height * HEAD_FRACTION[0]): y1 + int(height * HEAD_FRACTION[1]), x1:x2]
    torso = image[y1 + int(height * TORSO_FRACTION[0]): y1 + int(height * TORSO_FRACTION[1]), x1:x2]

    hardhat_score = _color_fraction(head, HARDHAT_HSV_RANGES)
    vest_score = _color_fraction(torso, VEST_HSV_RANGES)

    has_hardhat = hardhat_score >= HARDHAT_THRESHOLD
    has_vest = vest_score >= VEST_THRESHOLD

    if has_hardhat and has_vest:
        status = "Compliant"
    elif has_hardhat:
        status = "Missing vest"
    elif has_vest:
        status = "Missing hard hat"
    else:
        status = "Missing hard hat & vest"

    return {
        "box": (x1, y1, x2, y2),
        "has_hardhat": has_hardhat,
        "has_vest": has_vest,
        "hardhat_score": hardhat_score,
        "vest_score": vest_score,
        "status": status,
    }


STATUS_COLORS = {
    "Compliant": (0, 200, 0),
    "Missing vest": (0, 165, 255),
    "Missing hard hat": (0, 165, 255),
    "Missing hard hat & vest": (0, 0, 255),
}


def run(image: np.ndarray, conf: float = 0.35):
    """Detect people in an image and assess PPE compliance for each.

    Returns (annotated_image_bgr, list_of_person_results).
    """
    model = get_model()
    results = model(image, classes=[PERSON_CLASS_ID], conf=conf, verbose=False)[0]

    annotated = image.copy()
    people = []
    for box in results.boxes.xyxy.cpu().numpy():
        info = analyze_person(image, box)
        people.append(info)

        x1, y1, x2, y2 = info["box"]
        color = STATUS_COLORS[info["status"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, info["status"], (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    return annotated, people
