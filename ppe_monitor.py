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
    ((15, 60, 40), (35, 255, 255)),    # yellow/orange hard hats (lowered S/V minimums so hats read correctly under dim indoor lighting, not just direct sun)
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
VEST_THRESHOLD = 0.25

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def _color_fraction(region: np.ndarray, ranges, exclude_mask: np.ndarray = None) -> float:
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    if exclude_mask is not None:
        mask[exclude_mask] = 0
        valid = np.count_nonzero(~exclude_mask)
        return float(np.count_nonzero(mask)) / valid if valid else 0.0
    return float(np.count_nonzero(mask)) / mask.size


def _overlap_mask(shape, x1, y1, other_boxes) -> np.ndarray:
    """Pixels in a crop that also fall inside another detected person's box.

    Two people standing close together often have overlapping boxes, so a
    naive per-box color check can pick up a neighbor's hard hat or vest.
    Excluding those shared pixels keeps each person's score to their own
    body only.
    """
    mask = np.zeros(shape[:2], dtype=bool)
    ry2, rx2 = shape[0] + y1, shape[1] + x1
    for ox1, oy1, ox2, oy2 in other_boxes:
        ix1, iy1 = max(ox1, x1) - x1, max(oy1, y1) - y1
        ix2, iy2 = min(ox2, rx2) - x1, min(oy2, ry2) - y1
        if ix1 < ix2 and iy1 < iy2:
            mask[max(0, iy1):iy2, max(0, ix1):ix2] = True
    return mask


def analyze_person(image: np.ndarray, box, other_boxes=()) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    height = y2 - y1
    head_y1, head_y2 = y1 + int(height * HEAD_FRACTION[0]), y1 + int(height * HEAD_FRACTION[1])
    torso_y1, torso_y2 = y1 + int(height * TORSO_FRACTION[0]), y1 + int(height * TORSO_FRACTION[1])
    head = image[head_y1:head_y2, x1:x2]
    torso = image[torso_y1:torso_y2, x1:x2]

    head_exclude = _overlap_mask(head.shape, x1, head_y1, other_boxes)
    torso_exclude = _overlap_mask(torso.shape, x1, torso_y1, other_boxes)

    hardhat_score = _color_fraction(head, HARDHAT_HSV_RANGES, head_exclude)
    vest_score = _color_fraction(torso, VEST_HSV_RANGES, torso_exclude)

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
    all_boxes = [tuple(int(v) for v in b) for b in results.boxes.xyxy.cpu().numpy()]
    people = []
    for i, box in enumerate(all_boxes):
        other_boxes = all_boxes[:i] + all_boxes[i + 1:]
        info = analyze_person(image, box, other_boxes)
        people.append(info)

        x1, y1, x2, y2 = info["box"]
        color = STATUS_COLORS[info["status"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, info["status"], (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    return annotated, people
