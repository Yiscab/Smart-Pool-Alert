"""
PoolWatch – Step 2 (YOLO person-in-ROI, with toy mode)
------------------------------------------------------
Goal: Detect a person only inside the pool ROI. Supports two detector modes:
- "yolo" : Ultralytics YOLOv8 person class (0) – for real videos/cameras.
- "toy"  : Simple red-object detector – for testing with the synthetic test_pool.mp4 from Step 0.

Run
1) pip install ultralytics opencv-python numpy
2) Ensure you have roi.json from Step 1.5 (clicks → save)
3) Set CAMERA_SOURCE below (file path, 0 for webcam, or RTSP URL)
4) Optional: set DETECTOR_MODE = "toy" when testing the synthetic video
5) python step2_person_in_roi.py

Keys
- M : toggle detector mode (yolo/toy) at runtime
- Q : quit

Notes
- Loads and sanitizes ROI (removes duplicate consecutive points).
- Draws only detections whose centroid is inside the ROI polygon.
- YOLO model is yolov8n.pt by default (downloads on first run).
"""
from __future__ import annotations
import os
import time
import json
from typing import List, Tuple

import cv2
import numpy as np

# Try to import YOLO; allow running toy mode without ultralytics installed
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception:
    pass

# ---------------- Config ----------------
_env = os.getenv("POOL_CAM", "doll_distress.mp4")  # default to the synthetic video
CAMERA_SOURCE = 0 if _env == "0" else _env
ROI_PATH = os.getenv("POOL_ROI", "roi_doll.json")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
DETECTOR_MODE = os.getenv("DETECTOR_MODE", "toy")  # "yolo" or "toy"
CONF_THRES = 0.35
IOU_THRES = 0.45

WINDOW = "PoolWatch – Step 2"

# ---------------- ROI helpers ----------------
def load_roi(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.int32)
    return sanitize_roi(arr)


def sanitize_roi(pts: np.ndarray) -> np.ndarray:
    # remove consecutive duplicates & trailing equal to first
    cleaned: List[Tuple[int,int]] = []
    for p in pts.reshape(-1,2).tolist():
        tup = (int(p[0]), int(p[1]))
        if not cleaned or tup != cleaned[-1]:
            cleaned.append(tup)
    if len(cleaned) >= 3 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return np.array(cleaned, dtype=np.int32)


def point_in_poly(pt: Tuple[int,int], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, pt, False) >= 0


# ---------------- Detectors ----------------
class ToyRedDetector:
    """Simple red blob detector – good for the synthetic Step 0 video."""
    def __init__(self):
        pass

    def predict(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 100, 100]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100]); upper2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []  # list of (x1,y1,x2,y2,score,cls)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append((x, y, x+w, y+h, 0.9, 0))  # fake score, class 0
        return boxes


class YoloPersonDetector:
    def __init__(self, model_path: str, conf: float, iou: float):
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed. pip install ultralytics or use toy mode.")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def predict(self, frame):
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                if cls != 0:  # person only
                    continue
                x1,y1,x2,y2 = [int(v) for v in b.xyxy[0]]
                score = float(b.conf[0]) if b.conf is not None else 0.0
                boxes.append((x1,y1,x2,y2,score,cls))
        return boxes


# ---------------- Main ----------------
def main():
    roi = load_roi(ROI_PATH)
    if roi.shape[0] < 3:
        raise ValueError("ROI must have at least 3 points. Re-run Step 1.5.")

    mode = DETECTOR_MODE.lower()
    detector = None
    if mode == "yolo":
        try:
            detector = YoloPersonDetector(YOLO_MODEL, CONF_THRES, IOU_THRES)
        except Exception as e:
            print(f"[WARN] YOLO not available ({e}). Falling back to toy mode.")
            mode = "toy"
            detector = ToyRedDetector()
    else:
        detector = ToyRedDetector()

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video: {CAMERA_SOURCE}")

    print(f"Running mode={mode} | source={CAMERA_SOURCE} | press M to toggle mode, Q to quit")
    prev_t = time.time(); frames = 0; fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        now = time.time()
        if now - prev_t >= 1.0:
            fps = frames / (now - prev_t)
            frames = 0; prev_t = now

        vis = frame.copy()
        # draw ROI
        cv2.polylines(vis, [roi], True, (0,255,255), 2)

        # detections
        boxes = detector.predict(frame)
        for (x1,y1,x2,y2,score,cls) in boxes:
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            if point_in_poly((cx,cy), roi):
                color = (0,255,0) if mode == "yolo" else (255,0,0)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.circle(vis, (cx,cy), 3, color, -1)
                cv2.putText(vis, f"{('person' if mode=='yolo' else 'blob')} {score:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # HUD
        cv2.putText(vis, f"FPS: {fps:.1f} | mode: {mode} | source: {CAMERA_SOURCE}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, "M=toggle yolo/toy | Q=quit", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('m'), ord('M')):
            # toggle detector
            if mode == "yolo":
                mode = "toy"
                detector = ToyRedDetector()
                print("[INFO] Switched to toy detector")
            else:
                try:
                    detector = YoloPersonDetector(YOLO_MODEL, CONF_THRES, IOU_THRES)
                    mode = "yolo"
                    print("[INFO] Switched to YOLO detector")
                except Exception as e:
                    print(f"[WARN] Cannot switch to YOLO: {e}. Staying in toy mode.")

    cap.release()
    cv2.destroyAllWindows()

