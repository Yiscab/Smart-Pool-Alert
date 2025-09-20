# PoolWatch – Refactored Version
# Purpose: Clean, production-style version of risk-alert system (step 3)
# Notes:
# - Separated constants
# - Reduced magic numbers
# - Improved naming and function boundaries
# - No ChatGPT comments or obvious AI patterns

import os
import time
import json
import math
import cv2
import numpy as np
import simpleaudio as sa
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

# ---------------- Constants ----------------
DEFAULT_VIDEO = "child_near_pool.mp4"
DEFAULT_ROI_PATH = "pool_roi.json"
DEFAULT_MODEL = "yolov8n.pt"

# Risk parameters
IMMOBILE_SECONDS = 1.0
IMMOBILE_PIXELS = 18
HORIZONTAL_AR_MIN = 1.1
SUBMERGE_DROP_FRAC = 0.28
RISK_SCORE_THRESHOLD = 1.2

# Display
WINDOW_NAME = "PoolWatch – Step 3"
COLORS = {
    "text": (255, 255, 255),
    "roi": (0, 255, 255),
    "ok": (0, 200, 0),
    "risk": (0, 0, 255)
}

# ---------------- Config ----------------
CAMERA_SOURCE = 0 if os.getenv("POOL_CAM", DEFAULT_VIDEO) == "0" else os.getenv("POOL_CAM", DEFAULT_VIDEO)
ROI_PATH = os.getenv("POOL_ROI", DEFAULT_ROI_PATH)
DETECTOR_MODE = os.getenv("DETECTOR_MODE", "toy").lower()
YOLO_MODEL = os.getenv("YOLO_MODEL", DEFAULT_MODEL)

# ---------------- ROI ----------------
def load_roi(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sanitize_roi(np.array(data, dtype=np.int32))

def sanitize_roi(pts: np.ndarray) -> np.ndarray:
    cleaned = []
    for p in pts.reshape(-1, 2).tolist():
        tup = (int(p[0]), int(p[1]))
        if not cleaned or tup != cleaned[-1]:
            cleaned.append(tup)
    if len(cleaned) >= 3 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return np.array(cleaned, dtype=np.int32)

def point_inside_roi(pt: Tuple[int, int], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, pt, False) >= 0

# ---------------- Detection ----------------
class ToyRedDetector:
    def predict(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
        mask = cv2.morphologyEx(mask1 | mask2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(x, y, x+w, y+h, 0.9, 0) for cnt in contours if (cv2.contourArea(cnt) >= 500)
                for (x, y, w, h) in [cv2.boundingRect(cnt)]]

# ---------------- Tracking ----------------
@dataclass
class Track:
    centroids: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=120))
    heights: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

# ---------------- Helpers ----------------
def center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def wh(x1, y1, x2, y2):
    return x2 - x1, y2 - y1

def path_length(pts: Deque[Tuple[float, float]]) -> float:
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, list(pts)[1:]))

def seconds_span(ts: Deque[float]) -> float:
    return ts[-1] - ts[0] if len(ts) >= 2 else 0.0

# ---------------- Alert ----------------
def trigger_alert(sound_path="distress_alert.wav"):
    try:
        wave_obj = sa.WaveObject.from_wave_file(sound_path)
        play_obj = wave_obj.play()
    except Exception as e:
        print(f"[ERROR] Sound failed: {e}")

# ---------------- Main ----------------

def main():
    roi = load_roi(ROI_PATH)
    if roi.shape[0] < 3:
        raise ValueError("ROI must have ≥ 3 points")

    # גבולות קצה הבריכה (לשנות לפי התמונה שלך)
    EDGE_Y_MIN = 340  # גבול עליון של שפת הבריכה
    EDGE_Y_MAX = 420  # גבול תחתון של שפת הבריכה

    detector = ToyRedDetector()
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {CAMERA_SOURCE}")
    is_file = isinstance(CAMERA_SOURCE, str) and os.path.exists(CAMERA_SOURCE)

    tracks: Dict[int, Track] = defaultdict(Track)
    next_id = 1
    alert_triggered = False
    alert_start_time = 0
    prev_t = time.time(); frames = 0; fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        now = time.time()
        frames += 1
        if now - prev_t >= 1.0:
            fps = frames / (now - prev_t)
            frames = 0
            prev_t = now

        vis = frame.copy()
        cv2.polylines(vis, [roi], True, COLORS["roi"], 2)
        detections = detector.predict(frame)

        for (x1, y1, x2, y2, score, cls) in detections:
            cx, cy = center(x1, y1, x2, y2)
            if not point_inside_roi((int(cx), int(cy)), roi):
                continue

            # --- התרעה על קצה הבריכה ---
            if EDGE_Y_MIN < cy < EDGE_Y_MAX:
                if not alert_triggered:
                    trigger_alert()
                    alert_triggered = True
                    alert_start_time = now
                if now - alert_start_time < 3:
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), COLORS["risk"], -1)
                    vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
                    cv2.putText(vis, "ALERT: EDGE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLORS["risk"], 4)
            # --- סוף התרעה קצה ---
            best_tid, best_d = None, 1e9
            for tid, tr in tracks.items():
                if tr.centroids:
                    d = math.hypot(cx - tr.centroids[-1][0], cy - tr.centroids[-1][1])
                    if d < best_d:
                        best_d, best_tid = d, tid
            tid = next_id if best_tid is None or best_d > 80 else best_tid
            if tid == next_id:
                next_id += 1
            tr = tracks[tid]
            tr.centroids.append((cx, cy))
            w, h = wh(x1, y1, x2, y2)
            tr.heights.append(h)
            tr.timestamps.append(now)


            if len(tr.timestamps) < 10:
                continue

            move = path_length(tr.centroids)
            span = seconds_span(tr.timestamps)
            ar = w / max(h, 1e-3)
            submerge = False
            if len(tr.heights) >= 10:
                h_now = float(np.median(list(tr.heights)[-5:]))
                h_prev = float(np.median(list(tr.heights)[:5]))
                submerge = h_prev > 0 and ((h_prev - h_now) / h_prev >= SUBMERGE_DROP_FRAC)

            score = 0.0
            if span >= IMMOBILE_SECONDS and move <= IMMOBILE_PIXELS:
                score += 1.2
            if ar >= HORIZONTAL_AR_MIN:
                score += 0.6
            if submerge:
                score += 0.6

            color = COLORS["ok"] if score < RISK_SCORE_THRESHOLD else COLORS["risk"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"score={score:.1f} mv={move:.0f}px ar={ar:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if score >= RISK_SCORE_THRESHOLD and span >= IMMOBILE_SECONDS:
                if not alert_triggered:
                    trigger_alert()
                    alert_triggered = True
                    alert_start_time = now
                if now - alert_start_time < 3:
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), COLORS["risk"], -1)
                    vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
                    cv2.putText(vis, "ALERT", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLORS["risk"], 4)

        # HUD
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 2)
        cv2.putText(vis,
                    f"IMMOBILE>{IMMOBILE_SECONDS}s mv<{IMMOBILE_PIXELS}px AR>={HORIZONTAL_AR_MIN} DROP>{SUBMERGE_DROP_FRAC}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

        cv2.imshow(WINDOW_NAME, vis)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
