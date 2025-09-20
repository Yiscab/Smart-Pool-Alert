"""
PoolWatch – Step 1.5 (Interactive ROI + Save/Load)
---------------------------------------------------
Goal: Pick the pool ROI by clicking on the video, then save/load it.
No AI yet. This replaces manual editing of coordinates.


Run
1) pip install opencv-python numpy
2) Set CAMERA_SOURCE below (0 for webcam, or RTSP string)
3) python step1_5_interactive_roi.py


Controls
- Left Click : add point
- Z : undo last point
- C / Enter : close polygon (must have ≥ 3 points)
- N : start new polygon (clear)
- S : save ROI to roi.json
- L : load ROI from roi.json (if exists)
- Q : quit


Files
- roi.json : stores list of [x,y] points (int) in image coordinates


Tip
- Click the vertices around the water surface in order (clockwise/counterclockwise).
- Aim for 4–6 points. You can refine later by loading and adding/removing points.
"""

from __future__ import annotations
import os
import json
import time
from typing import List, Tuple


import cv2
import numpy as np

# ---------------- Config ----------------
_env = os.getenv("POOL_CAM", "child_near_pool.mp4")
CAMERA_SOURCE = int(_env) if _env.isdigit() else _env # 0 = default webcam
ROI_PATH = os.getenv("POOL_ROI", "roi_child.json")
WINDOW = "PoolWatch – ROI Editor"


FILL_COLOR = (0, 255, 255) # cyan-ish
EDGE_COLOR = (0, 200, 255)
POINT_COLOR = (30, 220, 30)
TEXT_COLOR = (255, 255, 255)
ALPHA = 0.25 # polygon fill transparency
DELAY_MS = 100  # 100ms = 10 FPS (slower than real-time)

# ---------------- Helpers ----------------
def draw_overlay(frame, pts: np.ndarray | None) -> None:
    if pts is None or len(pts) < 2:
        return
    overlay = frame.copy()
    cv2.polylines(overlay, [pts], isClosed=len(pts) >= 3, color=EDGE_COLOR, thickness=2)
    if len(pts) >= 3:
        cv2.fillPoly(overlay, [pts], color=FILL_COLOR)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
    # draw points on top
    for (x,y) in pts.reshape(-1,2):
        cv2.circle(frame, (int(x),int(y)), 4, POINT_COLOR, -1, cv2.LINE_AA)

def load_roi(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            arr = np.array(data, dtype=np.int32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr
    except Exception as e:
        print(f"[WARN] Invalid ROI data: {e}")
        return None

def save_roi(path: str, pts: np.ndarray) -> None:
    data = pts.reshape(-1,2).astype(int).tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] ROI saved to {path} → {data}")

# ---------------- Main ----------------
def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {CAMERA_SOURCE}")
    # State
    current: List[Tuple[int,int]] = []
    roi_poly: np.ndarray | None = load_roi(ROI_PATH)

    # Mouse callback
    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((int(x), int(y)))

    # Setup
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    # Main loop
    print("Running ROI editor. Keys: LeftClick add | Z undo | C/Enter close | N new | S save | L load | Q quit")
    prev_t = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed – check camera/network.")
            break

        # FPS estimate (for performance sanity)
        frames += 1
        now = time.time()
        dt = now - prev_t
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            prev_t = now

        # Draw overlays
        vis = frame.copy()
        # show loaded ROI if any
        if roi_poly is not None:
            draw_overlay(vis, roi_poly)
        # show current drafting poly
        if len(current) >= 1:
            draft = np.array(current, dtype=np.int32)
            draw_overlay(vis, draft)

        # HUD
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(vis, "LeftClick=Add | Z=Undo | C/Enter=Close | N=New | S=Save | L=Load | Q=Quit", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, vis)
        
        key = cv2.waitKey(DELAY_MS) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('z'), ord('Z')):
            if current:
                current.pop()
        elif key in (ord('n'), ord('N')):
            current.clear()
            roi_poly = None
        elif key in (ord('l'), ord('L')):
            roi_poly = load_roi(ROI_PATH)
        elif key in (ord('c'), ord('C'), 13): # 13 = Enter
            if len(current) >= 3:
                roi_poly = np.array(current, dtype=np.int32)
                current.clear()
                print(f"[OK] Polygon closed with {len(roi_poly)} points.")
            else:
                print("[INFO] Need at least 3 points to close polygon.")
        elif key in (ord('s'), ord('S')):
            if roi_poly is not None and len(roi_poly) >= 3:
                save_roi(ROI_PATH, roi_poly)
            else:
                print("[INFO] No closed ROI to save. Press C/Enter to close first.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



