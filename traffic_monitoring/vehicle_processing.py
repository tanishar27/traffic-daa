from functools import lru_cache
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from congestion import classify_density

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolo26n.pt"
VIDEO_DIR = BASE_DIR / "videos"
DEFAULT_VIDEO = "trafficv.mp4"
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv"}

VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

CONF_THRESHOLD = 0.20
IMAGE_SIZE = 640
FRAME_SKIP = 2
MAX_WIDTH = 960
DEVICE = 0 if torch.cuda.is_available() else "cpu"

GREEN, RED, YELLOW, CYAN = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)
DENSITY_COLORS = {"Low": GREEN, "Medium": (0, 165, 255), "High": RED}


@lru_cache(maxsize=1)
def _model():
    return YOLO(str(MODEL_PATH))


def _text(frame, text, org, scale, color):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def list_videos():
    files = VIDEO_DIR.iterdir() if VIDEO_DIR.exists() else []
    return sorted(p.name for p in files if p.suffix.lower() in VIDEO_TYPES)


def save_video(name, data):
    VIDEO_DIR.mkdir(exist_ok=True)
    name = Path(name).name
    (VIDEO_DIR / name).write_bytes(data)
    return name


def delete_video(name):
    (VIDEO_DIR / Path(name).name).unlink(missing_ok=True)


def _run(track, video):
    video_path = VIDEO_DIR / (video or DEFAULT_VIDEO)

    for path in (MODEL_PATH, video_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open traffic video: {video_path}")

    model = _model()
    model.predictor = None
    options = dict(classes=VEHICLE_CLASSES, conf=CONF_THRESHOLD, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)

    try:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            n += 1
            if n % FRAME_SKIP:
                continue
            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                frame = cv2.resize(frame, (MAX_WIDTH, h * MAX_WIDTH // w))
            result = (model.track(frame, persist=True, tracker="bytetrack.yaml", **options) if track
                      else model(frame, **options))[0]
            yield frame, result
    finally:
        cap.release()


def detect_vehicles(video=None):
    for frame, r in _run(False, video):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            _text(frame, CLASS_NAMES[int(box.cls[0])], (x1, max(y1 - 10, 20)), 0.6, GREEN)
        _text(frame, f"Vehicles: {len(r.boxes)}", (20, 40), 1, RED)
        yield frame, len(r.boxes)


def track_vehicles(video=None):
    for _, r in _run(True, video):
        ids = r.boxes.id
        yield r.plot(), 0 if ids is None else len(ids)


def count_vehicles(video=None):
    previous, counted, total = {}, set(), 0
    counts = dict.fromkeys(CLASS_NAMES.values(), 0)

    for frame, r in _run(True, video):
        h, w = frame.shape[:2]
        line_y = h // 2
        cv2.line(frame, (0, line_y), (w, line_y), YELLOW, 3)
        _text(frame, "COUNTING LINE", (20, line_y - 10), 0.6, YELLOW)

        boxes = r.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        for (x1, y1, x2, y2), cls, tid in zip(boxes.xyxy.int().tolist(), boxes.cls.int().tolist(), ids):
            name = CLASS_NAMES[cls]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv2.circle(frame, (cx, cy), 4, RED, -1)
            _text(frame, f"{name} ID:{tid}" if tid is not None else name, (x1, max(y1 - 10, 20)), 0.5, GREEN)

            if tid is None:
                continue
            py = previous.get(tid)
            if py is not None and tid not in counted and (py < line_y <= cy or py > line_y >= cy):
                counted.add(tid)
                total += 1
                counts[name] += 1
            previous[tid] = cy

        visible = len(boxes)
        density = classify_density(visible)
        _text(frame, f"Total Count: {total}", (20, 40), 0.8, YELLOW)
        _text(frame, f"Visible Vehicles: {visible}", (20, 80), 0.7, CYAN)
        _text(frame, f"Traffic Density: {density}", (20, 120), 0.8, DENSITY_COLORS.get(density, (255, 255, 255)))

        yield frame, {"total": total, "visible": visible, "density": density, **counts}